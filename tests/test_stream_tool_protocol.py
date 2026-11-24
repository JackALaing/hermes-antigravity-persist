"""Regression tests for the <tool_call> frame, flush, and quota watchdog.

Synthetic stream-json only: no agy process, no network, no host state.
"""

from __future__ import annotations

import io
import json
import logging
import os
import shlex
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from prompt import (
    _has_unfinished_tool_call_frame,
    _is_stray_quote,
    _parse_tool_block,
    _repaired_arguments,
    _tool_parameters,
)
from stream import AntigravityStream

TOOL_CALL_CLOSE = "</tool_call>"


def _block(content: str, identifier: str = "same", name: str = "write_file") -> str:
    call = {
        "id": identifier,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps({"path": "fixture.txt", "content": content}),
        },
    }
    return f"<tool_call>{json.dumps(call)}</tool_call>"


def _events(*deltas: str, final: str = "", status: str = "SUCCESS") -> list[dict]:
    events = [
        {"event": "step_update", "step_update": {"text_delta": delta}}
        for delta in deltas
    ]
    events.append({"event": "result", "result": {"status": status, "response": final}})
    return events


def _proc(event_lines: list[str], *, poll: int | None = 0):
    proc = MagicMock()
    proc.pid = None
    proc.stdout = io.StringIO("".join(line + "\n" for line in event_lines))
    proc.stderr = io.StringIO("")
    proc._stderr_drainer = None
    proc.poll.return_value = poll
    proc.wait.return_value = 0
    return proc


def _stream(events: list[dict], *, tools: bool = True, is_worker: bool = False):
    proc = _proc([json.dumps(event) for event in events])
    client = MagicMock()
    client._isolated_gemini_dir = None
    stream = AntigravityStream(
        proc=proc,
        client=client,
        model="fixture-only",
        timeout=5.0,
        tools=[{"type": "function", "function": {"name": "write_file"}}] if tools else None,
        is_worker=is_worker,
    )
    return stream, proc, client


def _consumed(chunks):
    calls = [
        call
        for chunk in chunks
        for choice in chunk.choices
        for call in (choice.delta.tool_calls or [])
    ]
    contents = [
        choice.delta.content
        for chunk in chunks
        for choice in chunk.choices
        if choice.delta and choice.delta.content
    ]
    finishes = [
        choice.finish_reason
        for chunk in chunks
        for choice in chunk.choices
        if choice.finish_reason
    ]
    return calls, contents, finishes


def test_closing_tag_inside_json_string_does_not_split_the_call():
    block = _block("text </tool_call> text")
    # Split inside the literal sequence so the streaming loop must carry the
    # candidate close across deltas without treating it as a frame boundary.
    split_at = block.index(TOOL_CALL_CLOSE) + len(TOOL_CALL_CLOSE) - 2
    stream, _proc_mock, _client = _stream(
        _events(block[:split_at], block[split_at:])
    )

    chunks = list(stream)
    calls, contents, finishes = _consumed(chunks)

    assert len(calls) == 1
    assert calls[0].function.name == "write_file"
    assert json.loads(calls[0].function.arguments)["content"] == "text </tool_call> text"
    assert contents == []
    assert finishes == ["tool_calls"]


def test_incomplete_tool_block_fails_instead_of_becoming_assistant_text():
    stream, _proc_mock, _client = _stream(_events('<tool_call>{"id":"incomplete",'))

    chunks = []
    with pytest.raises(RuntimeError) as ctx:
        for chunk in stream:
            chunks.append(chunk)

    assert "incomplete tool call block" in str(ctx.value)
    _calls, contents, finishes = _consumed(chunks)
    assert contents == []
    assert finishes == []


def test_incomplete_second_tool_block_after_a_complete_call_fails():
    stream, _proc_mock, _client = _stream(
        _events(_block("one"), '<tool_call>{"id":"second",')
    )

    chunks = []
    with pytest.raises(RuntimeError) as ctx:
        for chunk in stream:
            chunks.append(chunk)

    assert "incomplete tool call block" in str(ctx.value)
    calls, _contents, finishes = _consumed(chunks)
    # The first, complete call was already emitted and cannot be retracted.
    assert [call.id for call in calls] == ["same"]
    assert finishes == []


def test_tool_call_indices_and_ids_are_unique_across_blocks():
    stream, _proc_mock, _client = _stream(
        _events(_block("one"), _block("two"), _block("three"))
    )

    chunks = list(stream)
    calls, _contents, finishes = _consumed(chunks)

    assert [call.index for call in calls] == [0, 1, 2]
    assert [call.id for call in calls] == ["same", "call_2", "call_3"]
    assert [json.loads(call.function.arguments)["content"] for call in calls] == [
        "one",
        "two",
        "three",
    ]
    assert finishes == ["tool_calls"]


def _init_then_slow_result(seconds: float):
    init = json.dumps({"event": "init", "conversation_id": "fixture"})
    result = json.dumps(
        {"event": "result", "result": {"status": "SUCCESS", "response": "done"}}
    )
    calls = 0

    def readline() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return init
        if calls == 2:
            time.sleep(seconds)
            return result
        return ""

    proc = MagicMock()
    proc.pid = None
    proc.stdout.readline.side_effect = readline
    proc.stderr = io.StringIO("")
    proc.poll.return_value = 0
    proc.wait.return_value = 0
    return proc


def test_watchdog_keeps_probing_after_init_and_uses_wallclock_mtime():
    proc = _init_then_slow_result(1.8)
    client = MagicMock()
    client._isolated_gemini_dir = "/fixture/gemini"
    probe_calls: list[float | None] = []

    def probe(_gemini_dir, min_mtime=None, pid=None):
        probe_calls.append(min_mtime)
        return None

    before = time.time()
    with patch("stream._check_early_quota_error", side_effect=probe):
        chunks = list(
            AntigravityStream(
                proc=proc,
                client=client,
                model="fixture-only",
                timeout=5.0,
            )
        )
    after = time.time()

    assert probe_calls, "init must not disarm the quota watchdog before content"
    # File mtimes are epoch seconds; a monotonic clock here would be orders
    # of magnitude too small and would never filter a stale log.
    for min_mtime in probe_calls:
        assert min_mtime is not None
        assert before - 2.0 <= min_mtime <= after + 2.0
    assert any(choice.delta.content == "done" for chunk in chunks for choice in chunk.choices)


def test_init_then_quota_before_content_aborts_the_turn(tmp_path: Path):
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    (log_dir / "cli-fixture.log").write_text(
        "RESOURCE_EXHAUSTED: Individual quota reached, code 429\n",
        encoding="utf-8",
    )
    proc = _init_then_slow_result(1.8)
    client = MagicMock()
    client._isolated_gemini_dir = tmp_path
    client._terminate_process = MagicMock()

    stream = AntigravityStream(
        proc=proc,
        client=client,
        model="fixture-only",
        timeout=5.0,
    )
    with pytest.raises(RuntimeError) as ctx:
        list(stream)

    assert "RESOURCE_EXHAUSTED" in str(ctx.value)
    assert "Individual quota reached" in str(ctx.value)
    client._terminate_process.assert_called()


@pytest.mark.parametrize("is_worker", [False, True])
def test_nonzero_exit_with_stale_quota_log_reports_exit_code(
    tmp_path: Path, is_worker: bool
):
    log_dir = tmp_path / "log"
    log_dir.mkdir()
    stale_log = log_dir / "cli-stale.log"
    stale_log.write_text(
        "RESOURCE_EXHAUSTED: Individual quota reached, code 429\n",
        encoding="utf-8",
    )
    old = time.time() - 86400
    os.utime(stale_log, (old, old))

    proc = _proc([], poll=1)
    client = MagicMock()
    client._isolated_gemini_dir = tmp_path
    client._terminate_process = MagicMock()

    stream = AntigravityStream(
        proc=proc,
        client=client,
        model="fixture-only",
        timeout=5.0,
        is_worker=is_worker,
    )
    with pytest.raises(RuntimeError) as ctx:
        list(stream)

    message = str(ctx.value)
    assert "Antigravity execution failed" in message
    assert "return code 1" in message
    assert "RESOURCE_EXHAUSTED" not in message


# ---------------------------------------------------------------------------
# result.response uses the same frame parser as streamed deltas
# ---------------------------------------------------------------------------


def test_complete_tool_call_only_in_result_response_is_parsed():
    stream, _proc_mock, _client = _stream(_events(final=_block("valid")))

    calls, contents, finishes = _consumed(list(stream))

    assert [call.function.name for call in calls] == ["write_file"]
    assert json.loads(calls[0].function.arguments)["content"] == "valid"
    assert contents == []
    assert finishes == ["tool_calls"]


def test_result_only_call_with_literal_closing_tag_in_arguments_is_parsed():
    stream, _proc_mock, _client = _stream(
        _events(final=_block("text </tool_call> text"))
    )

    calls, contents, finishes = _consumed(list(stream))

    assert len(calls) == 1
    assert json.loads(calls[0].function.arguments)["content"] == (
        "text </tool_call> text"
    )
    assert contents == []
    assert finishes == ["tool_calls"]


def test_two_complete_tool_calls_only_in_result_response_are_unique():
    stream, _proc_mock, _client = _stream(
        _events(final=_block("one") + _block("two"))
    )

    calls, contents, finishes = _consumed(list(stream))

    assert [call.index for call in calls] == [0, 1]
    assert [call.id for call in calls] == ["same", "call_2"]
    assert [json.loads(call.function.arguments)["content"] for call in calls] == [
        "one",
        "two",
    ]
    assert contents == []
    assert finishes == ["tool_calls"]


def test_result_only_plain_text_with_tools_stays_content():
    stream, _proc_mock, _client = _stream(_events(final="all done"))

    calls, contents, finishes = _consumed(list(stream))

    assert calls == []
    assert contents == ["all done"]
    assert finishes == ["stop"]


@pytest.mark.parametrize(
    "final, expected",
    [(_block("valid"), 1), (_block("one") + _block("two"), 2)],
)
def test_result_only_tool_calls_still_parse_without_tools_configured(final, expected):
    stream, _proc_mock, _client = _stream(_events(final=final), tools=False)

    calls, contents, finishes = _consumed(list(stream))

    assert len(calls) == expected
    assert contents == []
    assert finishes == ["tool_calls"]


def test_result_only_prose_mentioning_tag_stays_text_without_tools():
    response = "Use <tool_call> to call tools."
    stream, _proc_mock, _client = _stream(_events(final=response), tools=False)

    calls, contents, finishes = _consumed(list(stream))

    assert calls == []
    assert contents == [response]
    assert finishes == ["stop"]


@pytest.mark.parametrize(
    "obj",
    [
        {},
        {"function": {"name": "", "arguments": "{}"}},
        {"function": {"name": "   ", "arguments": "{}"}},
        {"function": {"name": 123, "arguments": "{}"}},
        {"function": "not-a-dict"},
    ],
)
def test_tool_objects_without_a_nonblank_function_name_are_rejected(obj):
    calls, _text = _parse_tool_block(f"<tool_call>{json.dumps(obj)}</tool_call>")

    assert calls == []


def test_tool_call_function_name_is_stripped():
    body = '{"function": {"name": "  write_file  ", "arguments": "{}"}}'

    calls, _text = _parse_tool_block(f"<tool_call>{body}</tool_call>")

    assert [call.function.name for call in calls] == ["write_file"]


@pytest.mark.parametrize("identifier", [None, 123, "", "   "])
def test_tool_call_blank_or_non_string_id_falls_back(identifier):
    obj = {"id": identifier, "function": {"name": "write_file", "arguments": "{}"}}

    calls, _text = _parse_tool_block(f"<tool_call>{json.dumps(obj)}</tool_call>")

    assert [call.id for call in calls] == ["call_1"]


def test_tool_call_string_id_is_preserved():
    obj = {"id": "call_x", "function": {"name": "write_file", "arguments": "{}"}}

    calls, _text = _parse_tool_block(f"<tool_call>{json.dumps(obj)}</tool_call>")

    assert [call.id for call in calls] == ["call_x"]


# ---------------------------------------------------------------------------
# A stream cut inside any part of a frame fails closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tail", ["<tool_cal", "<tool_call"])
def test_stream_cut_inside_opening_tag_after_a_complete_call_fails(tail):
    stream, _proc_mock, _client = _stream(_events(_block("one"), tail))

    chunks = []
    with pytest.raises(RuntimeError) as ctx:
        for chunk in stream:
            chunks.append(chunk)

    assert "incomplete tool call block" in str(ctx.value)
    calls, contents, finishes = _consumed(chunks)
    assert [call.id for call in calls] == ["same"]
    assert contents == []
    assert finishes == []


@pytest.mark.parametrize("tail", ["<tool_cal", "<tool_call"])
def test_stream_cut_inside_opening_tag_without_a_call_fails(tail):
    stream, _proc_mock, _client = _stream(_events(tail))

    chunks = []
    with pytest.raises(RuntimeError) as ctx:
        for chunk in stream:
            chunks.append(chunk)

    assert "incomplete tool call block" in str(ctx.value)
    calls, contents, finishes = _consumed(chunks)
    assert calls == []
    assert contents == []
    assert finishes == []


def test_stream_cut_inside_closing_tag_after_a_complete_call_fails():
    partial = (
        '<tool_call>{"id":"b","function":{"name":"write_file",'
        '"arguments":"{}"}</tool_ca'
    )
    stream, _proc_mock, _client = _stream(_events(_block("one"), partial))

    chunks = []
    with pytest.raises(RuntimeError) as ctx:
        for chunk in stream:
            chunks.append(chunk)

    assert "incomplete tool call block" in str(ctx.value)
    calls, _contents, finishes = _consumed(chunks)
    assert [call.id for call in calls] == ["same"]
    assert finishes == []


# ---------------------------------------------------------------------------
# Native tool neutralization: security abort surfaced as explanatory content
# (issue #52) instead of a bare empty completion that read as "no content"
# and sent Hermes down the nudge/retry/silent-fallback path. Buffered frame
# text is still never flushed after the abort.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("partial", ['<tool_call>{"id":"x",', "<tool_cal"])
def test_native_tool_step_neutralization_with_partial_block_does_not_fail(partial):
    events = [
        {"event": "step_update", "step_update": {"text_delta": partial}},
        {
            "event": "step_update",
            "step_update": {"step_type": "tool", "tool_name": "run_command"},
        },
    ]
    stream, _proc_mock, client = _stream(events)

    chunks = list(stream)

    calls, contents, finishes = _consumed(chunks)
    assert calls == []
    # The partial frame prefix stays buffered (never flushed), but the
    # explanatory content chunk is emitted (issue #52).
    assert len(contents) == 1
    assert "blocked by host security policy" in contents[0]
    assert "run_command" in contents[0]
    assert finishes == ["stop"]
    client._terminate_process.assert_called()


# ---------------------------------------------------------------------------
# A closed but unparseable frame is invalid protocol, never assistant text
# ---------------------------------------------------------------------------

# One representative per rejected shape: broken JSON, absent function, blank
# function name, non-object function, non-object top level, empty body.
UNPARSEABLE_FRAME_BODIES = [
    '{"id":"x","function":{"name":"write_file","arguments":"{"}',
    '{"id":"x"}',
    '{"function":{"name":"   ","arguments":"{}"}}',
    '{"function":[]}',
    "[]",
    "",
]


def _consume_until_error(stream):
    chunks = []
    with pytest.raises(RuntimeError) as ctx:
        for chunk in stream:
            chunks.append(chunk)
    return _consumed(chunks), str(ctx.value)


@pytest.mark.parametrize("body", UNPARSEABLE_FRAME_BODIES)
def test_closed_unparseable_frame_falls_back_to_text_in_stream(body):
    raw = f"<tool_call>{body}</tool_call>"
    stream, _proc_mock, _client = _stream(_events(raw))

    calls, contents, finishes = _consumed(list(stream))

    assert calls == []
    assert "".join(contents) == raw
    assert finishes == ["stop"]


@pytest.mark.parametrize("body", UNPARSEABLE_FRAME_BODIES)
def test_closed_unparseable_frame_in_result_response_falls_back_to_text(body):
    raw = f"<tool_call>{body}</tool_call>"
    stream, _proc_mock, _client = _stream(_events(final=raw))

    calls, contents, finishes = _consumed(list(stream))

    assert calls == []
    assert "".join(contents) == raw
    assert finishes == ["stop"]


@pytest.mark.parametrize("body", UNPARSEABLE_FRAME_BODIES)
def test_closed_unparseable_frame_after_a_complete_call_in_stream(body):
    stream, _proc_mock, _client = _stream(
        _events(_block("one"), f"<tool_call>{body}</tool_call>")
    )

    calls, contents, finishes = _consumed(list(stream))

    assert [call.id for call in calls] == ["same"]
    assert contents == []
    assert finishes == ["tool_calls"]


def test_closed_unparseable_frame_after_a_complete_call_in_result_response():
    stream, _proc_mock, _client = _stream(
        _events(final=_block("one") + "<tool_call>{}</tool_call>")
    )

    calls, contents, finishes = _consumed(list(stream))

    assert [call.id for call in calls] == ["same"]
    assert contents == []
    assert finishes == ["tool_calls"]


def test_tool_call_tag_inside_markdown_code_streams_as_text():
    prose = (
        "Example:\n\n```xml\n"
        "<tool_call>{\"bad\": \"json\"</tool_call>\n"
        "```\n\nDone."
    )
    stream, _proc_mock, _client = _stream(_events(prose))

    calls, contents, finishes = _consumed(list(stream))

    assert calls == []
    assert "".join(contents) == prose
    assert finishes == ["stop"]


def test_post_loop_incomplete_tool_call_inside_open_fence_is_plain_text():
    stream, _proc_mock, _client = _stream(
        _events("```xml\n<tool", "_call>{\"unclosed\": true")
    )

    calls, contents, finishes = _consumed(list(stream))

    assert calls == []
    assert "<tool_call>{\"unclosed\": true" in "".join(contents)
    assert finishes == ["stop"]


def test_post_loop_unparseable_tool_call_inside_open_fence_no_warning(caplog):
    stream, _proc_mock, _client = _stream(
        _events("```xml\n", "<tool_call>{}</tool_call>")
    )

    with caplog.at_level("WARNING"):
        calls, contents, finishes = _consumed(list(stream))

    assert calls == []
    assert "<tool_call>{}</tool_call>" in "".join(contents)
    assert finishes == ["stop"]
    assert not any("Unparseable <tool_call>" in r.message for r in caplog.records)


def test_indented_code_block_ignored_and_escaped_backtick_parsed():
    stream, _proc_mock, _client = _stream(
        _events(
            "    <tool_call>{bad}</tool_call>\n\n",
            f"\\`{_block('one')}\\`",
        )
    )

    calls, contents, finishes = _consumed(list(stream))

    assert [call.id for call in calls] == ["same"]
    assert "<tool_call>{bad}</tool_call>" in "".join(contents)
    assert finishes == ["tool_calls"]


def test_commonmark_edge_cases_in_fences_and_headings():
    stream, _proc_mock, _client = _stream(
        _events(
            "# `<tool_call>{bad}</tool_call>` #\n\n",
            "```xml\n    ```\n<tool_call>{still_fenced}</tool_call>\n```\n",
            "```xml\n``` extra\n<tool_call>{still_fenced_2}</tool_call>\n```",
        )
    )

    calls, contents, finishes = _consumed(list(stream))
    joined = "".join(contents)
    assert calls == []
    assert "<tool_call>{bad}</tool_call>" in joined
    assert "<tool_call>{still_fenced}</tool_call>" in joined
    assert "<tool_call>{still_fenced_2}</tool_call>" in joined
    assert finishes == ["stop"]


# ---------------------------------------------------------------------------
# A lone "<" ending prose is text, a longer tag prefix remains a frame
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, unfinished",
    [
        ("Compare a <", False),
        ("Compare a <t", True),
        ("Compare a <tool_cal", True),
        ("Compare a <tool_call", True),
        ('<tool_call>{"id":"x",', True),
    ],
)
def test_lone_angle_bracket_is_text_but_longer_prefixes_are_frames(text, unfinished):
    assert _has_unfinished_tool_call_frame(text) is unfinished


@pytest.mark.parametrize("tools", [True, False])
def test_prose_ending_with_lone_angle_bracket_stays_content(tools):
    stream, _proc_mock, _client = _stream(_events("Compare a <"), tools=tools)

    calls, contents, finishes = _consumed(list(stream))

    assert calls == []
    assert "".join(contents) == "Compare a <"
    assert finishes == ["stop"]


def test_prose_ending_with_lone_angle_bracket_after_a_complete_call_succeeds():
    stream, _proc_mock, _client = _stream(_events(_block("one"), "Compare a <"))

    calls, _contents, finishes = _consumed(list(stream))

    assert [call.id for call in calls] == ["same"]
    assert finishes == ["tool_calls"]


# -- quote-escape repair of tool-call arguments ---------------------------------
# Shell and Python quote escapes the model wrote into JSON strings without
# escaping them for JSON (News 08:17-08:20 and Chief 18:17, 2026-11-21). Each
# excerpt keeps the real call's shape; Hermes replaced the originals with {}.

def _args_after_parse(arguments: str, name: str = "terminal", tool_params=None) -> str:
    obj = {"id": "call_1", "type": "function", "function": {"name": name, "arguments": arguments}}
    calls, _text = _parse_tool_block(f"<tool_call>{json.dumps(obj)}</tool_call>", tool_params=tool_params)
    assert len(calls) == 1
    return calls[0].function.arguments


def _curator_payload(command: str) -> dict:
    argv = shlex.split(command)
    assert argv[:4] == ["python3", "/tmp/curator.py", "render-native", "--input-json"]
    assert len(argv) == 5, argv
    return json.loads(argv[4])


_NEWS_CASES = [
    pytest.param(  # 08:17 '\''
        r"""{"command": "python3 /tmp/curator.py render-native --input-json '{\"what\":\"Caveat: this is Berkopec'\''s report of an OpenRouter email.\",\"why\":\"Charlie O'\''Neill shares a lesson; the authors'\'' own claims.\"}'"}""",
        r"""python3 /tmp/curator.py render-native --input-json '{"what":"Caveat: this is Berkopec'\''s report of an OpenRouter email.","why":"Charlie O'\''Neill shares a lesson; the authors'\'' own claims."}'""",
        {"what": "Caveat: this is Berkopec's report of an OpenRouter email.",
         "why": "Charlie O'Neill shares a lesson; the authors' own claims."},
        id="news-0817",
    ),
    pytest.param(  # 08:18 '\''
        r"""{"command":"python3 /tmp/curator.py render-native --input-json '{\"what\":\"Developer leoshen0 reports that Antigravity'\''s built-in prompt drives heavy upfront reads.\",\"why\":\"Based on the author'\''s report of an OpenRouter email.\"}'"}""",
        r"""python3 /tmp/curator.py render-native --input-json '{"what":"Developer leoshen0 reports that Antigravity'\''s built-in prompt drives heavy upfront reads.","why":"Based on the author'\''s report of an OpenRouter email."}'""",
        {"what": "Developer leoshen0 reports that Antigravity's built-in prompt drives heavy upfront reads.",
         "why": "Based on the author's report of an OpenRouter email."},
        id="news-0818",
    ),
    pytest.param(  # 08:19 '"'"', next to escaped quotes
        r"""{"command":"python3 /tmp/curator.py render-native --input-json '{\"what\":\"Decentralized Economy argues that Pearl'"'"'s \\\"useful work\\\" mining model is broken.\",\"why\":\"Verify OpenRouter'"'"'s official notice; the authors'"'"' own framing.\"}'"}""",
        r"""python3 /tmp/curator.py render-native --input-json '{"what":"Decentralized Economy argues that Pearl'"'"'s \"useful work\" mining model is broken.","why":"Verify OpenRouter'"'"'s official notice; the authors'"'"' own framing."}'""",
        {"what": 'Decentralized Economy argues that Pearl\'s "useful work" mining model is broken.',
         "why": "Verify OpenRouter's official notice; the authors' own framing."},
        id="news-0819",
    ),
    pytest.param(  # 08:20 '\''
        r"""{"command": "python3 /tmp/curator.py render-native --input-json '{\"what\":\"Hermes Release Watch highlights Nous Research'\''s official Advanced Delegation plugin.\",\"why\":\"Caveat: this is Berkopec'\''s report.\"}'"}""",
        r"""python3 /tmp/curator.py render-native --input-json '{"what":"Hermes Release Watch highlights Nous Research'\''s official Advanced Delegation plugin.","why":"Caveat: this is Berkopec'\''s report."}'""",
        {"what": "Hermes Release Watch highlights Nous Research's official Advanced Delegation plugin.",
         "why": "Caveat: this is Berkopec's report."},
        id="news-0820",
    ),
]


@pytest.mark.parametrize("arguments, command, payload", _NEWS_CASES)
def test_shell_quote_idioms_in_terminal_arguments_are_repaired(arguments, command, payload, caplog):
    with pytest.raises(ValueError):
        json.loads(arguments)

    with caplog.at_level(logging.INFO, logger="prompt"):
        repaired = _args_after_parse(arguments)

    # The decoded command is exactly the one the model wrote, and the shell
    # reads each idiom as an apostrophe inside the single-quoted payload.
    assert json.loads(repaired) == {"command": command}
    assert _curator_payload(command) == payload
    assert [r.getMessage() for r in caplog.records] == [
        f"Repaired unescaped quotes or backslashes in terminal tool-call arguments ({len(arguments)} chars)"]


def test_python_quote_escape_in_execute_code_arguments_is_repaired():
    # Chief 18:17: \' inside a single-quoted Python string, where a bare '
    # would end the string. The decoded source keeps \' and Python reads '.
    arguments = (r"""{"code":"line = ('\\t- Draft for Jack\'s approval. Awaiting Jack\'s approval on Dispatch.'"""
                 r""" + ' (Workhorse research)')\nprint(line)\n"}""")
    with pytest.raises(ValueError):
        json.loads(arguments)

    code = json.loads(_args_after_parse(arguments, "execute_code"))["code"]

    assert code == ("line = ('\\t- Draft for Jack\\'s approval. Awaiting Jack\\'s approval on Dispatch.'"
                    " + ' (Workhorse research)')\nprint(line)\n")
    scope: dict = {}
    exec(compile(code.replace("print(line)", ""), "<case>", "exec"), scope)
    assert scope["line"] == "\t- Draft for Jack's approval. Awaiting Jack's approval on Dispatch. (Workhorse research)"


@pytest.mark.parametrize(
    "value",
    [
        "echo 'it'\\''s'",          # '\\'' : an escaped backslash, then a quote
        "echo 'it'\"'\"'s'",        # '\"'\"' : already escaped
        "print('Jack\\'s')",        # \\' : an escaped backslash
        "C:\\\\path\\\\'x'",
        "plain text with ' and \" and \\",
        "",
    ],
)
def test_valid_arguments_are_never_changed(value, caplog):
    for arguments in (json.dumps({"command": value}), json.dumps({"command": value}, ensure_ascii=False, indent=2),
                      json.dumps([value, {"k": value}]), '{"command":' + json.dumps(value) + "}"):
        with caplog.at_level(logging.INFO, logger="prompt"):
            assert _args_after_parse(arguments) == arguments
    assert caplog.records == []


@pytest.mark.parametrize(
    "arguments",
    [
        r"""{\'command\': \'ls\'}""",                           # \' outside any string
        r"""{'command': 'ls'}""",                               # Python dict syntax
        r"""{"command":"echo it\'s""",                          # truncated
        r"""{"command":"echo it\'s",}x""",                      # still invalid after the repair
        r"""{"code":"print("a", "b")"}""",                      # bare quotes next to JSON punctuation
        r"""{"code":"d = {"a": 1}"}""",                         # could be structure: left alone
        r"""{"code":"x = '",'"}""",                              # a quote before a comma
        r"""{"a": 1, '"'"' : 2}""",                              # the idiom outside a string
        "{\"code\":\"x = 'Jack\\'s'\n\"}",                       # \' plus a raw newline
        "not json at all",
        "",
    ],
)
def test_unrepairable_arguments_pass_through_untouched(arguments, caplog):
    with caplog.at_level(logging.INFO, logger="prompt"):
        assert _args_after_parse(arguments) == arguments
    assert caplog.records == []


# -- stray double quotes in tool-call arguments ---------------------------------
# News 07:34-07:47 (2026-11-23): Python source in execute_code where the model
# escaped every double quote for JSON except the one inside a single-quoted
# literal ('"' and '    "'). Each excerpt is cut verbatim from the real
# arguments Hermes replaced with {}; the expected code is the literal reading.

_NEWS_1123_CASES = [
    pytest.param(
        '{"code": "        # Body fragments\\n        m_body = re.search(r\'^### Resolved body.*?\\\\n((?:>.*?\\\\n)*)((?:    .+\\\\n?)+)\', blk, re.M)\\n        body_parts = []\\n        if m_body:\\n            for l in m_body.group(2).splitlines():\\n                s = l.strip()\\n                if s.startswith(\'"\'):\\n                    try: body_parts.append(json.loads(s))\\n                    except Exception: pass\\n        body = \'\'.join(body_parts)\\nprint(\'=== ALL 16 NEWSLETTERS ===\')\\nfor i, d in enumerate(all_docs, 1):\\n    print(f\\"{i:02d}. [{d[\'pub\']}] {d[\'site\']} | {d[\'id\']} | {len(d[\'clean\']):5d} chars | {d[\'title\']}\\")\\n"}',
        '        # Body fragments\n        m_body = re.search(r\'^### Resolved body.*?\\n((?:>.*?\\n)*)((?:    .+\\n?)+)\', blk, re.M)\n        body_parts = []\n        if m_body:\n            for l in m_body.group(2).splitlines():\n                s = l.strip()\n                if s.startswith(\'"\'):\n                    try: body_parts.append(json.loads(s))\n                    except Exception: pass\n        body = \'\'.join(body_parts)\nprint(\'=== ALL 16 NEWSLETTERS ===\')\nfor i, d in enumerate(all_docs, 1):\n    print(f"{i:02d}. [{d[\'pub\']}] {d[\'site\']} | {d[\'id\']} | {len(d[\'clean\']):5d} chars | {d[\'title\']}")\n',
        id='news-0734',
    ),
    pytest.param(
        '{"code":"            for line in fm.group(1).splitlines():\\n                s = line.strip()\\n                if s.startswith(\'"\'):\\n                    try:\\n                        chunks.append(json.loads(s))\\n                    except Exception:\\n                        chunks.append(s)\\n            return \'\'.join(chunks)\\n"}',
        '            for line in fm.group(1).splitlines():\n                s = line.strip()\n                if s.startswith(\'"\'):\n                    try:\n                        chunks.append(json.loads(s))\n                    except Exception:\n                        chunks.append(s)\n            return \'\'.join(chunks)\n',
        id='news-0740',
    ),
    pytest.param(
        '{"code":"        if in_field:\\n            if l.startswith(\'    "\'):\\n                try:\\n                    out.append(json.loads(l.strip()))\\n                except Exception:\\n                    out.append(l.strip())\\n            elif l.startswith(\'    null\'):\\n                return None\\n            else:\\n                break\\n    print(f\'\\\\n--- [{c[\\"watchlist_id\\"]}] {c[\\"watchlist_label\\"]} ---\')\\n"}',
        '        if in_field:\n            if l.startswith(\'    "\'):\n                try:\n                    out.append(json.loads(l.strip()))\n                except Exception:\n                    out.append(l.strip())\n            elif l.startswith(\'    null\'):\n                return None\n            else:\n                break\n    print(f\'\\n--- [{c["watchlist_id"]}] {c["watchlist_label"]} ---\')\n',
        id='news-0746',
    ),
    pytest.param(
        '{"code":"        if capturing:\\n            if line.startswith(\'    "\'):\\n                try:\\n                    out.append(json.loads(line.strip()))\\n                except Exception:\\n                    out.append(line.strip())\\n            elif line.startswith(\'    null\'):\\n                return None\\n            elif line.startswith(\'> \'):\\n                continue\\n            else:\\n                break\\n"}',
        '        if capturing:\n            if line.startswith(\'    "\'):\n                try:\n                    out.append(json.loads(line.strip()))\n                except Exception:\n                    out.append(line.strip())\n            elif line.startswith(\'    null\'):\n                return None\n            elif line.startswith(\'> \'):\n                continue\n            else:\n                break\n',
        id='news-0747',
    ),
]


@pytest.mark.parametrize("arguments, code", _NEWS_1123_CASES)
def test_stray_double_quotes_in_execute_code_are_repaired(arguments, code, caplog):
    with pytest.raises(ValueError):
        json.loads(arguments)

    with caplog.at_level(logging.INFO, logger="prompt"):
        repaired = _args_after_parse(arguments, "execute_code")

    assert json.loads(repaired) == {"code": code}
    assert "startswith('\"')" in code or "startswith('    \"')" in code
    assert [r.getMessage() for r in caplog.records] == [
        f"Repaired unescaped quotes or backslashes in execute_code tool-call arguments ({len(arguments)} chars)"]


@pytest.mark.parametrize(
    "arguments, value",
    [
        (r"""{"command":"echo 'it'"'"s'"}""", "echo 'it'\"'\"s'"),     # '"'" without the closing quote
        (r"""{"code":"print("hi")"}""", 'print("hi")'),                 # both quotes stray
        (r"""{"code":"x = '\\"'"}""", "x = '\\\"'"),                   # an escaped backslash, then a bare quote
        (r"""{"code":"m = re.match('"'), s\'"}""", "m = re.match('\"'), s\\'"),  # with \' in the same value
    ],
)
def test_stray_quotes_decode_to_the_text_written(arguments, value):
    with pytest.raises(ValueError):
        json.loads(arguments)
    assert list(json.loads(_args_after_parse(arguments)).values()) == [value]


@pytest.mark.parametrize("arguments, code", _NEWS_1123_CASES)
def test_valid_json_never_has_a_stray_quote(arguments, code):
    values = [code, "echo 'it'\"'\"s'", 'a "quoted" word', '\\"', '"', "", "{\"k\": [\"v\"]}"]
    for doc in (json.dumps({"code": code}), json.dumps({"code": code}, indent=2),
                json.dumps({"a": values, "b": {"c": values}}), json.dumps(values, ensure_ascii=False),
                json.dumps(values, separators=(",", ":"))):
        assert [i for i, ch in enumerate(doc) if ch == '"' and _is_stray_quote(doc, i)] == []


# -- invalid escapes in tool-call arguments ----------------------------------
# News 22:33-22:51 (2026-11-23): Python regexes in execute_code with their
# backslashes left single (\(, \), \s, \.), which are not JSON escapes.
# Excerpts are cut from the real arguments Hermes replaced with {} (paths
# shortened). Each such backslash is doubled, so it decodes as written.

_NEWS_1123_ESCAPE_CASES = [
    pytest.param(
        '{"code":"for i in range(3):\\n    log_path = f\'/tmp/task-{i}.log\'\\n    text = open(log_path, errors=\'replace\').read()\\n    # Check how task logs are structured\\n    m = re.search(r\'(?m)^# AI News|# Readwise|## 1\\.\', text)\\n    start = m.start() if m else max(0, len(text) - 8000)\\n    summary = text[start:]\\n"}',
        "    m = re.search(r'(?m)^# AI News|# Readwise|## 1\\.', text)\n",
        id="news-2240",
    ),
    pytest.param(
        '{"code":"cron_log = open(\'/tmp/cron.md\').read()\\nshas = re.findall(r\'\\"sha256\\":\\s*\\"([0-9a-f]{64})\\"\', cron_log)\\nprint(\'sha256s in cron log:\', list(dict.fromkeys(shas)))\\n"}',
        "shas = re.findall(r'\"sha256\":\\s*\"([0-9a-f]{64})\"', cron_log)\n",
        id="news-2248",
    ),
    pytest.param(
        '{"code":"cron_log = open(\'/tmp/cron.md\').read()\\nshas = re.findall(r\'\\"sha256\\":\\s*\\"([0-9a-f]{64})\\"\', cron_log)\\nprint(\'=== sha256s in morning cron log ===\', shas[:10])\\n"}',
        "shas = re.findall(r'\"sha256\":\\s*\"([0-9a-f]{64})\"', cron_log)\n",
        id="news-2251",
    ),
]


@pytest.mark.parametrize("arguments, line", _NEWS_1123_ESCAPE_CASES)
def test_regex_backslashes_in_execute_code_are_repaired(arguments, line, caplog):
    with pytest.raises(ValueError):
        json.loads(arguments)

    with caplog.at_level(logging.INFO, logger="prompt"):
        repaired = _args_after_parse(arguments, "execute_code", _tool_parameters(_HERMES_TOOLS))

    code = json.loads(repaired)["code"]
    assert line in code
    compile(code, "<case>", "exec")
    assert [r.getMessage() for r in caplog.records] == [
        f"Repaired unescaped quotes or backslashes in execute_code tool-call arguments ({len(arguments)} chars)"]


def test_valid_escapes_keep_their_json_meaning_beside_repaired_ones():
    # News 22:33: the regex backslashes are repaired, but the model also
    # wrote Python's \n escapes single (r'\n{3,}', '\n\n', f'\n...'), the
    # same characters as the JSON \n it used for line breaks. They decode as
    # newlines, so this source does not compile; nothing tells which \n the
    # model meant, and Python reports the error instead of running it.
    arguments = '{"code":"def summarize_md(path, label, max_chars=2600):\\n    t = open(path).read()\\n    # Strip huge tracking URLs to keep text dense and readable\\n    t_clean = re.sub(r\'\\(https?://[^\\)\\s]{60,}\\)\', \'(URL)\', t)\\n    t_clean = re.sub(r\'\\n{3,}\', \'\\n\\n\', t_clean)\\n    print(f\'\\n================ {label} (orig={len(t)}) ================\')\\n    print(t_clean[:max_chars])\\n"}'
    code = json.loads(_args_after_parse(arguments, "execute_code"))["code"]

    assert "    t_clean = re.sub(r'\\(https?://[^\\)\\s]{60,}\\)', '(URL)', t)\n" in code
    assert "    t_clean = re.sub(r'\n{3,}', '\n\n', t_clean)\n" in code
    with pytest.raises(SyntaxError):
        compile(code, "<case>", "exec")


@pytest.mark.parametrize(
    "arguments, value",
    [
        (r"""{"command":"echo it\'s \q"}""", "echo it\\'s \\q"),
        (r"""{"command":"dir C:\users\me"}""", "dir C:\\users\\me"),      # \u without four hex digits
        (r"""{"command":"echo \u12"}""", "echo \\u12"),
        (r"""{"command":"grep -E '\s\u00e9\d'"}""", "grep -E '\\s\u00e9\\d'"),  # \u00e9 is valid: é
        (r"""{"command":"x\(\n\t\/\\\"y"}""", 'x\\(\n\t/\\"y'),     # only \( is doubled
    ],
)
def test_invalid_escapes_decode_to_the_text_written(arguments, value):
    with pytest.raises(ValueError):
        json.loads(arguments)
    assert json.loads(_args_after_parse(arguments)) == {"command": value}


@pytest.mark.parametrize(
    "arguments",
    [
        # News 22:40 and 23:01: bare quotes beside a comma or a space could be
        # structure, so these stay unparseable.
        '{"code":"for i, x in enumerate(rss.get(\'entries\', []), 1):\\n    s = (x.get(\'summary\') or x.get(\'description\') or x.get(\'excerpt\') or \'\').replace(\'\\n\', \' \')[:260]\\n    print(f\'{i:2d}. [{x.get("source_id")}] {x.get("title")} | ref={x.get("ref_id","")[:12]} | {s}\')\\n"}',
        '{"code":"print(f\'Removed {removed_x} accepted X lines; x_decisions count: {len(x_decisions)} (accept={sum(v=="accept" for v in x_decisions.values())}, defer={sum(v=="defer" for v in x_decisions.values())})\')\\n"}',
        r"""{"code":"re.findall(r'\b\d{4}\b', s)"}""",     # \b beside a doubled escape: a regex boundary
        r"""{"command":"echo \q"} trailing""",
        r"""{"command":"echo \q""",                          # truncated
        "{\"command\":\"echo \\q\n\"}",                     # \q plus a raw newline
        "{\"command\":\"echo \\",                           # ends inside an escape
    ],
)
def test_escape_near_misses_pass_through_untouched(arguments, caplog):
    with caplog.at_level(logging.INFO, logger="prompt"):
        assert _args_after_parse(arguments, "execute_code") == arguments
    assert caplog.records == []


# -- the key guard on repairs --------------------------------------------------
# A repair that parses with the wrong structure shows it in its keys.

_HERMES_TOOLS = [
    {"type": "function", "function": {"name": "terminal", "parameters": {
        "type": "object", "properties": {"command": {}, "timeout": {}, "workdir": {}}}}},
    {"type": "function", "function": {"name": "execute_code", "parameters": {
        "type": "object", "properties": {"code": {}, "reset": {}}}}},
    {"type": "function", "function": {"name": "no_schema"}},
]

_WRONG_STRUCTURE = r"""{"command":"s = \'", "code": "\'\nif s.startswith(\'"\'):","timeout":60}"""


def test_tool_parameters_reads_listed_properties_only():
    assert _tool_parameters(_HERMES_TOOLS + [None, {"function": "x"}, {"name": "flat", "parameters": {
        "properties": {"a": {}}}}]) == {
        "terminal": frozenset({"command", "timeout", "workdir"}),
        "execute_code": frozenset({"code", "reset"}),
        "flat": frozenset({"a"}),
    }
    assert _tool_parameters(None) == {}


def test_repair_with_a_key_outside_the_schema_is_declined(caplog):
    params = _tool_parameters(_HERMES_TOOLS)
    with caplog.at_level(logging.INFO, logger="prompt"):
        assert _args_after_parse(_WRONG_STRUCTURE, "terminal", params) == _WRONG_STRUCTURE
    assert [r.getMessage() for r in caplog.records] == [
        f"Declined an escape repair of terminal tool-call arguments ({len(_WRONG_STRUCTURE)} chars): "
        "its keys do not fit the tool"]
    # Without a schema for the tool, identifier-shaped keys are all it can check.
    assert json.loads(_args_after_parse(_WRONG_STRUCTURE, "no_schema", params)).keys() == {
        "command", "code", "timeout"}


@pytest.mark.parametrize(
    "arguments",
    [
        r"""{"code":"a = \'", "code": "\'\nif s.startswith(\'"\'):"}""",   # a key read twice
        r"""{"command":"echo it\'s","env":{"A":"1","A":"2"}}""",          # a nested duplicate
        r"""{"code":"x = \'", "a b": "\'\nif s.startswith(\'"\'):"}""",   # not identifier-shaped
        r"""["echo it\'s"]""",                                           # not an object
    ],
)
def test_repairs_with_implausible_keys_are_declined(arguments):
    with pytest.raises(ValueError):
        json.loads(arguments)
    assert _args_after_parse(arguments, "no_schema") == arguments


def test_stream_checks_repairs_against_the_request_tools():
    def call(name, arguments):
        return "<tool_call>" + json.dumps(
            {"id": name, "type": "function", "function": {"name": name, "arguments": arguments}}) + "</tool_call>"

    real = _NEWS_1123_ESCAPE_CASES[0].values[0]
    stream = AntigravityStream(
        proc=_proc([json.dumps(e) for e in _events(call("terminal", _WRONG_STRUCTURE), call("execute_code", real))]),
        client=MagicMock(_isolated_gemini_dir=None), model="fixture-only", timeout=5.0, tools=_HERMES_TOOLS,
    )

    calls, _contents, _finishes = _consumed(list(stream))

    assert calls[0].function.arguments == _WRONG_STRUCTURE
    assert "re.search(r'(?m)^# AI News|# Readwise|## 1\\.', text)" in json.loads(calls[1].function.arguments)["code"]


# -- fuzz: valid JSON is never changed, invalid escapes decode as written -----

_FUZZ_ALPHABET = ['"', "'", "\\", "\\n", "\\u", "\\b", " ", "\t", "\n", ",", ":", "{", "}", "[", "]",
                  "a", "s", "(", ")", ".", "u", "0", "e", "\x01", "\x0c", "é", "\ud83d"]


def test_fuzz_valid_json_arguments_are_never_changed():
    import random

    rng = random.Random(2311)

    def text():
        return "".join(rng.choice(_FUZZ_ALPHABET) for _ in range(rng.randint(0, 12)))

    def value(depth=0):
        roll = rng.random()
        if depth > 3 or roll < 0.5:
            return text()
        if roll < 0.6:
            return rng.choice([0, -1.5, True, None])
        if roll < 0.8:
            return [value(depth + 1) for _ in range(rng.randint(0, 3))]
        return {text(): value(depth + 1) for _ in range(rng.randint(0, 3))}

    for _ in range(3000):
        obj = {"command": value(), text(): value()}
        for doc in (json.dumps(obj), json.dumps(obj, ensure_ascii=False, indent=rng.choice([None, 2])),
                    json.dumps(obj, separators=(",", ":")), json.dumps([obj, value()])):
            assert _repaired_arguments("terminal", doc, _tool_parameters(_HERMES_TOOLS)) == doc


def test_fuzz_single_backslashes_before_non_escapes_decode_as_written():
    import random

    rng = random.Random(2312)
    pieces = ["a", " ", "(", ")", ".", "'", "x = 1", "\\(", "\\)", "\\s", "\\d", "\\.", "\\'", "\\w",
              "\\$", "\\|", "\\[", "\\]", "\\{", "\\}", "\\-", "\\ ", "\\user", "\\g<1>", "é"]
    for _ in range(3000):
        written = "".join(rng.choice(pieces) for _ in range(rng.randint(1, 12)))
        arguments = '{"code":"' + written + '"}'
        assert json.loads(_repaired_arguments("execute_code", arguments)) == {"code": written}


def test_many_stray_quotes_repair_in_linear_time():
    arguments = '{"content":"' + 'a"b ' * 60000 + '"}'
    started = time.perf_counter()
    repaired = _repaired_arguments("write_file", arguments)
    assert json.loads(repaired) == {"content": 'a"b ' * 60000}
    assert time.perf_counter() - started < 2.0
