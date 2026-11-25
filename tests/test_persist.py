"""Persistent-conversation records, locks, fingerprints, decisions and packing."""

import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import persist
from persist import (
    MAX_STDIN_BYTES,
    ConversationLock,
    PersistStore,
    StdinPayloadTooLarge,
    check_stdin_bytes,
    decide,
    pack_oneshot,
    pack_refresh,
    reply_fingerprint,
    split_delta,
    tools_hash,
    utf8_len,
)

REPO = Path(__file__).resolve().parents[1]
TOOLS = [{"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}}]


def _call(cid, name, args):
    return {"id": cid, "type": "function", "function": {"name": name, "arguments": args}}


def _record(reply, **over):
    record = {
        "version": persist.RECORD_VERSION, "scope_hash": "k", "agy_conversation_id": "conv-1",
        "last_reply_fp": reply_fingerprint(reply), "model": "m", "effort": "low",
        "tools_hash": tools_hash(TOOLS), "agy_size_tokens": 1000, "needs_refresh": False,
        "last_user_fp": persist.user_fingerprint([{"role": "user", "content": "fix it"}], 1),
    }
    record.update(over)
    return record


REPLY = {"role": "assistant", "content": "Done. The file is fixed.",
         "tool_calls": [_call("c3", "terminal", '{"command": "ls", "cwd": "/tmp"}')]}
HISTORY = [
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "fix it"},
    REPLY,
]


class FingerprintTests(unittest.TestCase):
    def test_renamed_tool_call_id_still_matches(self):
        renamed = dict(REPLY, tool_calls=[_call("c3_d2", "terminal", '{"cwd":"/tmp","command":"ls"}')])
        self.assertEqual(reply_fingerprint(renamed), reply_fingerprint(REPLY))

    def test_reasoning_stripped_reply_still_matches(self):
        with_think = dict(REPLY, content="<think>let me see\nmore</think>\nDone.   The file is fixed.")
        self.assertEqual(reply_fingerprint(with_think), reply_fingerprint(REPLY))

    def test_tool_call_frame_in_text_is_ignored(self):
        framed = dict(REPLY, content='Done. The file is fixed.<tool_call>{"x":1}</tool_call>')
        self.assertEqual(reply_fingerprint(framed), reply_fingerprint(REPLY))

    def test_different_arguments_do_not_match(self):
        other = dict(REPLY, tool_calls=[_call("c3", "terminal", '{"command": "pwd", "cwd": "/tmp"}')])
        self.assertNotEqual(reply_fingerprint(other), reply_fingerprint(REPLY))


class DecideTests(unittest.TestCase):
    def setUp(self):
        self.th = tools_hash(TOOLS)
        self.messages = [*HISTORY, {"role": "tool", "tool_call_id": "c3", "content": "ok"}]

    def _decide(self, record, messages=None, model="m", th=None):
        return decide(record, messages or self.messages, model=model, tools_hash=th or self.th)

    def test_matching_record_resumes_with_only_messages_after_the_reply(self):
        d = self._decide(_record(REPLY))
        self.assertTrue(d.resume, d.reason)
        self.assertEqual(d.delta, [self.messages[-1]])

    def test_each_refresh_trigger(self):
        cases = {
            "no record": None,
            "needs refresh": _record(REPLY, needs_refresh=True),
            "record version": _record(REPLY, version=0),
            "no conversation id": _record(REPLY, agy_conversation_id=""),
            "model changed": _record(REPLY, model="other"),
            "tools changed": _record(REPLY, tools_hash="x"),
            "conversation size": _record(REPLY, agy_size_tokens=persist.REFRESH_SIZE_TOKENS + 1),
        }
        for reason, record in cases.items():
            with self.subTest(reason):
                d = self._decide(record)
                self.assertFalse(d.resume)
                self.assertEqual(d.reason, reason)

    def test_refresh_threshold_is_configurable(self):
        record = _record(REPLY, agy_size_tokens=150_000)
        self.assertTrue(self._decide(record).resume)
        with patch.dict(os.environ, {"ANTIGRAVITY_REFRESH_SIZE_TOKENS": "100000"}):
            self.assertEqual(self._decide(record).reason, "conversation size")

    def test_effort_change_alone_resumes(self):
        self.assertTrue(self._decide(_record(REPLY, effort="high")).resume)

    def test_assistant_message_after_ours_refreshes(self):
        messages = [*self.messages, {"role": "assistant", "content": "fallback answer"},
                    {"role": "user", "content": "next"}]
        d = self._decide(_record(REPLY), messages)
        self.assertEqual((d.resume, d.reason), (False, "fingerprint mismatch"))

    def test_interrupt_placeholder_as_last_assistant_refreshes(self):
        messages = [*self.messages,
                    {"role": "assistant", "content": "[interrupt: no assistant output for this turn]"},
                    {"role": "user", "content": "again"}]
        self.assertFalse(self._decide(_record(REPLY), messages).resume)

    def test_identical_earlier_reply_does_not_match(self):
        messages = [{"role": "user", "content": "a"}, REPLY,
                    {"role": "tool", "tool_call_id": "c3", "content": "ok"},
                    {"role": "assistant", "content": "something else"},
                    {"role": "user", "content": "b"}]
        self.assertFalse(self._decide(_record(REPLY), messages).resume)

    def test_merged_summary_wrapper_refreshes(self):
        wrapped = dict(REPLY, content="[CONTEXT COMPACTION] summary of earlier turns...\n\n"
                       + REPLY["content"] + "\n[END OF SUMMARY]")
        messages = [{"role": "system", "content": "s"}, wrapped, {"role": "user", "content": "go"}]
        self.assertFalse(self._decide(_record(REPLY), messages).resume)

    def test_no_new_messages_refreshes(self):
        self.assertEqual(self._decide(_record(REPLY), HISTORY).reason, "empty delta")


def _hermes_redacts():
    try:
        from agent.redact import redact_sensitive_text
    except Exception:
        return False
    return redact_sensitive_text("`ROMA_API_KEY=...` x") == "`ROMA_API_KEY=*** x"


def _committed(streamed, user="fix it"):
    """The record session.commit writes for a streamed reply."""
    return _record(streamed, last_reply_stored_fp=persist.stored_reply_fingerprint(streamed),
                   last_user_fp=persist.user_fingerprint([{"role": "user", "content": user}], 1))


class HermesStoredReplyTests(unittest.TestCase):
    """The next request carries the reply as Hermes stored it, not as agy streamed it.

    Cases observed on Chief (2026-11-21), each refreshed with "fingerprint mismatch".
    """

    def setUp(self):
        self.th = tools_hash(TOOLS)

    def _decide(self, record, stored, user="fix it", tail=None):
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": user}, stored,
                    *(tail or [{"role": "tool", "tool_call_id": "call_2", "content": "ok"}])]
        return decide(record, messages, model="m", tools_hash=self.th)

    def test_repeated_call_from_a_second_model_step_matches_hermes_dedupe(self):
        # 01:30 and 02:32: agy re-emitted call_2; Hermes renamed it call_2_d2,
        # then dropped it as a duplicate and stored a single call.
        streamed = {"role": "assistant", "content": "", "tool_calls": [
            _call("call_2", "web_extract", '{"urls": ["https://roma.app/learn/properties"]}'),
            _call("call_2_d2", "web_extract", '{"urls":["https://roma.app/learn/properties"]}')]}
        stored = dict(streamed, tool_calls=streamed["tool_calls"][:1])
        self.assertEqual(reply_fingerprint(stored), reply_fingerprint(streamed))
        self.assertTrue(self._decide(_committed(streamed), stored).resume)

    def test_one_repeat_among_several_calls(self):
        # 00:45: six calls streamed, five stored (one web_extract repeated).
        calls = [_call("c1", "web_search", '{"query": "a"}'), _call("c2", "web_search", '{"query": "b"}'),
                 _call("c3", "web_extract", '{"urls": ["u"]}'), _call("c4", "web_search", '{"query": "c"}'),
                 _call("c5", "web_extract", '{ "urls" : ["u"] }'), _call("c6", "web_search", '{"query": "d"}')]
        streamed = {"role": "assistant", "content": "", "tool_calls": calls}
        stored = dict(streamed, tool_calls=[c for c in calls if c["id"] != "c5"])
        self.assertTrue(self._decide(_committed(streamed), stored).resume)

    @unittest.skipUnless(_hermes_redacts(), "Hermes' redaction is not importable")
    def test_redacted_reply_matches(self):
        # 02:22: 1,047 characters delivered, 1,046 stored.
        streamed = {"role": "assistant", "content": "Put `ROMA_API_KEY=...` in the profile's `.env`."}
        stored = dict(streamed, content="Put `ROMA_API_KEY=*** in the profile's `.env`.")
        self.assertNotEqual(reply_fingerprint(stored), reply_fingerprint(streamed))
        d = self._decide(_committed(streamed), stored, tail=[{"role": "user", "content": "next"}])
        self.assertTrue(d.resume, d.reason)

    def test_unredacted_reply_still_matches(self):
        streamed = {"role": "assistant", "content": "Put `ROMA_API_KEY=...` in `.env`."}
        d = self._decide(_committed(streamed), streamed, tail=[{"role": "user", "content": "next"}])
        self.assertTrue(d.resume, d.reason)

    def test_record_without_stored_fingerprint_still_resumes(self):
        # Records written before the stored fingerprint existed.
        self.assertTrue(self._decide(_record(REPLY), REPLY, tail=[{"role": "tool", "content": "ok"}]).resume)

    # -- the safety property ------------------------------------------------
    def test_distinct_calls_are_not_merged(self):
        one = {"role": "assistant", "content": "", "tool_calls": [_call("a", "terminal", '{"command": "ls"}')]}
        two = dict(one, tool_calls=[*one["tool_calls"], _call("b", "terminal", '{"command": "pwd"}')])
        self.assertNotEqual(reply_fingerprint(one), reply_fingerprint(two))
        self.assertFalse(self._decide(_committed(two), one).resume)

    def test_edited_reply_refreshes(self):
        streamed = {"role": "assistant", "content": "Put `ROMA_API_KEY=...` in `.env`."}
        for edited in ("Put `ROMA_API_KEY=...` in `.env` now.", "Put `OTHER_KEY=*** in `.env`."):
            with self.subTest(edited):
                d = self._decide(_committed(streamed), dict(streamed, content=edited),
                                 tail=[{"role": "user", "content": "next"}])
                self.assertEqual((d.resume, d.reason), (False, "fingerprint mismatch"))

    def test_rewound_history_refreshes(self):
        earlier = {"role": "assistant", "content": "First answer."}
        latest = {"role": "assistant", "content": "", "tool_calls": [
            _call("call_2", "terminal", '{"command": "ls"}'), _call("call_2", "terminal", '{"command": "ls"}')]}
        messages = [{"role": "user", "content": "q1"}, earlier, {"role": "user", "content": "q2 again"}]
        d = decide(_committed(latest, user="q2"), messages, model="m", tools_hash=self.th)
        self.assertEqual((d.resume, d.reason), (False, "fingerprint mismatch"))

    def test_identical_earlier_reply_does_not_stand_in_for_the_last(self):
        streamed = {"role": "assistant", "content": "Put `ROMA_API_KEY=...` in `.env`."}
        stored = dict(streamed, content="Put `ROMA_API_KEY=*** in `.env`.")
        record = _committed(streamed, user="second")
        # The same reply to an earlier request, history undone to just after it.
        d = decide(record, [{"role": "user", "content": "first"}, stored, {"role": "user", "content": "x"}],
                   model="m", tools_hash=self.th)
        self.assertEqual((d.resume, d.reason), (False, "earlier request changed"))
        # The same reply earlier on, a different reply last.
        d = decide(record, [{"role": "user", "content": "second"}, stored,
                            {"role": "user", "content": "more"}, {"role": "assistant", "content": "other"},
                            {"role": "user", "content": "go"}], model="m", tools_hash=self.th)
        self.assertEqual((d.resume, d.reason), (False, "fingerprint mismatch"))


def _hermes_splits_batches():
    try:
        from agent.tool_call_batches import expand_local_tool_batches  # noqa: F401
        from agent.transports.types import ToolCall  # noqa: F401
    except Exception:
        return False
    return True


def _stored_row(first_id, second_id):
    """Tool calls of Chief's state.db rows 33059 (02:45) and 33114 (02:49), as stored."""
    def call(cid, entry):
        return {"id": cid, "call_id": cid, "response_item_id": "fc_" + cid.removeprefix("call_"),
                "type": "function", "function": {"name": "tool_call",
                                                 "arguments": json.dumps({"calls": [entry]})}}
    return {"role": "assistant", "content": "", "tool_calls": [
        call(first_id, {"name": "mcp__roma__list_agents", "arguments": {}}),
        call(second_id, {"name": "mcp__roma__manage_triggers", "arguments": {"action": "list"}})]}


# (time, stored row): the second id is the child id Hermes derived from the first.
BATCH_SPLIT_ROWS = (("02:45", _stored_row("call_51", "call_ea496249b67e")),
                    ("02:49", _stored_row("call_56", "call_28504e7be997")))
# Spellings agy may stream for the same two-entry batch; Hermes normalises each.
BATCH_SPELLINGS = (
    [{"name": "mcp__roma__list_agents", "arguments": {}},
     {"name": "mcp__roma__manage_triggers", "arguments": {"action": "list"}}],
    [{"name": "mcp__roma__list_agents"},
     {"name": "mcp__roma__manage_triggers", "arguments": '{"action": "list"}'}],
    [{"name": "mcp__roma__list_agents", "arguments": ""},
     {"name": "mcp__roma__manage_triggers", "arguments": {"action": "list"}}],
)


def _batch(cid, entries):
    return {"role": "assistant", "content": "", "tool_calls": [
        _call(cid, "tool_call", json.dumps({"calls": entries}))]}


class HermesToolPipelineTests(unittest.TestCase):
    """Hermes reshapes a reply's tool calls before storing it (agent/turn_tool_round.py):
    validation turns blank arguments into "{}", a tool_call batch of local tools is
    split into one call per entry, delegate_task calls past the limit are dropped,
    then repeated calls are. The record must match the result."""

    def setUp(self):
        self.th = tools_hash(TOOLS)

    def _decide(self, record, stored):
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "fix it"}, stored,
                    {"role": "tool", "tool_call_id": "x", "content": "ok"}]
        return decide(record, messages, model="m", tools_hash=self.th)

    @unittest.skipUnless(_hermes_splits_batches(), "Hermes' batch splitting is not importable")
    def test_split_tool_call_batch_resumes(self):
        # 02:45:42 and 02:49:19 on Chief: agy streamed one tool_call carrying
        # list_agents and manage_triggers; Hermes stored two calls.
        from agent.message_sanitization import deterministic_call_id

        for time_, stored in BATCH_SPLIT_ROWS:
            first, second = stored["tool_calls"]
            self.assertEqual(second["id"], deterministic_call_id(
                "tool_call", first["id"] + second["function"]["arguments"], 1), "split by Hermes")
            for spelling in BATCH_SPELLINGS:
                with self.subTest(time_, spelling=spelling):
                    streamed = _batch(first["id"], spelling)
                    self.assertNotEqual(reply_fingerprint(streamed), reply_fingerprint(stored))
                    d = self._decide(_committed(streamed), stored)
                    self.assertTrue(d.resume, d.reason)

    @unittest.skipUnless(_hermes_splits_batches(), "Hermes' batch splitting is not importable")
    def test_split_batch_with_a_different_entry_refreshes(self):
        _time, stored = BATCH_SPLIT_ROWS[0]
        for entries in (
            [{"name": "mcp__roma__list_agents", "arguments": {}},
             {"name": "mcp__roma__manage_triggers", "arguments": {"action": "create"}}],
            [{"name": "mcp__roma__manage_triggers", "arguments": {"action": "list"}},
             {"name": "mcp__roma__list_agents", "arguments": {}}],
            [{"name": "mcp__roma__list_agents", "arguments": {}}],
        ):
            with self.subTest(entries=entries):
                d = self._decide(_committed(_batch("call_51", entries)), stored)
                self.assertEqual((d.resume, d.reason), (False, "fingerprint mismatch"))

    @unittest.skipUnless(_hermes_splits_batches(), "Hermes' batch splitting is not importable")
    def test_split_batch_and_a_repeated_entry_drops_the_repeat(self):
        # Splitting comes before dropping repeats: the batch's own copy of an
        # entry that also arrived on its own is dropped.
        lone = _call("c1", "tool_call", json.dumps({"calls": [{"name": "mcp__roma__list_agents", "arguments": {}}]}))
        streamed = {"role": "assistant", "content": "", "tool_calls": [
            lone, *_batch("c2", BATCH_SPELLINGS[0])["tool_calls"]]}
        stored = dict(streamed, tool_calls=[lone, _stored_row("c2", "c3")["tool_calls"][1]])
        self.assertTrue(self._decide(_committed(streamed), stored).resume)

    def test_blank_arguments_stored_as_empty_object(self):
        streamed = {"role": "assistant", "content": "", "tool_calls": [_call("a", "todo", "")]}
        stored = dict(streamed, tool_calls=[_call("a", "todo", "{}")])
        self.assertTrue(self._decide(_committed(streamed), stored).resume)
        # Normalised first, so "" and "{}" are one call repeated.
        streamed = dict(streamed, tool_calls=[_call("a", "todo", " "), _call("b", "todo", "{}")])
        self.assertTrue(self._decide(_committed(streamed), stored).resume)

    def test_delegate_calls_past_hermes_limit_are_dropped(self):
        calls = [_call(f"d{i}", "delegate_task", json.dumps({"goal": f"g{i}"})) for i in range(3)]
        streamed = {"role": "assistant", "content": "", "tool_calls": [calls[0], _call("t", "terminal", "{}"),
                                                                       *calls[1:]]}
        stored = dict(streamed, tool_calls=streamed["tool_calls"][:3])
        helpers = dict(persist._hermes_helpers(), max_children=lambda: 2)
        with patch.object(persist, "_hermes_helpers_cache", helpers):
            self.assertTrue(self._decide(_committed(streamed), stored).resume)
        # Within the limit nothing is dropped, so the shorter history is a different reply.
        helpers["max_children"] = lambda: 3
        with patch.object(persist, "_hermes_helpers_cache", helpers):
            d = self._decide(_committed(streamed), stored)
        self.assertEqual((d.resume, d.reason), (False, "fingerprint mismatch"))

    def test_missing_hermes_helpers_warn_once_and_refresh_split_batches(self):
        _time, stored = BATCH_SPLIT_ROWS[0]
        streamed = _batch("call_51", BATCH_SPELLINGS[0])
        real_import = persist.importlib.import_module

        def no_agent(name, *args, **kwargs):
            if name.startswith(("agent.", "tools.")):
                raise ImportError(f"no module named {name!r}")
            return real_import(name, *args, **kwargs)

        with patch.object(persist, "_hermes_helpers_cache", None), \
                patch.object(persist.importlib, "import_module", no_agent), \
                self.assertLogs(persist.logger, level="WARNING") as logs:
            first = self._decide(_committed(streamed), stored)
            again = self._decide(_committed(streamed), stored)
            # Text and other calls are still fingerprinted, unchanged.
            self.assertTrue(self._decide(_committed(REPLY), REPLY).resume)
        self.assertEqual(len(logs.output), 1, logs.output)
        self.assertIn("agent.tool_call_batches.expand_local_tool_batches", logs.output[0])
        self.assertEqual((first.resume, first.reason), (False, "fingerprint mismatch"))
        self.assertFalse(again.resume)


def _hermes_repair():
    try:
        from agent.message_sanitization import _repair_tool_call_arguments
    except Exception:
        return None
    return _repair_tool_call_arguments


# Replies Hermes did not keep (2026-11-21): the model wrote tool arguments that
# are not JSON, Hermes' repair gave up ("Unrepairable tool_call arguments ...
# replaced with empty object") and Hermes re-sent the same request without
# storing the reply. Arguments trimmed from the logged ones; each keeps its flaw.
NEWS_HISTORY = [  # News state.db rows 5179-5183, trimmed
    {"role": "system", "content": "sys"},
    {"role": "user", "content": '[IMPORTANT: The user has invoked the "x-twitter" skill] X morning read'},
    {"role": "assistant", "content": "", "tool_calls": [
        _call("call_recover_text", "terminal", '{"command": "python3 x-ranked-feed-curator.py recover-text"}')]},
    {"role": "tool", "tool_call_id": "call_recover_text", "content": '{"output": "{\\"halted\\": null}"}'},
]
CHIEF_HISTORY = [  # Chief state.db rows 34287 and 34296, trimmed
    {"role": "system", "content": "sys"},
    {"role": "user", "content": "[Jack Laing] Pause the Roma bridge."},
    {"role": "assistant", "content": "The Roma bridge is paused and disabled."},
    {"role": "user", "content": "[Jack Laing] [ASYNC DELEGATION BATCH COMPLETE — deleg_cffed125] ..."},
]
DROPPED_REPLIES = (
    # 08:17, 08:18, 08:20 and 08:21 on News: a shell '\'' inside a JSON string.
    ("news 08:17", NEWS_HISTORY, "terminal",
     r"""{"command": "python3 x-ranked-feed-curator.py render-native --input-json"""
     r""" '{\"why\":\"Based on the author'\''s report.\"}'"}"""),
    # 08:19 on News: a shell '"'"' ends the JSON string early.
    ("news 08:19", NEWS_HISTORY, "terminal",
     r"""{"command":"python3 x-ranked-feed-curator.py render-native --input-json"""
     r""" '{\"why\":\"verify OpenRouter'"'"'s notice.\"}'"}"""),
    # 18:17 on Chief: a Python \' inside a JSON string.
    ("chief 18:17", CHIEF_HISTORY, "execute_code",
     r"""{"code":"print('plans for Jack\'s approval')\n"}"""),
)


def _turn_record(messages, reply):
    """The record session.commit writes after agy answers ``messages`` with ``reply``."""
    return _record(reply, last_reply_stored_fp=persist.stored_reply_fingerprint(reply),
                   last_user_fp=persist.user_fingerprint(messages, len(messages)),
                   last_request_fp=persist.request_fingerprint(messages))


class DroppedReplyTests(unittest.TestCase):
    """Hermes sends the request agy just answered again when it does not keep the
    reply. That must refresh (agy's conversation holds the discarded reply), and
    the log must say so rather than report a fingerprint mismatch."""

    def setUp(self):
        self.th = tools_hash(TOOLS)

    def _decide(self, record, messages):
        return decide(record, messages, model="m", tools_hash=self.th)

    def test_hermes_discards_these_replies(self):
        repair = _hermes_repair()
        if repair is None:
            self.skipTest("Hermes' argument repair is not importable")
        for label, _history, name, args in DROPPED_REPLIES:
            with self.subTest(label):
                with self.assertLogs("agent.message_sanitization", level="WARNING") as logs:
                    self.assertEqual(repair(args, name), "{}")
                self.assertIn("Unrepairable tool_call arguments", logs.output[0])

    def test_resent_request_refreshes_as_reply_not_kept(self):
        for label, history, name, args in DROPPED_REPLIES:
            with self.subTest(label):
                dropped = {"role": "assistant", "content": "", "tool_calls": [_call("call_9", name, args)]}
                d = self._decide(_turn_record(history, dropped), list(history))
                self.assertEqual((d.resume, d.reason), (False, "last reply not kept (same request re-sent)"))

    def test_dropped_reply_identical_to_the_previous_one_does_not_resume(self):
        # Hermes discards "Done." and sends the request again, which also ends
        # with "Done.": the reply fingerprint matches, but agy holds the extra reply.
        history = [{"role": "user", "content": "go"}, {"role": "assistant", "content": "Done."},
                   {"role": "user", "content": "again"}]
        reply = {"role": "assistant", "content": "Done."}
        d = self._decide(_turn_record(history, reply), list(history))
        self.assertEqual((d.resume, d.reason), (False, "last reply not kept (same request re-sent)"))
        # Kept and followed by the same words, it resumes: the request grew.
        follow = [*history, reply, {"role": "user", "content": "again"}]
        d = self._decide(_turn_record(history, reply), follow)
        self.assertTrue(d.resume, d.reason)
        self.assertEqual(d.delta, [{"role": "user", "content": "again"}])

    def test_kept_reply_still_resumes(self):
        for label, history, _name, _args in DROPPED_REPLIES:
            with self.subTest(label):
                reply = {"role": "assistant", "content": "", "tool_calls": [
                    _call("call_9", "terminal", '{"command": "ls"}')]}
                follow = [*history, reply, {"role": "tool", "tool_call_id": "call_9", "content": "ok"}]
                d = self._decide(_turn_record(history, reply), follow)
                self.assertTrue(d.resume, d.reason)

    def test_other_requests_after_a_dropped_reply_still_report_a_mismatch(self):
        _label, history, name, args = DROPPED_REPLIES[0]
        dropped = {"role": "assistant", "content": "", "tool_calls": [_call("call_9", name, args)]}
        record = _turn_record(history, dropped)
        changed = (
            [*history[:-1], dict(history[-1], content="different output")],  # tool output differs
            [*history, {"role": "user", "content": "stop"}],  # a message added
            history[:-1],  # a message removed
        )
        for messages in changed:
            with self.subTest(messages=messages[-1]):
                d = self._decide(record, messages)
                self.assertEqual((d.resume, d.reason), (False, "fingerprint mismatch"))
        # Records written before the request fingerprint existed.
        record.pop("last_request_fp")
        d = self._decide(record, list(history))
        self.assertEqual((d.resume, d.reason), (False, "fingerprint mismatch"))


def _big_history(n, size, char="x"):
    msgs = [{"role": "system", "content": "You are Hermes."}]
    for i in range(n):
        msgs.append({"role": "user", "content": f"question {i} " + char * size})
        msgs.append({"role": "assistant", "content": f"answer {i} " + char * size})
    return msgs


def _seed_labels(case, out):
    """Each seed's label ("context" or "active"), checking its header and order.

    Older-history seeds keep ``[CONTEXT ONLY - PART k OF N``; seeds carrying
    any of a spilled active tool loop use ``[ACTIVE TOOL LOOP - PART k OF N``.
    The loop is the newest history, so no context-only seed follows one.
    """
    n = len(out)
    labels = []
    for k, payload in enumerate(out[:-1], start=1):
        if payload.startswith(f"[CONTEXT ONLY - PART {k} OF {n}. This message restores earlier conversation state."):
            labels.append("context")
        elif payload.startswith(f"[ACTIVE TOOL LOOP - PART {k} OF {n}. This message carries part of the "
                                "in-progress tool loop. Do NOT act on it yet"):
            labels.append("active")
            case.assertIn("wait for the final part before continuing. Reply with exactly: ACK]", payload)
        else:
            case.fail(f"unexpected seed header: {payload[:80]!r}")
    case.assertEqual(labels, sorted(labels, key=lambda label: label == "active"),
                     "active-loop seeds come after every older-history seed")
    return labels


class PackingTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for name in ("ANTIGRAVITY_REFRESH_MAX_TOKENS", "ANTIGRAVITY_REFRESH_SIZE_TOKENS"):
            os.environ.pop(name, None)

    def _assert_well_formed(self, out, active_loop=False):
        """Seeds are older history only, unless ``active_loop`` (a spilled loop)."""
        n = len(out)
        labels = _seed_labels(self, out)
        for k, payload in enumerate(out, start=1):
            self.assertLess(utf8_len(payload), MAX_STDIN_BYTES)
            if k < n:
                self.assertIn("Reply with exactly: ACK", payload)
            else:
                self.assertTrue(payload.startswith(f"[PART {n} OF {n}."), payload[:60])
        if active_loop:
            self.assertIn("active", labels)
            self.assertIn("start of the in-progress tool loop", out[-1])
        else:
            self.assertEqual(set(labels) - {"context"}, set(), "older history keeps CONTEXT ONLY")
            if n > 1:
                self.assertIn("The earlier parts were context only", out[-1])

    def test_small_history_is_one_message_without_seed(self):
        out = pack_refresh(HISTORY + [{"role": "user", "content": "hi"}], model="m", tools=TOOLS,
                           tool_choice=None)
        self.assertEqual(len(out), 1)

    def test_large_history_grows_into_as_many_seeds_as_needed_multibyte(self):
        # ~400-600 KB of history, under the default token cap: nothing dropped.
        for char, size in (("x", 2000), ("é", 1000), ("日", 700)):
            with self.subTest(char=char):
                msgs = _big_history(100, size, char) + [{"role": "user", "content": "LATEST ASK"}]
                out = pack_refresh(msgs, model="m", tools=TOOLS, tool_choice=None)
                self.assertGreater(len(out), 2)
                self._assert_well_formed(out)
                self.assertNotIn("omitted]", "".join(out))
                self.assertIn("You are Hermes.", out[0])
                for payload in out[1:]:
                    self.assertNotIn("You are Hermes.", payload)
                self.assertIn("LATEST ASK", out[-1])
                for payload in out[:-1]:
                    self.assertNotIn("LATEST ASK", payload)
                joined = "\n".join(out)
                markers = [f"question {i} " for i in range(100)] + [f"answer {i} " for i in range(100)]
                for marker in markers:
                    self.assertEqual(joined.count(marker), 1, marker)
                order = [joined.index(f"{kind} {i} ") for i in range(100) for kind in ("question", "answer")]
                self.assertEqual(order, sorted(order))

    def test_history_over_the_token_cap_drops_oldest_with_a_warning(self):
        msgs = _big_history(400, 2000) + [{"role": "user", "content": "LATEST ASK"}]
        with self.assertLogs("persist", level="WARNING") as logs:
            out = pack_refresh(msgs, model="m", tools=TOOLS, tool_choice=None)
        self._assert_well_formed(out)
        total = sum(persist.estimate_tokens(p) for p in out)
        self.assertLessEqual(total, persist.REFRESH_MAX_TOKENS)
        self.assertGreater(total, persist.REFRESH_MAX_TOKENS * 0.9, "the cap is used, not trimmed far below")
        joined = "\n".join(out)
        self.assertNotIn("question 0 ", joined)
        self.assertIn("answer 399 ", joined)
        marker = re.search(r"\[(\d+) earlier messages omitted\]", out[0])
        self.assertIsNotNone(marker)
        dropped = int(marker.group(1))
        self.assertIn(f"question {dropped // 2} " if dropped % 2 == 0 else f"answer {dropped // 2} ", joined)
        message = "\n".join(logs.output)
        self.assertIn(f"dropped the {dropped} oldest message(s)", message)
        self.assertRegex(message, r"~\d+ estimated tokens")

    def test_cap_is_configurable_and_clamped_below_the_refresh_threshold(self):
        msgs = _big_history(100, 2000) + [{"role": "user", "content": "LATEST ASK"}]
        os.environ["ANTIGRAVITY_REFRESH_MAX_TOKENS"] = "60000"
        out = pack_refresh(msgs, model="m", tools=TOOLS, tool_choice=None)
        self._assert_well_formed(out)
        self.assertLessEqual(sum(persist.estimate_tokens(p) for p in out), 60_000)
        self.assertIn("earlier messages omitted]", out[0])
        os.environ["ANTIGRAVITY_REFRESH_SIZE_TOKENS"] = "100000"
        os.environ["ANTIGRAVITY_REFRESH_MAX_TOKENS"] = "500000"
        with self.assertLogs("persist", level="WARNING"):
            self.assertEqual(persist.refresh_max_tokens(), 75_000)
        os.environ["ANTIGRAVITY_REFRESH_MAX_TOKENS"] = "junk"
        with self.assertLogs("persist", level="WARNING"):
            self.assertEqual(persist.refresh_max_tokens(), 75_000)  # default 200k, clamped

    def test_cap_alone_forcing_a_split_still_sends_one_message_when_it_fits(self):
        msgs = _big_history(20, 2000) + [{"role": "user", "content": "LATEST ASK"}]
        out = pack_refresh(msgs, model="m", tools=TOOLS, tool_choice=None, max_tokens=5000)
        self.assertEqual(len(out), 1)
        self.assertIn("earlier messages omitted]", out[0])
        self.assertIn("LATEST ASK", out[0])
        self.assertLessEqual(persist.estimate_tokens(out[0]), 5000)

    def test_active_tool_loop_stays_intact_in_the_final_part(self):
        msgs = _big_history(100, 2000)
        msgs.append({"role": "user", "content": "LOOP REQUEST"})
        for i in range(6):
            msgs.append({"role": "assistant", "content": "",
                         "tool_calls": [_call(f"t{i}", "terminal", json.dumps({"n": i}))]})
            msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": f"RESULT-{i} " + "r" * 15000})
        out = pack_refresh(msgs, model="m", tools=TOOLS, tool_choice=None)
        self.assertGreater(len(out), 2)
        self._assert_well_formed(out)
        self.assertIn("LOOP REQUEST", out[-1])
        for i in range(6):
            self.assertIn(f"RESULT-{i} ", out[-1])
        positions = [out[-1].index(f"RESULT-{i} ") for i in range(6)]
        self.assertEqual(positions, sorted(positions))
        self.assertIn("USER REQUEST THIS TOOL LOOP IS SERVING", out[-1])

    def test_oversized_prefix_raises(self):
        msgs = [{"role": "system", "content": "s" * 175_000}, {"role": "user", "content": "hi"}]
        with self.assertRaises(StdinPayloadTooLarge):
            pack_refresh(msgs, model="m", tools=None, tool_choice=None)

    @staticmethod
    def _loop_of_long_assistant_turns():
        msgs = [{"role": "user", "content": "go"}]
        for i in range(15):
            msgs.append({"role": "assistant", "content": f"STEP-{i} " + "a" * 30000,
                         "tool_calls": [_call(f"t{i}", "x", "{}")]})
            msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": f"RES-{i} " + "z" * 1000})
        return msgs

    def test_active_loop_too_large_spills_into_context_only_parts(self):
        # 450 KB of assistant text in one tool loop: no pruning makes it fit
        # one message, so its oldest steps go to context-only parts.
        msgs = self._loop_of_long_assistant_turns()
        with self.assertLogs("persist", level="WARNING") as logs:
            out = pack_refresh(msgs, model="m", tools=None, tool_choice=None)
        self.assertGreater(len(out), 2)
        self._assert_well_formed(out, active_loop=True)
        self.assertIn("RES-14 ", out[-1], "the newest tool result is in the final part")
        self.assertIn("Continue the conversation from the latest tool result", out[-1])
        self.assertNotIn("STEP-0 ", out[-1])
        joined = "\n".join(out)
        positions = [joined.index(f"STEP-{i} ") for i in range(15)]
        self.assertEqual(positions, sorted(positions), "every step is kept, in order")
        self.assertIn("does not fit one message", "\n".join(logs.output))

    def test_oneshot_active_loop_too_large_still_raises_instead_of_truncating(self):
        # A one-shot has no context-only parts to spill into.
        with self.assertRaises(StdinPayloadTooLarge):
            pack_oneshot(self._loop_of_long_assistant_turns(), model="m", tools=None, tool_choice=None)

    def test_oneshot_pack_is_single_message_under_limit(self):
        msgs = _big_history(400, 2000, "é") + [{"role": "user", "content": "LATEST ASK"}]
        out = pack_oneshot(msgs, model="m", tools=TOOLS, tool_choice=None)
        self.assertLess(utf8_len(out), MAX_STDIN_BYTES)
        self.assertIn("LATEST ASK", out)
        self.assertIn("earlier messages omitted]", out)


def _tool_loop(n, chars, request="Refactor the repo please.", unit="y = f(x)\n"):
    loop: list[dict] = [{"role": "user", "content": request}]
    for i in range(n):
        loop.append({"role": "assistant", "content": "",
                     "tool_calls": [_call(f"c{i}", "terminal", json.dumps({"command": f"cat f{i}.py"}))]})
        loop.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"OUT-{i} " + unit * (chars // len(unit))})
    return loop


class SummaryAndLoopPackingTests(unittest.TestCase):
    """Review B1 (summaries survive the cap), 4 (long active loops) and 5 (one-shot warning)."""

    def setUp(self):
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for name in ("ANTIGRAVITY_REFRESH_MAX_TOKENS", "ANTIGRAVITY_REFRESH_SIZE_TOKENS"):
            os.environ.pop(name, None)

    def test_summary_markers_are_recognised(self):
        summaries = [
            "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted into the summary below.",
            "[PRIOR CONTEXT — for reference only; not a new message]\nstuff",
            "[Current user objective preserved from compacted history]\nfix the build",
            "[Recent Summary (d0, node 12)]\nthe deploy key lives in vault B",
            "  [Session Arc Summary (d1, node 3)]\n...",
            "[CONTEXT SUMMARY]: legacy handoff",
        ]
        for text in summaries:
            with self.subTest(text[:30]):
                self.assertTrue(persist.is_summary_message({"role": "user", "content": text}))
        # Markers count only at the start: a message quoting one is ordinary
        # history, not a mandatory summary moved ahead of everything else.
        quoting = [
            "intro\n[Recent Summary (d0, node 12)]\nthe deploy key lives in vault B",
            "body\n--- END OF CONTEXT SUMMARY — respond to the message below, not the summary above ---",
            "LCM printed: [CONTEXT COMPACTION — REFERENCE ONLY] ...",
        ]
        for text in quoting:
            with self.subTest(text[:30]):
                self.assertFalse(persist.is_summary_message({"role": "user", "content": text}))
        self.assertTrue(persist.is_summary_message(
            {"role": "assistant", "content": [{"type": "text", "text": summaries[0]}]}))
        self.assertFalse(persist.is_summary_message({"role": "user", "content": "please summarize this"}))
        self.assertFalse(persist.is_summary_message({"role": "tool", "content": summaries[0]}))

    def _capped_history(self, summary_text):
        msgs = [{"role": "system", "content": "SYSTEM-SENTINEL"},
                {"role": "user", "content": summary_text}, {"role": "assistant", "content": "ok"}]
        msgs += _big_history(400, 2000)[1:]
        msgs.append({"role": "user", "content": "LATEST ASK"})
        return msgs

    def test_capped_refresh_keeps_native_and_lcm_summaries_and_drops_oldest_ordinary(self):
        for summary in (
            "[CONTEXT COMPACTION — REFERENCE ONLY] SUMMARY-FACT-ZETA: the key is in vault B.",
            "[Current user objective preserved from compacted history]\nobjective\n\n"
            "[Recent Summary (d0, node 1)]\nSUMMARY-FACT-ZETA: the key is in vault B.",
        ):
            with self.subTest(summary[:20]), self.assertLogs("persist", level="WARNING") as logs:
                out = pack_refresh(self._capped_history(summary), model="m", tools=TOOLS, tool_choice=None)
                self.assertGreater(len(out), 2)
                self.assertIn("SYSTEM-SENTINEL", out[0])
                self.assertIn("SUMMARY-FACT-ZETA", out[0], "summary sits with the prefix")
                head = out[0]
                self.assertLess(head.index("SYSTEM-SENTINEL"), head.index("SUMMARY-FACT-ZETA"))
                marker = re.search(r"\[(\d+) earlier messages omitted\]", head)
                assert marker is not None
                self.assertLess(head.index("SUMMARY-FACT-ZETA"), marker.start(),
                                "the omission marker follows the summary")
                joined = "\n".join(out)
                self.assertNotIn("question 0 ", joined)
                self.assertIn("answer 399 ", joined)
                self.assertIn("LATEST ASK", out[-1])
                self.assertLessEqual(sum(persist.estimate_tokens(p) for p in out), persist.REFRESH_MAX_TOKENS)
                self.assertIn("1 summary message(s)", "\n".join(logs.output))

    def test_oneshot_keeps_the_summary_and_warns_about_omitted_history(self):
        msgs = self._capped_history("[CONTEXT COMPACTION] SUMMARY-FACT-ZETA")
        with self.assertLogs("persist", level="WARNING") as logs:
            out = pack_oneshot(msgs, model="m", tools=TOOLS, tool_choice=None)
        self.assertIn("SUMMARY-FACT-ZETA", out)
        self.assertIn("earlier messages omitted]", out)
        self.assertLess(out.index("SUMMARY-FACT-ZETA"), out.index("earlier messages omitted]"))
        self.assertIn("one-shot prompt omits", "\n".join(logs.output))

    def test_oneshot_that_fits_logs_nothing(self):
        with self.assertNoLogs("persist", level="WARNING"):
            pack_oneshot(_big_history(5, 100) + [{"role": "user", "content": "hi"}], model="m",
                         tools=TOOLS, tool_choice=None)

    def test_long_active_tool_loop_with_production_prefix_refreshes(self):
        # The review's probe: a 118 KB system prompt and 16-40 tool results of
        # 12K chars raised StdinPayloadTooLarge before old-output pruning.
        system = {"role": "system", "content": ("Hermes SOUL and skills. " * 4800)[:118_000]}
        for steps in (16, 40):
            with self.subTest(steps=steps):
                msgs = [system, *_tool_loop(steps, 12_000)]
                out = pack_refresh(msgs, model="m", tools=TOOLS, tool_choice=None)
                for payload in out:
                    self.assertLess(utf8_len(payload), MAX_STDIN_BYTES)
                final = out[-1]
                self.assertIn("Refactor the repo please.", final)
                self.assertIn(f"OUT-{steps - 1} ", final, "the newest tool result is kept")
                self.assertIn(f"OUT-{steps - 1} " + "y = f(x)\n" * 50, final, "and in full")
                self.assertIn("[... tool output truncated:", final, "older output is a stub")

    def test_loop_pruning_steps_down_until_it_fits_and_warns(self):
        # 19K-char multibyte results (57 KB each, under the 20K-char cap):
        # neither 8 nor 4 retained results fit one message; 1 does.
        system = {"role": "system", "content": "s" * 60_000}
        msgs = [system, *_tool_loop(10, 19_000, unit="日")]
        with self.assertLogs("persist", level="WARNING") as logs:
            out = pack_refresh(msgs, model="m", tools=TOOLS, tool_choice=None)
        self.assertIn("OUT-9 ", out[-1])
        self.assertRegex("\n".join(logs.output), r"only fits with the 1 newest tool result")
        for payload in out:
            self.assertLess(utf8_len(payload), MAX_STDIN_BYTES)


    def test_single_message_keeps_summaries_ahead_of_ordinary_history(self):
        # Review S1 (h525_probe_single_summary): a 118 KB prefix and a 60 KB
        # summary used to lose the summary while 21 newer ordinary pieces stayed.
        system = {"role": "system", "content": ("Hermes SOUL and skills. " * 8000)[:118_000]}
        for summary_kb in (60, 80):
            with self.subTest(summary_kb=summary_kb):
                summary = ("[Recent Summary (d0, node 7)]\nFACT-SUMMARY-QX "
                           + ("summary words here " * 6000)[:summary_kb * 1000])
                msgs = [system, {"role": "user", "content": summary}, {"role": "assistant", "content": "ok"}]
                for i in range(10):
                    msgs += [{"role": "user", "content": f"note {i} " + "ordinary " * 400},
                             {"role": "assistant", "content": f"ack {i}"}]
                msgs.append({"role": "user", "content": "What is FACT-SUMMARY-QX?"})
                with self.assertLogs("persist", level="WARNING"):
                    out = pack_oneshot(msgs, model="m", tools=TOOLS, tool_choice=None)
                self.assertLess(utf8_len(out), MAX_STDIN_BYTES)
                self.assertIn("FACT-SUMMARY-QX ", out)
                self.assertIn("What is FACT-SUMMARY-QX?", out)
                self.assertIn("[summary truncated to fit]", out)
                self.assertNotIn("note 0 ", out, "ordinary history yields to the summary")
                self.assertLess(out.index("FACT-SUMMARY-QX "), out.index("earlier messages omitted]"))

    def test_single_message_keeps_the_newest_summary_pieces_when_not_all_fit(self):
        system = {"role": "system", "content": "s" * 120_000}
        summaries = [{"role": "user", "content": f"[Depth-{d} Summary (d{d}, node {d})]\nSUM-{d} " + "w" * 30_000}
                     for d in (3, 2, 1)]
        msgs = [system, *summaries, {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "ordinary " * 500}, {"role": "assistant", "content": "fine"},
                {"role": "user", "content": "LATEST ASK"}]
        with self.assertLogs("persist", level="WARNING") as logs:
            out = pack_oneshot(msgs, model="m", tools=TOOLS, tool_choice=None)
        self.assertLess(utf8_len(out), MAX_STDIN_BYTES)
        self.assertIn("LATEST ASK", out)
        self.assertIn("SUM-1 ", out, "the newest summary is kept")
        self.assertIn("SUM-2 ", out, "the next one is cut rather than dropped")
        self.assertNotIn("SUM-3 ", out, "the oldest summary piece goes first")
        self.assertLess(out.index("SUM-2 "), out.index("[summary truncated to fit]"), )
        self.assertLess(out.index("[summary truncated to fit]"), out.index("SUM-1 "))
        self.assertNotIn("ordinary ordinary", out, "ordinary history only gets what summaries leave")
        self.assertIn("summary pieces", "\n".join(logs.output))


def _pulse_run(request_bytes=54_000, result_chars=15_000):
    """A news-briefing cron turn: one request, 25 tool calls in parallel batches.

    The shape that failed in production: a ~54 KB request (restated in full in
    the closing), large write_file arguments, and many 12-20K-char results.
    """
    request = "PULSE-REQUEST " + ("brief the news per the skill. " * 4000)[:request_bytes]
    msgs: list[dict] = [{"role": "system", "content": ("Hermes SOUL and skills. " * 4500)[:100_000]},
                        {"role": "user", "content": request}]
    n = 0
    for batch in (4, 4, 4, 4, 3):
        calls = [_call(f"c{n + j}", "web_search", json.dumps({"query": f"q{n + j}"})) for j in range(batch)]
        msgs.append({"role": "assistant", "content": "", "tool_calls": calls})
        for j in range(batch):
            msgs.append({"role": "tool", "tool_call_id": f"c{n + j}",
                         "content": f"OUT-{n + j} " + "headline text. " * (result_chars // 15)})
        n += batch
    for size in (36_000, 24_000):  # writing the briefing files
        args = json.dumps({"path": f"/tmp/brief-{n}.md", "content": f"ARGS-{n} " + "w" * size})
        msgs.append({"role": "assistant", "content": "", "tool_calls": [_call(f"c{n}", "write_file", args)]})
        msgs.append({"role": "tool", "tool_call_id": f"c{n}", "content": f"OUT-{n} wrote file"})
        n += 1
    for _ in range(4):  # small patches
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [_call(f"c{n}", "patch", json.dumps({"old": "a", "new": "b"}))]})
        msgs.append({"role": "tool", "tool_call_id": f"c{n}", "content": f"OUT-{n} patched " + "ok " * 200})
        n += 1
    return msgs, n


class ActiveLoopSpillTests(unittest.TestCase):
    """A refresh whose active tool loop alone exceeds one stdin message."""

    def setUp(self):
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for name in ("ANTIGRAVITY_REFRESH_MAX_TOKENS", "ANTIGRAVITY_REFRESH_SIZE_TOKENS"):
            os.environ.pop(name, None)

    def _assert_parts(self, out):
        n = len(out)
        self.assertGreaterEqual(n, 2)
        labels = _seed_labels(self, out)
        self.assertIn("active", labels, "seeds carrying the loop use the active-tool-loop label")
        for k, payload in enumerate(out[:-1], start=1):
            self.assertLess(utf8_len(payload), MAX_STDIN_BYTES)
            self.assertIn("Reply with exactly: ACK", payload)
            self.assertNotIn("Continue the conversation from the latest tool result", payload)
        self.assertLess(utf8_len(out[-1]), MAX_STDIN_BYTES)
        self.assertTrue(out[-1].startswith(f"[PART {n} OF {n}. The earlier parts carried context and the "
                                           "start of the in-progress tool loop"), out[-1][:120])
        return labels
        self.assertIn("Continue the conversation from the latest tool result", out[-1])
        self.assertIn("USER REQUEST THIS TOOL LOOP IS SERVING", out[-1])

    def test_pulse_shaped_loop_no_longer_aborts_and_keeps_its_tail_last(self):
        msgs, calls = _pulse_run()
        self.assertGreaterEqual(calls, 19)
        history = [m for m in msgs if m["role"] != "system"]
        prefix = persist._prefix_sections(msgs, TOOLS, None, None)
        for keep in persist.TOOL_RETENTION_STEPS:  # the production failure
            with self.assertRaises(persist._ActiveLoopTooLarge):
                persist._pack_split(persist._prune_tool_results(history, keep), prefix,
                                    persist.REFRESH_MAX_TOKENS, None)

        with self.assertLogs("persist", level="WARNING") as logs:
            out = pack_refresh(msgs, model="m", tools=TOOLS, tool_choice=None)
        self._assert_parts(out)
        self.assertIn("does not fit one message", "\n".join(logs.output))
        final = out[-1]
        # assertTrue, not assertIn: a failure must not print 160 KB payloads.
        self.assertTrue("PULSE-REQUEST" in final, "the request is restated in the final part")
        self.assertTrue((f"OUT-{calls - 1} patched " + "ok " * 200).rstrip() in final,
                        "the newest result is in the final part, in full")
        self.assertTrue(f'"id": "c{calls - 1}"' in final, "with the call that produced it")
        self.assertTrue(any("User:\nPULSE-REQUEST" in p for p in out[:-1]),
                        "the loop's oldest entry (the request) spilled into a context-only part")
        joined = "\n".join(out)
        positions = [joined.index(f"OUT-{i} ") for i in range(calls)]
        self.assertEqual(positions, sorted(positions), "every tool result is kept, in order")
        self.assertTrue("ARGS-19 " in joined and "ARGS-20 " in joined, "large arguments are kept")
        # The canonical retention: the 8 newest results in full, older ones stubbed.
        self.assertTrue("[... tool output truncated:" in joined)
        self.assertLessEqual(sum(persist.estimate_tokens(p) for p in out), persist.REFRESH_MAX_TOKENS)

    def test_older_history_seeds_keep_context_only_and_loop_seeds_get_their_own_label(self):
        msgs = [{"role": "system", "content": "SYS"}]
        for i in range(40):  # ~320 KB: more than one seed of older history
            msgs.append({"role": "user", "content": f"OLD-Q{i} " + "q" * 4000})
            msgs.append({"role": "assistant", "content": f"OLD-A{i} " + "a" * 4000})
        msgs += PackingTests._loop_of_long_assistant_turns()
        with self.assertLogs("persist", level="WARNING"):
            out = pack_refresh(msgs, model="m", tools=TOOLS, tool_choice=None)
        labels = self._assert_parts(out)
        self.assertIn("context", labels, "older history still uses the CONTEXT ONLY label")
        for label, payload in zip(labels, out[:-1]):
            # The loop opens with its request ("go"), then its steps.
            if "STEP-" in payload or "User:\ngo" in payload:
                self.assertEqual(label, "active", payload[:80])
            else:
                self.assertEqual(label, "context", payload[:80])
                self.assertTrue(payload.startswith("[CONTEXT ONLY - PART "))
        self.assertTrue("OLD-A39 " in "\n".join(out[:-1]), "older history sits in the seeds")
        self.assertTrue("RES-14 " in out[-1])

    def test_huge_single_entries_are_split_and_the_newest_piece_stays_final(self):
        cases = (
            # A 300 KB tool-call argument in the middle of the loop.
            ("huge arguments", 120_000, 300_000, "LAST-RESULT ok"),
            # The newest result itself is bigger than the room beside a 150 KB request.
            ("huge newest result", 150_000, 10, "LAST-RESULT " + "日" * 19_000),
        )
        for name, request_bytes, args_bytes, last in cases:
            with self.subTest(name):
                msgs = [{"role": "system", "content": "SYS"},
                        {"role": "user", "content": "ASK " + "r" * request_bytes},
                        {"role": "assistant", "content": "", "tool_calls": [
                            _call("w1", "write_file", json.dumps({"content": "BIG-ARGS " + "w" * args_bytes}))]},
                        {"role": "tool", "tool_call_id": "w1", "content": "MIDDLE-RESULT wrote"},
                        {"role": "assistant", "content": "", "tool_calls": [_call("r1", "read_file", "{}")]},
                        {"role": "tool", "tool_call_id": "r1", "content": last}]
                with self.assertLogs("persist", level="WARNING"):
                    out = pack_refresh(msgs, model="m", tools=TOOLS, tool_choice=None)
                self._assert_parts(out)
                joined = "\n".join(out)
                self.assertLess(joined.index("BIG-ARGS "), joined.index("MIDDLE-RESULT"))
                self.assertLess(joined.index("MIDDLE-RESULT"), joined.index("LAST-RESULT"))
                self.assertTrue(last[-500:] in out[-1], "the newest result's tail is in the final part")
                if name == "huge arguments":
                    self.assertTrue("[message continued, piece" in joined)
                    self.assertTrue("LAST-RESULT ok" in out[-1])
                else:
                    self.assertTrue("[message continued, piece" in out[-1])

    def test_request_too_large_for_the_final_part_fails_closed(self):
        msgs = [{"role": "user", "content": "ASK " + "r" * 176_500},
                {"role": "assistant", "content": "", "tool_calls": [_call("r1", "read_file", "{}")]},
                {"role": "tool", "tool_call_id": "r1", "content": "result"}]
        with self.assertRaises(StdinPayloadTooLarge) as ctx:
            pack_refresh(msgs, model="m", tools=TOOLS, tool_choice=None)
        self.assertNotIsInstance(ctx.exception, persist._ActiveLoopTooLarge)
        self.assertIn("leaving no room", str(ctx.exception))
        # Hermes still compacts on it, as on every stdin-limit error.
        from providers import get_provider_profile
        hook = get_provider_profile("antigravity-subscription-directsdk").classify_api_error
        self.assertEqual(hook(ctx.exception),
                         {"reason": "context_overflow", "retryable": True, "should_compress": True})


class HermesRequestTokensTests(unittest.TestCase):
    """Review B2: the size reported to Hermes for persistent turns."""

    def test_reflects_hermes_history_and_shrinks_after_compaction(self):
        system = {"role": "system", "content": "SOUL"}
        history = [system]
        for i in range(40):
            history += [{"role": "user", "content": f"q{i} " + "w " * 2000},
                        {"role": "assistant", "content": f"a{i}"}]
        full = persist.hermes_request_tokens(history, TOOLS, None)
        self.assertGreater(full, 40 * 1000)
        compacted = [system, {"role": "user", "content": "[CONTEXT COMPACTION] summary"}, *history[-4:]]
        self.assertLess(persist.hermes_request_tokens(compacted, TOOLS, None), full // 5)

    def test_counts_every_tool_output_in_full(self):
        # Final review should-fix: stubbing all but the 8 newest results made
        # the reported size 13-72% of what Hermes and LCM-X measure, so
        # mid-turn and native-compression checks barely fired.
        loop = _tool_loop(12, 20_000)
        tokens = persist.hermes_request_tokens(loop, TOOLS, None)
        self.assertGreater(tokens, 12 * 20_000 // 4, "old tool output is not stubbed")
        self.assertLess(tokens, 13 * 20_000 // 4)
        # Adding older tool output grows the size one for one, past the 8th.
        longer = _tool_loop(20, 20_000)
        grown = persist.hermes_request_tokens(longer, TOOLS, None) - tokens
        self.assertGreater(grown, 8 * 20_000 // 4)

    def test_is_never_below_what_a_rebuild_sends(self):
        # A rebuild stubs old tool output, so a threshold on this size keeps
        # every rebuild below it.
        loop = _tool_loop(12, 20_000)
        rebuilt = persist.estimate_tokens("\n\n".join(
            persist._flatten_content(m.get("content"))
            for m in persist._prune_tool_results(loop, persist.TOOL_RETENTION_STEPS[0])))
        self.assertGreater(persist.hermes_request_tokens(loop, TOOLS, None), rebuilt)

    def test_reads_text_only_and_never_resolves_media(self):
        msgs = [{"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "https://example.invalid/cat.png"}}]}]
        with patch("prompt._materialize_media", side_effect=AssertionError("media resolved")):
            self.assertGreater(persist.hermes_request_tokens(msgs, None, None), 0)


class ByteGuardTests(unittest.TestCase):
    def test_guard_rejects_at_limit_and_counts_bytes_not_chars(self):
        check_stdin_bytes("a" * (MAX_STDIN_BYTES - 1))
        with self.assertRaises(StdinPayloadTooLarge):
            check_stdin_bytes("a" * MAX_STDIN_BYTES)
        with self.assertRaises(StdinPayloadTooLarge):
            check_stdin_bytes("日" * 60_000)  # 60k chars, 180k bytes

    def test_small_delta_is_one_message(self):
        out = split_delta([{"role": "user", "content": "hi"}])
        self.assertEqual(len(out), 1)
        self.assertIn("LATEST USER REQUEST", out[0])

    def test_oversized_delta_splits_into_seeds_with_request_last(self):
        delta = [{"role": "tool", "tool_call_id": f"t{i}", "content": f"OUT-{i} " + "日" * 6000}
                 for i in range(40)]
        out = split_delta(delta, pending_user_text="THE REQUEST")
        self.assertGreater(len(out), 1)
        for k, payload in enumerate(out[:-1], start=1):
            self.assertLess(utf8_len(payload), MAX_STDIN_BYTES)
            self.assertTrue(payload.startswith(f"[CONTEXT ONLY - PART {k} OF {len(out)}"))
            self.assertNotIn("THE REQUEST", payload)
        self.assertLess(utf8_len(out[-1]), MAX_STDIN_BYTES)
        self.assertIn("THE REQUEST", out[-1])
        joined = "\n".join(out)
        positions = [joined.index(f"OUT-{i} ") for i in range(40)]
        self.assertEqual(positions, sorted(positions))

    def test_long_request_tool_step_delta_restates_an_excerpt(self):
        # Pulse: a 54 KB request re-pasted on every tool step of a persistent turn.
        request = "PULSE-REQUEST " + "brief the news per the skill. " * 1800
        delta = [{"role": "assistant", "content": "", "tool_calls": [_call("c1", "web_search", "{}")]},
                 {"role": "tool", "tool_call_id": "c1", "content": "OUT-1 headlines"}]
        out = split_delta(delta, pending_user_text=request)
        self.assertEqual(len(out), 1)
        self.assertLess(utf8_len(out[0]), 4_000)
        self.assertIn("PULSE-REQUEST", out[0])
        self.assertIn("appears earlier in this conversation", out[0])
        self.assertNotIn(request, out[0])

    def test_rebuild_still_restates_a_long_request_in_full(self):
        request = "PULSE-REQUEST " + "brief the news per the skill. " * 1800 + "END"
        msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": request},
                {"role": "assistant", "content": "", "tool_calls": [_call("c1", "web_search", "{}")]},
                {"role": "tool", "tool_call_id": "c1", "content": "OUT-1 headlines"}]
        out = pack_refresh(msgs, model="m", tools=TOOLS, tool_choice=None)
        tail = out[-1][out[-1].index("USER REQUEST THIS TOOL LOOP IS SERVING"):]
        self.assertTrue(request in tail, "the rebuild's closing keeps the whole request")

    def test_single_huge_message_is_split_across_seeds(self):
        delta = [{"role": "user", "content": "PASTE " + "q" * 400_000},
                 {"role": "assistant", "content": "ok"}, {"role": "user", "content": "now answer"}]
        out = split_delta(delta)
        self.assertGreater(len(out), 2)
        for payload in out:
            self.assertLess(utf8_len(payload), MAX_STDIN_BYTES)
        self.assertIn("now answer", out[-1])


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = PersistStore(Path(self.tmp.name))
        self.store.records_dir.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_save_is_0600_and_round_trips(self):
        record = _record(REPLY, scope_hash="abc")
        self.store.save(record)
        path = self.store.record_path("abc")
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
        self.assertEqual(self.store.load("abc"), record)

    def test_corrupt_record_reads_as_missing(self):
        self.store.record_path("bad").write_text("{not json")
        self.assertIsNone(self.store.load("bad"))

    def test_mark_needs_refresh(self):
        self.store.save(_record(REPLY, scope_hash="abc"))
        self.store.mark_needs_refresh("abc")
        self.assertTrue(self.store.load("abc")["needs_refresh"])

    def test_gc_drops_old_records_and_their_conversation(self):
        conv_dir = self.store.gemini_dir / "conversations"
        conv_dir.mkdir(parents=True)
        (conv_dir / "old-conv.db").write_text("x")
        self.store.save(_record(REPLY, scope_hash="old", agy_conversation_id="old-conv"))
        self.store.save(_record(REPLY, scope_hash="new"))
        past = time.time() - persist.RECORD_MAX_AGE_SECONDS - 10
        os.utime(self.store.record_path("old"), (past, past))
        self.assertEqual(self.store.gc(), 1)
        self.assertIsNone(self.store.load("old"))
        self.assertIsNotNone(self.store.load("new"))
        self.assertFalse((conv_dir / "old-conv.db").exists())

    def _touch(self, path, age=0.0, text="x"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        when = time.time() - age
        os.utime(path, (when, when))
        return path

    def test_delete_conversation_removes_brain_and_annotations(self):
        g = self.store.gemini_dir
        cid = "71bce5e9-0691-4c20-8e15-a40b154820cd"
        db = self._touch(g / "conversations" / f"{cid}.db")
        wal = self._touch(g / "conversations" / f"{cid}.db-wal")
        ann = self._touch(g / "annotations" / f"{cid}.pbtxt")
        transcript = self._touch(g / "brain" / cid / ".system_generated" / "logs" / "overview.txt")
        other = self._touch(g / "brain" / "25ea8570-0199-42e0-8080-5b2160db9f02" / "x.txt")
        outside = self._touch(Path(self.tmp.name) / "outside" / "keep.txt")
        os.symlink(outside.parent, g / "brain" / cid / "link")
        self.store.delete_conversation(cid)
        for gone in (db, wal, ann, transcript.parents[2]):
            self.assertFalse(gone.exists(), gone)
        self.assertTrue(other.exists())
        self.assertTrue(outside.exists(), "symlinks are not followed")

    def test_gc_removes_orphans_old_logs_implicit_and_idle_locks(self):
        g = self.store.gemini_dir
        day = persist.ORPHAN_MAX_AGE_SECONDS
        live = "aaaaaaaa-0000-0000-0000-000000000001"
        orphan = "bbbbbbbb-0000-0000-0000-000000000002"
        fresh_orphan = "cccccccc-0000-0000-0000-000000000003"
        self.store.save(_record(REPLY, scope_hash="livekey", agy_conversation_id=live))
        live_db = self._touch(g / "conversations" / f"{live}.db", age=10 * day)
        live_brain = self._touch(g / "brain" / live / "t.txt", age=10 * day)
        orphan_db = self._touch(g / "conversations" / f"{orphan}.db", age=2 * day)
        orphan_brain = self._touch(g / "brain" / orphan / "t.txt", age=2 * day)
        os.utime(orphan_brain.parent, (time.time() - 2 * day, time.time() - 2 * day))
        fresh_db = self._touch(g / "conversations" / f"{fresh_orphan}.db", age=60)
        brain_only = "dddddddd-0000-0000-0000-000000000004"
        brain_only_file = self._touch(g / "brain" / brain_only / "new.txt", age=60)
        os.utime(brain_only_file.parent, (time.time() - 3 * day, time.time() - 3 * day))
        old_log = self._touch(g / "log" / "cli-20261101_000000.log", age=4 * day)
        new_log = self._touch(g / "log" / "cli-20261120_000000.log", age=3600)
        old_pb = self._touch(g / "implicit" / "e1.pb", age=15 * day)
        new_pb = self._touch(g / "implicit" / "e2.pb", age=day)
        idle_lock = self._touch(self.store.lock_path("gonekey"), age=2 * day)
        live_lock = self._touch(self.store.lock_path("livekey"), age=20 * day)
        held_lock = self._touch(self.store.lock_path("heldkey"), age=2 * day)
        holder = ConversationLock(held_lock)
        self.assertTrue(holder.acquire(0))
        os.utime(held_lock, (time.time() - 2 * day, time.time() - 2 * day))
        try:
            self.assertEqual(self.store.gc(), 0)
        finally:
            holder.release()
        for kept in (live_db, live_brain, fresh_db, brain_only_file, new_log, new_pb, live_lock, held_lock):
            self.assertTrue(kept.exists(), kept)
        for gone in (orphan_db, orphan_brain.parent, old_log, old_pb, idle_lock):
            self.assertFalse(gone.exists(), gone)

    def test_gc_of_an_old_record_removes_its_brain_directory(self):
        g = self.store.gemini_dir
        cid = "eeeeeeee-0000-0000-0000-000000000005"
        brain = self._touch(g / "brain" / cid / "t.txt")
        self.store.save(_record(REPLY, scope_hash="old", agy_conversation_id=cid))
        past = time.time() - persist.RECORD_MAX_AGE_SECONDS - 10
        os.utime(self.store.record_path("old"), (past, past))
        self.assertEqual(self.store.gc(), 1)
        self.assertFalse(brain.parent.exists())

    def test_kill_9_during_writes_never_leaves_a_corrupt_record(self):
        script = textwrap.dedent(f"""
            import sys
            sys.path.insert(0, {str(REPO)!r})
            from pathlib import Path
            from persist import PersistStore
            store = PersistStore(Path({self.tmp.name!r}))
            i = 0
            print("ready", flush=True)
            while True:
                i += 1
                store.save({{"scope_hash": "k", "n": i, "pad": "p" * 400000}})
        """)
        for _ in range(3):
            proc = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
            self.assertEqual(proc.stdout.readline().strip(), "ready")
            time.sleep(0.3)
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait()
            proc.stdout.close()
            record = self.store.load("k")
            self.assertIsNotNone(record)
            self.assertEqual(len(record["pad"]), 400000)


    def test_gc_keeps_an_old_record_whose_lock_is_held(self):
        conv_dir = self.store.gemini_dir / "conversations"
        conv_dir.mkdir(parents=True)
        (conv_dir / "busy-conv.db").write_text("x")
        self.store.save(_record(REPLY, scope_hash="busy", agy_conversation_id="busy-conv"))
        past = time.time() - persist.RECORD_MAX_AGE_SECONDS - 10
        os.utime(self.store.record_path("busy"), (past, past))
        holder = ConversationLock(self.store.lock_path("busy"))
        self.assertTrue(holder.acquire(0))
        try:
            self.assertEqual(self.store.gc(), 0)
        finally:
            holder.release()
        self.assertIsNotNone(self.store.load("busy"))
        self.assertTrue((conv_dir / "busy-conv.db").exists())
        self.assertEqual(self.store.gc(), 1, "collected once nobody holds it")


class LockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "locks" / "k.lock"

    def tearDown(self):
        self.tmp.cleanup()

    def _holder(self, hold_seconds, log=None):
        script = textwrap.dedent(f"""
            import sys, time
            sys.path.insert(0, {str(REPO)!r})
            from pathlib import Path
            from persist import ConversationLock
            lock = ConversationLock(Path({str(self.path)!r}))
            assert lock.acquire(30)
            log = {str(log) if log else ''!r}
            if log:
                open(log, "a").write("start\\n")
            print("locked", flush=True)
            time.sleep({hold_seconds})
            if log:
                open(log, "a").write("end\\n")
            lock.release()
        """)
        return subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)

    def test_timeout_when_another_process_holds_it(self):
        holder = self._holder(1.5)
        self.assertEqual(holder.stdout.readline().strip(), "locked")
        lock = ConversationLock(self.path)
        self.assertFalse(lock.acquire(timeout=0.2))
        self.assertTrue(lock.acquire(timeout=10))
        lock.release()
        holder.wait()
        holder.stdout.close()

    def test_two_processes_serialize(self):
        log = Path(self.tmp.name) / "log"
        a = self._holder(0.5, log)
        b = self._holder(0.5, log)
        for p in (a, b):
            p.wait()
            p.stdout.close()
        self.assertEqual(log.read_text().split(), ["start", "end", "start", "end"])

    def test_same_process_instances_exclude_each_other_and_release_is_idempotent(self):
        first, second = ConversationLock(self.path), ConversationLock(self.path)
        self.assertTrue(first.acquire(1))
        self.assertFalse(second.acquire(0.1))
        first.release()
        first.release()
        self.assertTrue(second.acquire(0.1))
        second.release()


    def test_a_lock_on_an_unlinked_file_is_not_kept(self):
        # Orphan-lock cleanup may unlink the file a waiter has open; the
        # waiter must then lock the current file, or a newcomer that creates
        # a fresh file holds the "same" lock at the same time.
        import threading

        holder = ConversationLock(self.path)
        self.assertTrue(holder.acquire(0))
        waiter = ConversationLock(self.path)
        got = []
        thread = threading.Thread(target=lambda: got.append(waiter.acquire(5)))
        thread.start()
        time.sleep(0.2)
        self.path.unlink()
        holder.release()
        thread.join(5)
        self.assertEqual(got, [True])
        newcomer = ConversationLock(self.path)
        self.assertFalse(newcomer.acquire(0.1), "the waiter holds the file now at the path")
        waiter.release()
        self.assertTrue(newcomer.acquire(0.1))
        newcomer.release()


if __name__ == "__main__":
    unittest.main()
