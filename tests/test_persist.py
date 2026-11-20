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


def _big_history(n, size, char="x"):
    msgs = [{"role": "system", "content": "You are Hermes."}]
    for i in range(n):
        msgs.append({"role": "user", "content": f"question {i} " + char * size})
        msgs.append({"role": "assistant", "content": f"answer {i} " + char * size})
    return msgs


class PackingTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        for name in ("ANTIGRAVITY_REFRESH_MAX_TOKENS", "ANTIGRAVITY_REFRESH_SIZE_TOKENS"):
            os.environ.pop(name, None)

    def _assert_well_formed(self, out):
        n = len(out)
        for k, payload in enumerate(out, start=1):
            self.assertLess(utf8_len(payload), MAX_STDIN_BYTES)
            if k < n:
                self.assertTrue(payload.startswith(f"[CONTEXT ONLY - PART {k} OF {n}"), payload[:60])
                self.assertIn("Reply with exactly: ACK", payload)
            else:
                self.assertTrue(payload.startswith(f"[PART {n} OF {n}."), payload[:60])

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

    def test_active_loop_too_large_raises_instead_of_truncating(self):
        msgs = [{"role": "user", "content": "go"}]
        for i in range(15):
            msgs.append({"role": "assistant", "content": "a" * 30000,
                         "tool_calls": [_call(f"t{i}", "x", "{}")]})
            msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": "z" * 1000})
        with self.assertRaises(StdinPayloadTooLarge):
            pack_refresh(msgs, model="m", tools=None, tool_choice=None)

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
            "intro\n[Recent Summary (d0, node 12)]\nthe deploy key lives in vault B",
            "[Session Arc Summary (d1, node 3)]\n...",
            "body\n--- END OF CONTEXT SUMMARY — respond to the message below, not the summary above ---",
        ]
        for text in summaries:
            with self.subTest(text[:30]):
                self.assertTrue(persist.is_summary_message({"role": "user", "content": text}))
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


if __name__ == "__main__":
    unittest.main()
