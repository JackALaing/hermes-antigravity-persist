"""Persistent agy conversations through AntigravityClient (session.py).

A scripted fake agy stands in for the CLI: it answers the init handshake on
the first stdin write, acknowledges context-only parts with ACK, answers real
turns with per-step usage, and records every message it received.
"""

import json
import os
import queue
import sys
import tempfile
import threading
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

import persist  # noqa: E402
import session  # noqa: E402
import stream as stream_mod  # noqa: E402
from client import AntigravityClient  # noqa: E402
from persist import PersistStore, scope_key  # noqa: E402

MODEL = "gemini-3.8-flash-high"
SCOPE = "session-abc"
TOOLS = [{"type": "function", "function": {"name": "terminal", "parameters": {"type": "object"}}}]


class _Stdin:
    def __init__(self, proc):
        self.proc = proc

    def write(self, data):
        for line in data.splitlines():
            if line.strip():
                self.proc.on_message(json.loads(line)["message"]["content"])

    def flush(self):
        pass

    def close(self):
        pass


class _Stdout:
    def __init__(self):
        self.lines = queue.Queue()

    def readline(self):
        try:
            return self.lines.get(timeout=0.05)
        except queue.Empty:
            return ""

    def close(self):
        pass


class FakeAgy:
    """One fake agy process. ``world`` holds shared conversation state."""

    def __init__(self, world, cmd):
        self.world = world
        self.cmd = cmd
        self.pid = 4242
        self.returncode = None
        self.stdin = _Stdin(self)
        self.stdout = _Stdout()
        self.stderr = None
        self.received = []
        self.cumulative = {}
        self.first = True
        resume = cmd[cmd.index("--conversation") + 1] if "--conversation" in cmd else None
        if resume is not None and resume in world.conversations and not world.lose_conversations:
            self.conversation = resume
        else:
            self.conversation = str(uuid.uuid4())
            world.conversations[self.conversation] = []
            conv_dir = world.gemini_dir / "conversations"
            conv_dir.mkdir(parents=True, exist_ok=True)
            (conv_dir / f"{self.conversation}.db").write_text("db")
        world.procs.append(self)

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode or 0

    def kill(self):
        self.returncode = -9

    terminate = kill

    def _emit(self, event):
        self.stdout.lines.put(json.dumps(event) + "\n")

    def on_message(self, content):
        self.received.append(content)
        self.world.conversations[self.conversation].append(content)
        if self.first:
            self.first = False
            self._emit({"event": "init", "conversation_id": self.conversation,
                        "init": {"model": "m"}})
        if content.startswith("[CONTEXT ONLY"):
            reply = self.world.seed_reply
        else:
            reply = self.world.next_reply()
        n = len(self.world.conversations[self.conversation])
        usage = {"input_tokens": 100 * n, "output_tokens": 7, "cache_read_tokens": 1000 * n}
        step = {"conversation_id": self.conversation, "step_type": "agent_response",
                "text_delta": reply}
        if not self.world.no_step_usage:
            step["usage"] = usage
        self._emit({"event": "step_update", "step_update": step})
        # result.usage accumulates over the process lifetime, as agy's does.
        for field, value in usage.items():
            self.cumulative[field] = self.cumulative.get(field, 0) + value
        self._emit({"event": "result", "result": {
            "conversation_id": self.conversation, "status": "SUCCESS", "response": reply,
            "usage": dict(self.cumulative)}})


class World:
    def __init__(self, gemini_dir):
        self.gemini_dir = gemini_dir
        self.conversations = {}
        self.procs = []
        self.replies = []
        self.seed_reply = "ACK"
        self.lose_conversations = False
        self.no_step_usage = False

    def next_reply(self):
        return self.replies.pop(0) if self.replies else f"reply {len(self.procs)}"

    def popen(self, cmd, **kwargs):
        return FakeAgy(self, cmd)


def _fake_setup_home(cwd):
    home = Path(cwd) / "home"
    gemini = home / ".gemini" / "antigravity-cli"
    gemini.mkdir(parents=True, exist_ok=True)
    return home, gemini


class PersistentSessionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name) / "persist"
        self.store = PersistStore(root)
        self.world = World(self.store.gemini_dir)
        patches = [
            patch.dict(os.environ, {"ANTIGRAVITY_PERSIST_DIR": str(root),
                                    "ANTIGRAVITY_WORKER_IDLE_SECONDS": "0"}),
            patch("client.is_authenticated", return_value=True),
            patch("client.resolve_agy_command", return_value="agy"),
            patch("process.resolve_real_token_path", return_value=None),
            patch("process._link_macos_keychains"),
            patch("process.setup_isolated_home", side_effect=_fake_setup_home),
            patch("session.sync_isolated_token"),
            patch("session.start_stderr_drainer"),
            patch("session.register_tracked_process"),
            patch("session.terminate_process", side_effect=lambda p: p.kill()),
            patch("session.subprocess.Popen", side_effect=self.world.popen),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(session._REGISTRY.clear)
        self.key = scope_key(SCOPE)

    def _client(self):
        cwd = tempfile.TemporaryDirectory(prefix="hermes_agy_test_")
        self.addCleanup(cwd.cleanup)
        with patch("client.setup_isolated_home", side_effect=_fake_setup_home):
            client = AntigravityClient(cwd=cwd.name)
        self.addCleanup(client.close)
        return client

    def _create(self, client, messages, *, origin="assistant_tool", scope=SCOPE, stream=False,
                tools=TOOLS, model=MODEL, effort=None):
        kwargs = {"model": model, "messages": messages, "tools": tools, "stream": stream}
        if effort:
            kwargs["reasoning_effort"] = effort
        if scope is not None:
            kwargs["_agy_cache_scope"] = scope
            kwargs["_agy_write_origin"] = origin
        return client.chat.completions.create(**kwargs)

    def _record(self):
        return self.store.load(self.key)

    @staticmethod
    def _reply(completion):
        msg = completion.choices[0].message
        out = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            out["tool_calls"] = [{"id": tc.id, "type": "function",
                                  "function": {"name": tc.function.name,
                                               "arguments": tc.function.arguments}}
                                 for tc in msg.tool_calls]
        return out

    def _two_turns(self, client):
        m1 = [{"role": "system", "content": "sys"}, {"role": "user", "content": "first question"}]
        c1 = self._create(client, m1)
        m2 = m1 + [self._reply(c1), {"role": "user", "content": "second question"}]
        c2 = self._create(client, m2)
        return m2, c2

    # -- resume and refresh ------------------------------------------------
    def test_first_turn_refreshes_then_resume_sends_only_new_messages(self):
        client = self._client()
        m2, c2 = self._two_turns(client)
        self.assertEqual(len(self.world.procs), 1, "second turn reuses the live worker")
        received = self.world.procs[0].received
        self.assertEqual(len(received), 2)
        self.assertIn("second question", received[1])
        self.assertNotIn("first question", received[1])
        self.assertNotIn("reply 1", received[1], "the last reply is not re-sent")
        record = self._record()
        self.assertEqual(record["agy_conversation_id"], self.world.procs[0].conversation)
        self.assertEqual(record["last_reply_fp"], persist.reply_fingerprint(self._reply(c2)))
        self.assertFalse(record["needs_refresh"])
        self.assertEqual(os.stat(self.store.record_path(self.key)).st_mode & 0o777, 0o600)

    def test_recreated_client_resumes_after_worker_exit_with_conversation_flag(self):
        client = self._client()
        m1 = [{"role": "user", "content": "plant the fact"}]
        c1 = self._create(client, m1)
        self.world.procs[0].kill()  # process gone (idle reap, crash, restart)
        session._REGISTRY.clear()
        m2 = m1 + [self._reply(c1), {"role": "user", "content": "recall it"}]
        c2 = self._create(self._client(), m2)
        self.assertEqual(len(self.world.procs), 2)
        second = self.world.procs[1]
        self.assertIn("--conversation", second.cmd)
        self.assertEqual(second.conversation, self.world.procs[0].conversation)
        self.assertEqual(len(second.received), 1)
        self.assertIn("recall it", second.received[0])
        usage = c2.usage
        # Second message on the resumed conversation: this step's context
        # (which includes the planted turn), not a per-process counter.
        self.assertEqual((usage.prompt_tokens, usage.completion_tokens), (200 + 2000, 7))

    def test_turns_log_refresh_then_resume_with_cache_hit_and_gap(self):
        client = self._client()
        with self.assertLogs(level="INFO") as logs:
            self._two_turns(client)
        lines = [r.getMessage() for r in logs.records
                 if r.getMessage().startswith("Antigravity turn cache:")]
        self.assertEqual(len(lines), 2)
        fields = [dict(p.split("=", 1) for p in line.split(": ", 1)[1].split()) for line in lines]
        self.assertEqual([f["kind"] for f in fields], ["refresh", "resume"])
        self.assertEqual({f["conv"] for f in fields}, {f"persist:{self.key}"[:20]})
        # The fake agy's second message: input 200, cache reads 2000.
        self.assertEqual((fields[1]["input"], fields[1]["cache_read"], fields[1]["hit"]),
                         ("200", "2000", "90.9%"))
        self.assertTrue(fields[1]["gap"].endswith("s"))

    def test_resume_in_a_new_process_takes_the_gap_from_the_record(self):
        client = self._client()
        m1 = [{"role": "user", "content": "plant"}]
        c1 = self._create(client, m1)
        record = self._record()
        assert record is not None
        record["updated_at"] -= 95.0  # the previous turn ended 95s ago
        self.store.save(record)
        self.world.procs[0].kill()
        session._REGISTRY.clear()
        stream_mod._TURN_ENDS.clear()  # as in a fresh Hermes process
        m2 = m1 + [self._reply(c1), {"role": "user", "content": "recall"}]
        with self.assertLogs(level="INFO") as logs:
            self._create(self._client(), m2)
        line = [r.getMessage() for r in logs.records
                if r.getMessage().startswith("Antigravity turn cache:")][-1]
        self.assertIn("kind=resume", line)
        gap = float(line.split("gap=", 1)[1].split("s", 1)[0])
        self.assertGreaterEqual(gap, 95.0)
        self.assertLess(gap, 120.0)

    def test_resume_rejected_when_conversation_not_found_refreshes(self):
        client = self._client()
        m1 = [{"role": "user", "content": "plant"}]
        c1 = self._create(client, m1)
        old = self.world.procs[0].conversation
        self.world.procs[0].kill()
        session._REGISTRY.clear()
        self.world.lose_conversations = True  # agy opens a different conversation
        m2 = m1 + [self._reply(c1), {"role": "user", "content": "recall"}]
        self._create(self._client(), m2)
        refreshed = self.world.procs[-1]
        self.assertNotIn("--conversation", refreshed.cmd)
        self.assertIn("plant", refreshed.received[0], "refresh re-sends the history")
        record = self._record()
        self.assertEqual(record["agy_conversation_id"], refreshed.conversation)
        conv_dir = self.store.gemini_dir / "conversations"
        self.assertFalse((conv_dir / f"{old}.db").exists(), "old conversation deleted after commit")
        self.assertFalse((conv_dir / f"{self.world.procs[1].conversation}.db").exists(),
                         "the wrongly opened conversation is discarded")

    def test_each_record_trigger_forces_a_refresh(self):
        triggers = {
            "model changed": lambda r: r.update(model="other"),
            "tools changed": lambda r: r.update(tools_hash="x"),
            "size": lambda r: r.update(agy_size_tokens=persist.REFRESH_SIZE_TOKENS + 1),
            "needs refresh": lambda r: r.update(needs_refresh=True),
            "fingerprint": lambda r: r.update(last_reply_fp="0" * 64),
        }
        for name, mutate in triggers.items():
            with self.subTest(name):
                session._REGISTRY.clear()
                self.world.procs.clear()
                client = self._client()
                m1 = [{"role": "user", "content": f"q-{name}"}]
                c1 = self._create(client, m1)
                record = self._record()
                mutate(record)
                self.store.save(record)
                m2 = m1 + [self._reply(c1), {"role": "user", "content": "next"}]
                self._create(client, m2)
                self.assertEqual(len(self.world.procs), 2)
                self.assertNotIn("--conversation", self.world.procs[1].cmd)
                self.assertIn(f"q-{name}", self.world.procs[1].received[0])

    def test_effort_change_respawns_with_conversation_not_refresh(self):
        client = self._client()
        m1 = [{"role": "user", "content": "q"}]
        c1 = self._create(client, m1, model="gemini-3.8-flash", effort="low")
        m2 = m1 + [self._reply(c1), {"role": "user", "content": "deeper"}]
        self._create(client, m2, model="gemini-3.8-flash", effort="high")
        self.assertEqual(len(self.world.procs), 2)
        self.assertIn("--conversation", self.world.procs[1].cmd)

    # -- classification ------------------------------------------------------
    def test_side_calls_run_as_one_shots_and_leave_the_record(self):
        client = self._client()
        m2, c2 = self._two_turns(client)
        before = self._record()
        worker = self.world.procs[0]
        oneshots = []

        def fake_oneshot(**kw):
            oneshots.append(kw)
            return "oneshot-result"

        side = m2 + [self._reply(c2), {"role": "user", "content": "review this"}]
        with patch.object(AntigravityClient, "_run_oneshot_completion", side_effect=fake_oneshot), \
                patch.object(AntigravityClient, "_get_or_spawn_worker",
                             side_effect=AssertionError("per-client worker must not be used")):
            other = self._client()
            other._worker_lock.acquire()  # force the existing path's one-shot branch
            for origin, scope in (("background_review", SCOPE), ("side_question", SCOPE),
                                  ("assistant_tool", None)):
                self.assertEqual(self._create(other, side, origin=origin, scope=scope),
                                 "oneshot-result")
            other._worker_lock.release()
        self.assertEqual(len(oneshots), 3)
        self.assertEqual(self._record(), before)
        self.assertEqual(len(worker.received), 2, "persistent worker untouched")

    # -- packing, seeds and the byte guard ---------------------------------
    def test_large_refresh_sends_seeds_internally_and_streams_only_the_answer(self):
        client = self._client()
        history = [{"role": "system", "content": "SYS"}]
        for i in range(120):
            history += [{"role": "user", "content": f"u{i} " + "é" * 1500},
                        {"role": "assistant", "content": f"a{i} " + "é" * 1500}]
        history.append({"role": "user", "content": "FINAL ASK"})
        self.world.replies = ["the real answer"]
        chunks = list(self._create(client, history, stream=True))
        text = "".join(c.choices[0].delta.content or "" for c in chunks if c.choices)
        self.assertEqual(text, "the real answer")
        self.assertNotIn("ACK", text)
        proc = self.world.procs[0]
        n = len(proc.received)
        self.assertGreater(n, 2, "a ~720 KB history grows into several seeds")
        for k, payload in enumerate(proc.received, start=1):
            self.assertLess(persist.utf8_len(payload), persist.MAX_STDIN_BYTES)
            if k < n:
                self.assertTrue(payload.startswith(f"[CONTEXT ONLY - PART {k} OF {n}"))
        self.assertIn("FINAL ASK", proc.received[-1])
        joined = "".join(proc.received)
        self.assertIn("u0 ", joined)
        self.assertIn("a119 ", joined)
        record = self._record()
        self.assertGreater(record["seed_tokens_total"], 0)
        # Usage is the real turn's step usage only (n-th message on this conversation).
        usage = [c for c in chunks if getattr(c, "usage", None)][-1].usage
        self.assertEqual(usage.prompt_tokens, 100 * n + 1000 * n)
        self.assertEqual(usage.completion_tokens, 7)

    def test_each_seed_and_the_real_turn_get_the_full_timeout(self):
        client = self._client()
        history = []
        for i in range(120):
            history += [{"role": "user", "content": f"u{i} " + "x" * 3000},
                        {"role": "assistant", "content": f"a{i} " + "x" * 3000}]
        history.append({"role": "user", "content": "FINAL"})
        clock = [1000.0]
        consumed = []
        original = session.PersistentTurn.consume_seed

        def slow_seed(turn, pending, deadline):
            consumed.append(deadline - clock[0])
            original(turn, pending, deadline)
            clock[0] += 50.0  # each ACK takes 50s of a 60s timeout

        with patch("session.time.monotonic", side_effect=lambda: clock[0]), \
                patch.object(session.PersistentTurn, "consume_seed", slow_seed):
            client.chat.completions.create(model=MODEL, messages=history, tools=TOOLS, timeout=60,
                                           _agy_cache_scope=SCOPE, _agy_write_origin="assistant_tool")
        self.assertGreater(len(consumed), 1)
        self.assertTrue(all(budget == 60.0 for budget in consumed), consumed)
        self.assertFalse(self._record()["needs_refresh"])

    def test_seed_advances_usage_baseline_for_result_usage_fallback(self):
        client = self._client()
        self.world.no_step_usage = True
        history = []
        for i in range(120):
            history += [{"role": "user", "content": f"u{i} " + "x" * 3000},
                        {"role": "assistant", "content": f"a{i} " + "x" * 3000}]
        history.append({"role": "user", "content": "FINAL"})
        completion = self._create(client, history)
        n = len(self.world.procs[0].received)
        self.assertGreater(n, 2)
        # result.usage after the turn is cumulative over every seed and the
        # turn; only the real turn's increment is reported.
        self.assertEqual(completion.usage.prompt_tokens, 100 * n + 1000 * n)
        self.assertEqual(completion.usage.completion_tokens, 7)

    def test_non_ack_seed_aborts_discards_conversation_and_raises(self):
        client = self._client()
        history = []
        for i in range(120):
            history += [{"role": "user", "content": f"u{i} " + "x" * 3000},
                        {"role": "assistant", "content": f"a{i} " + "x" * 3000}]
        history.append({"role": "user", "content": "FINAL"})
        self.world.seed_reply = "Sure, here is a summary of the context"
        with self.assertRaisesRegex(RuntimeError, "not acknowledged"):
            self._create(client, history)
        proc = self.world.procs[0]
        self.assertEqual(len(proc.received), 1, "the real turn is never sent")
        self.assertEqual(proc.returncode, -9)
        conv_dir = self.store.gemini_dir / "conversations"
        self.assertFalse((conv_dir / f"{proc.conversation}.db").exists())
        self.assertNotIn((str(self.store.root), self.key), session._REGISTRY)
        # The lock was released: the next turn proceeds.
        self.world.seed_reply = "ACK"
        self._create(client, history)

    def test_oversized_resume_delta_is_split_with_request_last(self):
        client = self._client()
        m1 = [{"role": "user", "content": "start"}]
        c1 = self._create(client, m1)
        delta_source = m1 + [self._reply(c1),
                             {"role": "user", "content": "paste big " + "日" * 70000},
                             {"role": "user", "content": "THE QUESTION"}]
        self._create(client, delta_source)
        received = self.world.procs[0].received[1:]
        self.assertGreater(len(received), 1)
        for payload in received[:-1]:
            self.assertTrue(payload.startswith("[CONTEXT ONLY"))
            self.assertLess(persist.utf8_len(payload), persist.MAX_STDIN_BYTES)
        self.assertIn("THE QUESTION", received[-1])
        self.assertLess(persist.utf8_len(received[-1]), persist.MAX_STDIN_BYTES)

    # -- failure, cancel, locking ------------------------------------------
    def test_failed_turn_marks_needs_refresh_and_releases_lock(self):
        client = self._client()
        m1 = [{"role": "user", "content": "q"}]
        c1 = self._create(client, m1)
        proc = self.world.procs[0]
        original = proc.on_message

        def die(content):
            proc.received.append(content)
            proc.returncode = 1

        proc.on_message = die
        m2 = m1 + [self._reply(c1), {"role": "user", "content": "boom"}]
        with self.assertRaises(RuntimeError):
            self._create(client, m2)
        proc.on_message = original
        self.assertTrue(self._record()["needs_refresh"])
        lock = persist.ConversationLock(self.store.lock_path(self.key))
        self.assertTrue(lock.acquire(0))
        lock.release()

    def test_cancel_mid_stream_marks_needs_refresh(self):
        client = self._client()
        m1 = [{"role": "user", "content": "q"}]
        c1 = self._create(client, m1)
        m2 = m1 + [self._reply(c1), {"role": "user", "content": "long task"}]
        stream = self._create(client, m2, stream=True)
        client.cancel()
        with self.assertRaises(RuntimeError):
            list(stream)
        self.assertTrue(self._record()["needs_refresh"])

    def test_lock_timeout_falls_back_to_one_shot_and_leaves_record(self):
        client = self._client()
        self._create(client, [{"role": "user", "content": "q"}])
        before = self._record()
        holder = persist.ConversationLock(self.store.lock_path(self.key))
        self.assertTrue(holder.acquire(0))
        try:
            with patch.object(persist, "LOCK_TIMEOUT_SECONDS", 0.1), \
                    patch.object(AntigravityClient, "_run_oneshot_completion",
                                 return_value="oneshot") as oneshot:
                self.assertEqual(self._create(client, [{"role": "user", "content": "other"}]),
                                 "oneshot")
            oneshot.assert_called_once()
        finally:
            holder.release()
        self.assertEqual(self._record(), before)

    def test_concurrent_turns_on_one_conversation_serialize(self):
        client_a, client_b = self._client(), self._client()
        m1 = [{"role": "user", "content": "q"}]
        stream_a = self._create(client_a, m1, stream=True)
        results = []

        def second():
            results.append(self._create(client_b, m1))

        t = threading.Thread(target=second)
        t.start()
        t.join(0.5)
        self.assertTrue(t.is_alive(), "second turn waits for the first turn's lock")
        list(stream_a)
        t.join(10)
        self.assertFalse(t.is_alive())
        self.assertEqual(len(results), 1)


class KwargsForwardingTests(unittest.TestCase):
    def test_scope_and_origin_forwarded_only_with_a_scope(self):
        from providers import get_provider_profile
        from tools.skill_provenance import reset_current_write_origin, set_current_write_origin

        profile = get_provider_profile("antigravity-subscription-directsdk")
        token = set_current_write_origin("assistant_tool")
        try:
            _, top = profile.build_api_kwargs_extras(model=MODEL, cache_scope_id="scope-1", session_id="s")
            self.assertEqual(top["_agy_cache_scope"], "scope-1")
            self.assertEqual(top["_agy_write_origin"], "assistant_tool")
            _, top = profile.build_api_kwargs_extras(model=MODEL, session_id=None, cache_scope_id=None)
            self.assertNotIn("_agy_cache_scope", top)
        finally:
            reset_current_write_origin(token)
        token = set_current_write_origin("background_review")
        try:
            _, top = profile.build_api_kwargs_extras(model=MODEL, cache_scope_id="scope-1")
            self.assertEqual(top["_agy_write_origin"], "background_review")
        finally:
            reset_current_write_origin(token)

    def test_private_kwargs_never_reach_the_per_client_path(self):
        self.assertTrue(persist.is_persistent_request("s", "assistant_tool"))
        for scope, origin in (("s", "background_review"), ("s", "side_question"), ("", "assistant_tool"),
                              (None, "assistant_tool"), ("s", "foreground")):
            self.assertFalse(persist.is_persistent_request(scope, origin))


if __name__ == "__main__":
    unittest.main()
