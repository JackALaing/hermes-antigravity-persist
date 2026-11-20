"""Per-turn cache log: cache reads, hit %, idle gap and how each turn was sent.

Gemini's implicit prompt cache is reported to decay with the idle time
between requests, so every completed turn logs one INFO line that lets a
live run correlate hit rate with the gap and the send path.
"""

import io
import json
import logging
import tempfile
import threading
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import stream as stream_mod
from client import AntigravityClient
from stream import AntigravityStream, _StepUsage, cache_hit_pct, log_turn_cache


def _step(usage, text="x"):
    return {"event": "step_update", "step_update": {
        "conversation_id": "conv-1234abcd", "step_type": "agent_response", "state": "DONE",
        "text_delta": text, "usage": usage}}


def _result(usage=None):
    return {"event": "result", "result": {"conversation_id": "conv-1234abcd", "status": "SUCCESS",
                                          "response": "ok", "usage": usage or {}}}


def _stream(lines, **kwargs):
    proc = MagicMock()
    proc.stdout = io.StringIO("".join(json.dumps(e) + "\n" for e in lines))
    proc.poll.return_value = None
    client = SimpleNamespace(
        _lock=threading.RLock(), _active_processes={proc}, _terminate_process=lambda p: None,
        _worker_proc=proc, _terminate_worker=lambda: None,
        _update_worker_history=lambda m: None, _note_worker_activity=lambda p: None,
        _set_worker_conversation_id=lambda c: None, _cancelled=False,
    )
    return AntigravityStream(proc=proc, client=client, model="m", timeout=5, is_worker=True,
                             usage_baseline={}, **kwargs)


def _cache_lines(logs):
    return [r.getMessage() for r in logs.records if r.getMessage().startswith("Antigravity turn cache:")]


def _fields(line):
    return dict(part.split("=", 1) for part in line.split(": ", 1)[1].split() if "=" in part)


class CacheLogFormatTests(unittest.TestCase):
    def test_hit_pct_uses_input_excluding_cache_reads(self):
        self.assertAlmostEqual(cache_hit_pct(2172, 18704), 89.595, places=2)
        self.assertEqual(cache_hit_pct(0, 0), 0.0)
        self.assertEqual(cache_hit_pct(20761, 0), 0.0)

    def test_first_step_leads_and_multi_step_adds_turn_totals(self):
        su = _StepUsage()
        su.add({"input_tokens": 2000, "cache_read_tokens": 18000, "output_tokens": 5})
        su.add({"input_tokens": 500, "cache_read_tokens": 19500, "output_tokens": 5})
        with self.assertLogs(level="INFO") as logs:
            log_turn_cache(su, kind="resume", conversation_key="persist:abc", gap=91.24,
                           agy_conversation="e94c86b6-c157")
        f = _fields(_cache_lines(logs)[0])
        self.assertEqual((f["kind"], f["conv"], f["agy"]), ("resume", "persist:abc", "e94c86b6"))
        self.assertEqual((f["input"], f["cache_read"], f["hit"]), ("2000", "18000", "90.0%"))
        self.assertEqual((f["gap"], f["steps"]), ("91.2s", "2"))
        self.assertEqual((f["turn_input"], f["turn_cache_read"], f["turn_hit"]),
                         ("2500", "37500", "93.8%"))

    def test_without_step_usage_falls_back_and_unknown_gap_reads_na(self):
        with self.assertLogs(level="INFO") as logs:
            log_turn_cache(_StepUsage(), kind="one-shot", conversation_key="", gap=None,
                           fallback=(300, 700))
        f = _fields(_cache_lines(logs)[0])
        self.assertEqual((f["input"], f["cache_read"], f["hit"], f["gap"], f["conv"]),
                         ("300", "700", "70.0%", "n/a", "-"))
        self.assertNotIn("turn_hit", f)


class StreamCacheLogTests(unittest.TestCase):
    def test_gap_runs_from_previous_turn_end_to_this_send(self):
        key = f"worker:{uuid.uuid4().hex}"
        usage = {"input_tokens": 100, "cache_read_tokens": 900, "output_tokens": 3}
        with patch("time.time", return_value=1000.0):
            first = _stream([_step(usage), _result()], turn_kind="worker-full", conversation_key=key)
        with self.assertLogs(level="INFO") as logs, patch("time.time", return_value=1010.0):
            list(first)  # this turn ends at 1010
        self.assertEqual(_fields(_cache_lines(logs)[0])["gap"], "n/a")
        with patch("time.time", return_value=1100.5):
            second = _stream([_step(usage), _result()], turn_kind="worker-delta", conversation_key=key)
        with self.assertLogs(level="INFO") as logs, patch("time.time", return_value=1104.0):
            list(second)
        f = _fields(_cache_lines(logs)[0])
        self.assertEqual((f["kind"], f["gap"], f["hit"], f["agy"]),
                         ("worker-delta", "90.5s", "90.0%", "conv-123"))

    def test_persisted_end_time_covers_a_turn_this_process_never_saw(self):
        key = f"persist:{uuid.uuid4().hex}"
        usage = {"input_tokens": 50, "cache_read_tokens": 0, "output_tokens": 3}
        with patch("time.time", return_value=5000.0):
            s = _stream([_step(usage), _result()], turn_kind="resume", conversation_key=key,
                        previous_turn_end=4988.0)
        with self.assertLogs(level="INFO") as logs, patch("time.time", return_value=5003.0):
            list(s)
        self.assertEqual(_fields(_cache_lines(logs)[0])["gap"], "12.0s")

    def test_stream_without_conversation_key_logs_no_gap(self):
        usage = {"input_tokens": 50, "cache_read_tokens": 150, "output_tokens": 3}
        with self.assertLogs(level="INFO") as logs:
            list(_stream([_step(usage), _result()], turn_kind="one-shot"))
        f = _fields(_cache_lines(logs)[0])
        self.assertEqual((f["kind"], f["gap"], f["hit"]), ("one-shot", "n/a", "75.0%"))

    def test_a_failing_cache_log_never_fails_the_turn(self):
        usage = {"input_tokens": 50, "cache_read_tokens": 150, "output_tokens": 3}
        with patch.object(stream_mod, "log_turn_cache", side_effect=RuntimeError("boom")):
            chunks = list(_stream([_step(usage), _result()], turn_kind="resume", conversation_key="k"))
        self.assertEqual([c for c in chunks if getattr(c, "usage", None)][-1].usage.prompt_tokens, 200)


class WorkerPathKindTests(unittest.TestCase):
    def test_per_client_worker_logs_full_then_delta_with_a_gap(self):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        client = AntigravityClient(cwd=tmp_dir.name)
        self.addCleanup(client.close)
        proc = MagicMock()
        proc.poll.return_value = None
        proc.stdin = MagicMock()
        events = [json.dumps({"event": "init", "conversation_id": "conv-worker"}) + "\n",
                  json.dumps(_step({"input_tokens": 10, "cache_read_tokens": 30})) + "\n",
                  json.dumps(_result()) + "\n"]
        proc.stdout.readline.side_effect = events * 2 + [""]
        msgs1 = [{"role": "user", "content": "first turn"}]
        msgs2 = msgs1 + [{"role": "tool", "tool_call_id": "t1", "content": "res"}]
        with patch("subprocess.Popen", return_value=proc), patch("client.is_authenticated", return_value=True), \
                self.assertLogs(level="INFO") as logs:
            list(client.chat.completions.create(model="gemini-3.8-flash", messages=msgs1, stream=True))
            list(client.chat.completions.create(model="gemini-3.8-flash", messages=msgs2, stream=True))
        lines = [_fields(line) for line in _cache_lines(logs)]
        self.assertEqual([f["kind"] for f in lines], ["worker-full", "worker-delta"])
        self.assertTrue(lines[0]["conv"].startswith("worker:"))
        self.assertEqual(lines[0]["conv"], lines[1]["conv"])
        self.assertEqual(lines[0]["gap"], "n/a")
        self.assertTrue(lines[1]["gap"].endswith("s"))


if __name__ == "__main__":
    unittest.main()
