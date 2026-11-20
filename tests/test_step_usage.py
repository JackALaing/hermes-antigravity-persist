"""Per-turn usage from agy step_update events (persistent-session build spec, usage).

Event shapes and numbers are taken from live agy 1.x captures: each finished
agent_response step carries its own usage, input_tokens excludes cache reads,
and result.usage accumulates for the life of the process.
"""

import io
import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from stream import AntigravityStream, _StepUsage


def _step(conv, index, usage=None, text=None):
    step = {"conversation_id": conv, "step_index": index, "state": "DONE",
            "step_type": "agent_response"}
    if text is not None:
        step["text_delta"] = text
    if usage is not None:
        step["usage"] = usage
    return {"event": "step_update", "step_update": step}


def _result(conv, usage, response="ok"):
    return {"event": "result", "result": {"conversation_id": conv, "status": "SUCCESS",
                                          "response": response, "usage": usage}}


def _worker_stream(lines, baseline):
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
                             usage_baseline=baseline)


def _usage(chunks):
    return [c for c in chunks if getattr(c, "usage", None)][-1].usage


class StepUsageTests(unittest.TestCase):
    def test_step_usage_accumulates_output_and_keeps_last_context(self):
        su = _StepUsage()
        su.add({"input_tokens": 9856, "output_tokens": 133, "cache_read_tokens": 0})
        su.add({"input_tokens": 2344, "output_tokens": 43, "cache_read_tokens": 7799})
        self.assertTrue(su.seen)
        self.assertEqual(su.context_tokens, 2344 + 7799)
        self.assertEqual(su.output_sum, 176)
        self.assertEqual(su.first_context, 9856)

    def test_turns_on_one_process_report_their_own_usage(self):
        # t5: two seeds then the real turn on one process; result.usage is
        # cumulative (49,415 -> 86,736 -> 88,981) but each turn must report itself.
        baseline: dict = {}
        turns = [
            ({"input_tokens": 49415, "output_tokens": 75, "cache_read_tokens": 0},
             {"input_tokens": 49415, "output_tokens": 75, "cache_read_tokens": 0}),
            ({"input_tokens": 37321, "output_tokens": 63, "cache_read_tokens": 49416},
             {"input_tokens": 86736, "output_tokens": 138, "cache_read_tokens": 49416}),
            ({"input_tokens": 2245, "output_tokens": 496, "cache_read_tokens": 84688},
             {"input_tokens": 88981, "output_tokens": 634, "cache_read_tokens": 134104}),
        ]
        reported = []
        for step_usage, result_usage in turns:
            chunks = list(_worker_stream(
                [_step("c", 1, step_usage, "x"), _result("c", result_usage)], baseline))
            u = _usage(chunks)
            reported.append((u.prompt_tokens, u.prompt_tokens_details.cached_tokens,
                             u.completion_tokens))
        self.assertEqual(reported[2], (2245 + 84688, 84688, 496))
        self.assertEqual(reported[1], (37321 + 49416, 49416, 63))
        # Baseline tracks the cumulative counters so a fallback turn measures itself.
        self.assertEqual(baseline["input_tokens"], 88981)

    def test_first_turn_after_resume_uses_steps_not_process_totals(self):
        # t2 resume: the new process restarts its counters; the step usage is the turn.
        chunks = list(_worker_stream([
            _step("c", 4, {"input_tokens": 2329, "output_tokens": 77, "cache_read_tokens": 7790}, "a"),
            _result("c", {"input_tokens": 12176, "output_tokens": 131, "cache_read_tokens": 7790}),
        ], {}))
        u = _usage(chunks)
        self.assertEqual((u.prompt_tokens, u.completion_tokens), (2329 + 7790, 77))
        self.assertEqual(u.total_tokens, 2329 + 7790 + 77)

    def test_fallback_without_steps_never_forwards_cumulative_counters(self):
        baseline = {"input_tokens": 1000, "output_tokens": 100, "cache_read_tokens": 500}
        chunks = list(_worker_stream(
            [_result("c", {"input_tokens": 1200, "output_tokens": 130, "cache_read_tokens": 900})],
            baseline))
        u = _usage(chunks)
        self.assertEqual((u.prompt_tokens, u.completion_tokens), (200 + 400, 30))
        self.assertEqual(u.prompt_tokens_details.cached_tokens, 400)


if __name__ == "__main__":
    unittest.main()
