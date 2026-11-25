"""The user request a tool loop serves must stay in every tool-result prompt.

Regression for long tool loops on the persistent worker: the delta prompt for a
tool-result turn used to end with a generic "continue" line only, so after
several tool calls the model lost the user's question and re-sent an earlier
report from the transcript instead of answering it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

plugin_dir = Path(__file__).resolve().parent.parent
if str(plugin_dir) not in sys.path:
    sys.path.insert(0, str(plugin_dir))

from prompt import (  # noqa: E402
    PENDING_REQUEST_FULL_MAX_CHARS,
    _delta_parts,
    _format_delta_prompt,
    _format_messages_as_prompt,
    _latest_user_text,
)

QUESTION = "Is there any historical PO ledger data at all?"


def _history_with_tool_loop() -> list[dict]:
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Hide the menu and deploy."},
        {"role": "assistant", "content": "### Completion report\nMenu hidden and deployed."},
        {"role": "user", "content": QUESTION},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c1", "function": {"name": "terminal", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "supply.db 108K"},
    ]


class TestToolLoopKeepsUserRequest(unittest.TestCase):
    def test_latest_user_text_picks_last_user_message(self):
        self.assertEqual(_latest_user_text(_history_with_tool_loop()), QUESTION)
        self.assertEqual(_latest_user_text([{"role": "assistant", "content": "x"}]), "")

    def test_delta_tool_prompt_restates_pending_request_after_results(self):
        history = _history_with_tool_loop()
        delta = history[-2:]  # assistant tool_call + tool result: no user message
        prompt = _format_delta_prompt(delta, pending_user_text=_latest_user_text(history))

        self.assertIn(QUESTION, prompt)
        # The request is the newest text the model reads, after the tool output.
        self.assertGreater(prompt.rindex(QUESTION), prompt.index("supply.db 108K"))
        self.assertIn("Do NOT repeat earlier reports", prompt)

    def test_delta_tool_prompt_without_pending_text_is_backward_compatible(self):
        delta = _history_with_tool_loop()[-2:]
        prompt = _format_delta_prompt(delta)
        self.assertNotIn("USER REQUEST THIS TOOL LOOP IS SERVING", prompt)
        self.assertIn("Continue the conversation from the latest tool result.", prompt)

    def test_full_tool_prompt_restates_pending_request_at_tail(self):
        prompt = _format_messages_as_prompt(_history_with_tool_loop())
        tail = prompt[prompt.index("### LATEST TOOL RESULTS RECEIVED."):]
        self.assertIn(QUESTION, tail)

    def test_full_tool_prompt_anchors_pending_request_above_transcript(self):
        prompt = _format_messages_as_prompt(_history_with_tool_loop())
        head = prompt[: prompt.index("Conversation transcript:")]
        self.assertIn("### ACTIVE OBJECTIVE:", head)
        self.assertIn(QUESTION, head)
        self.assertNotIn("Hide the menu and deploy.", head)

    def test_full_user_prompt_keeps_single_objective(self):
        history = _history_with_tool_loop()[:4]
        prompt = _format_messages_as_prompt(history)
        self.assertEqual(prompt.count("### ACTIVE OBJECTIVE:"), 1)
        self.assertIn("The user has submitted a new request", prompt)

    def test_user_turn_prompt_is_unchanged(self):
        history = _history_with_tool_loop()[:4]
        prompt = _format_delta_prompt(history[-1:], pending_user_text=QUESTION)
        self.assertNotIn("USER REQUEST THIS TOOL LOOP IS SERVING", prompt)
        self.assertEqual(prompt.count(QUESTION), 1)


# A skill-driven cron request (Pulse news briefing) of ~54 KB.
LONG_REQUEST = "PULSE-HEAD brief the news. " + ("follow the skill step by step. " * 1800) + " PULSE-TAIL"
NUDGE = (
    "You just executed tool calls but returned an empty response. "
    "Please process the tool results above and continue with the task."
)


def _tool_step(i: int) -> list[dict]:
    return [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": f"c{i}", "function": {"name": "web_search", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": f"c{i}", "content": f"RESULT-{i} headline"},
    ]


class TestLongRequestUpdatesStayCompact(unittest.TestCase):
    """Turn-by-turn updates restate a long request as a bounded excerpt.

    Regression for the Pulse cron: every tool-call step re-pasted the whole
    54 KB request into the persistent agy conversation, which then holds the
    request once per step.
    """

    def test_each_tool_step_update_is_compact_and_points_back(self):
        self.assertGreater(len(LONG_REQUEST), 50_000)
        sizes = []
        for i in range(20):
            prompt = _format_delta_prompt(_tool_step(i), pending_user_text=LONG_REQUEST)
            sizes.append(len(prompt))
            self.assertIn(f"RESULT-{i} headline", prompt)
            self.assertIn("USER REQUEST THIS TOOL LOOP IS SERVING", prompt)
            # The excerpt keeps the request's start and end, after the tool output.
            self.assertGreater(prompt.index("PULSE-HEAD"), prompt.index(f"RESULT-{i} headline"))
            self.assertIn("PULSE-TAIL", prompt)
            self.assertIn("appears earlier in this conversation", prompt)
            self.assertNotIn(LONG_REQUEST, prompt)
        # Bounded per step, independent of the request's size.
        self.assertLess(max(sizes), 4_000)

    def test_short_request_is_still_restated_in_full(self):
        request = "q" * PENDING_REQUEST_FULL_MAX_CHARS
        prompt = _format_delta_prompt(_tool_step(0), pending_user_text=request)
        self.assertIn(request, prompt)
        self.assertNotIn("omitted here", prompt)

    def test_recovery_nudge_update_restates_an_excerpt(self):
        delta = [*_tool_step(0), {"role": "user", "content": NUDGE}]
        prompt = _format_delta_prompt(delta, pending_user_text=LONG_REQUEST)
        self.assertIn("LATEST USER REQUEST TO ANSWER", prompt)
        self.assertIn("PULSE-HEAD", prompt)
        self.assertIn("appears earlier in this conversation", prompt)
        self.assertNotIn(LONG_REQUEST, prompt)
        self.assertLess(len(prompt), 4_000)

    def test_full_pending_request_keeps_the_whole_text(self):
        _, closing = _delta_parts(_tool_step(0), LONG_REQUEST, full_pending_request=True)
        self.assertIn(LONG_REQUEST, "\n".join(closing))

    def test_new_long_user_request_is_sent_whole(self):
        # A new request is not yet in the conversation: it is never cut.
        prompt = _format_delta_prompt([{"role": "user", "content": LONG_REQUEST}])
        self.assertIn(LONG_REQUEST, prompt)


if __name__ == "__main__":
    unittest.main()
