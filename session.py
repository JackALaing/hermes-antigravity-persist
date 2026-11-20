"""Drive persistent agy conversations for main-agent turns (see persist.py).

One turn:

1. Take the conversation's cross-process lock (timeout: the caller falls back
   to a one-shot and the record is left alone).
2. Decide resume or refresh from the record and the incoming history.
3. Resume: reuse the live worker, or spawn ``agy --conversation <id>`` and
   check the init event names that conversation (otherwise refresh). Send
   only the messages after the last reply.
4. Refresh: spawn a new conversation in the persistent home and send the
   packed history: as many context-only parts as it needs, each consumed
   here and acknowledged with ACK (they never reach Hermes), then the final
   part carrying the latest request and its active tool loop.
5. Hand Hermes an ``AntigravityStream`` whose "client" is the turn object
   below. On success the stream calls ``_on_worker_turn_success`` before the
   finish chunk: the record is written and the lock released. Any failure,
   interruption or cancel marks the record ``needs_refresh``.

Live workers are kept in a module-level registry keyed by (state root, scope
hash), so a client Hermes recreates for the same conversation reuses the
process. An idle worker is terminated after the usual idle bound; its record
stays valid and the next turn resumes with ``--conversation``.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from typing import Any

try:
    from . import persist
    from .persist import (
        ConversationLock,
        PersistStore,
        ack_failure,
        check_stdin_bytes,
        decide,
        pack_refresh,
        reply_fingerprint,
        scope_key,
        split_delta,
        tools_hash,
    )
    from .process import (
        _own_process_group,
        build_child_env,
        read_media_rules,
        register_tracked_process,
        start_stderr_drainer,
        sync_isolated_token,
        terminate_process,
        write_media_read_permissions,
    )
    from .models import split_effort_suffix
    from .prompt import _latest_user_text
    from .stream import AntigravityStream, _StepUsage, _delta_worker_usage
except ImportError:
    import persist
    from persist import (
        ConversationLock,
        PersistStore,
        ack_failure,
        check_stdin_bytes,
        decide,
        pack_refresh,
        reply_fingerprint,
        scope_key,
        split_delta,
        tools_hash,
    )
    from process import (
        _own_process_group,
        build_child_env,
        read_media_rules,
        register_tracked_process,
        start_stderr_drainer,
        sync_isolated_token,
        terminate_process,
        write_media_read_permissions,
    )
    from models import split_effort_suffix
    from prompt import _latest_user_text
    from stream import AntigravityStream, _StepUsage, _delta_worker_usage

logger = logging.getLogger(__name__)


def _base_model(model: str) -> str:
    return split_effort_suffix(model)[0]


class ResumeRejected(Exception):
    """agy did not reopen the recorded conversation; refresh instead."""


class _Worker:
    def __init__(self, proc: subprocess.Popen, model: str, effort: str | None,
                 media_rules: frozenset[str], expected_conversation: str | None):
        self.proc = proc
        self.model = model
        self.effort = effort
        self.media_rules = media_rules
        # Set from the init event; None until the first write is answered.
        self.conversation_id: str | None = None
        self.expected_conversation = expected_conversation
        self.awaiting_init = True
        # result.usage is cumulative per process: the fallback baseline.
        self.usage_baseline: dict[str, int] = {}
        self.idle_timer: threading.Timer | None = None

    def alive(self) -> bool:
        return self.proc.poll() is None


_REGISTRY: dict[tuple[str, str], _Worker] = {}
_REGISTRY_LOCK = threading.Lock()


def _registered(key: tuple[str, str]) -> _Worker | None:
    with _REGISTRY_LOCK:
        return _REGISTRY.get(key)


def _kill_worker(key: tuple[str, str], worker: _Worker | None) -> None:
    if worker is None:
        return
    with _REGISTRY_LOCK:
        if _REGISTRY.get(key) is worker:
            del _REGISTRY[key]
    timer, worker.idle_timer = worker.idle_timer, None
    if timer is not None:
        timer.cancel()
    terminate_process(worker.proc)


def _arm_idle_timer(store: PersistStore, key: tuple[str, str], worker: _Worker,
                    idle_seconds: float) -> None:
    timer = worker.idle_timer
    if timer is not None:
        timer.cancel()
    worker.idle_timer = None
    if idle_seconds <= 0:
        return
    timer = threading.Timer(idle_seconds, _reap_idle, args=(store, key, worker, idle_seconds))
    timer.daemon = True
    worker.idle_timer = timer
    timer.start()


def _reap_idle(store: PersistStore, key: tuple[str, str], worker: _Worker,
               idle_seconds: float) -> None:
    """Terminate an idle worker; a turn holding the lock means it is busy."""
    if _registered(key) is not worker:
        return
    lock = ConversationLock(store.lock_path(key[1]))
    if not lock.acquire(timeout=0):
        _arm_idle_timer(store, key, worker, idle_seconds)
        return
    try:
        if _registered(key) is worker:
            logger.info("Terminating idle persistent agy worker pid=%s", getattr(worker.proc, "pid", None))
            worker.idle_timer = None
            _kill_worker(key, worker)
    finally:
        lock.release()


def _write_user_event(proc: subprocess.Popen, content: str) -> None:
    check_stdin_bytes(content)
    if proc.stdin is None:
        raise RuntimeError("Antigravity execution failed: worker has no stdin")
    proc.stdin.write(json.dumps({"event": "user", "message": {"content": content}}) + "\n")
    proc.stdin.flush()


def _parse(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        event = json.loads(line)
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def _read_line(proc: subprocess.Popen, pending: list[str]) -> str:
    if pending:
        return pending.pop(0)
    return proc.stdout.readline() if proc.stdout else ""


def _await_init(proc: subprocess.Popen, deadline: float) -> tuple[str | None, list[str], str]:
    """Read stdout up to agy's init event, which follows the first write.

    Returns (conversation id or None, lines read, failure detail). The lines
    are replayed to whoever reads the turn next.
    """
    lines: list[str] = []
    while time.monotonic() < deadline:
        line = proc.stdout.readline() if proc.stdout else ""
        if not line:
            if proc.poll() is not None:
                return None, lines, f"process exited with return code {proc.poll()} before init"
            time.sleep(0.01)
            continue
        lines.append(line)
        event = _parse(line)
        if event is None:
            continue
        kind = event.get("event")
        if kind == "init":
            init = event.get("init") or {}
            conv = event.get("conversation_id") or init.get("conversation_id") or ""
            return (conv or None), lines, ("" if conv else "init without conversation id")
        if kind == "result":
            result = event.get("result") or {}
            return None, lines, f"result before init: {result.get('error') or result.get('status') or 'no status'}"
    return None, lines, "no init event before the deadline"


class PersistentTurn:
    """The "client" an AntigravityStream sees for one persistent turn.

    Exposes exactly the attributes AntigravityStream uses on its client and
    routes them to the conversation's worker, record and lock.
    """

    def __init__(self, client: Any, store: PersistStore, key: str, lock: ConversationLock, *,
                 model: str, effort: str | None, tools_hash_value: str,
                 previous: dict[str, Any] | None, idle_seconds: float):
        self._client = client
        self._store = store
        self._key = key
        self._registry_key = (str(store.root), key)
        self._turn_lock = lock
        self._model = model
        self._effort = effort
        self._tools_hash = tools_hash_value
        self._previous = previous
        self._idle_seconds = idle_seconds
        self._lock = threading.RLock()
        self._active_processes: set = set()
        self._isolated_gemini_dir = store.gemini_dir
        self.worker: _Worker | None = None
        self.seed_tokens = 0
        self.committed = False
        self.finished = False

    # -- attributes AntigravityStream reads -------------------------------
    @property
    def _cancelled(self) -> bool:
        return getattr(self._client, "_cancelled", False) is True

    @property
    def _worker_proc(self) -> subprocess.Popen | None:
        worker = self.worker
        return worker.proc if worker is not None else None

    @staticmethod
    def _terminate_process(proc: subprocess.Popen) -> None:
        terminate_process(proc)

    def _terminate_worker(self, keep_conversation: bool = False) -> None:
        self.abort("stream closed before the turn completed")

    def _update_worker_history(self, messages: Any) -> None:
        worker = self.worker
        if worker is not None and _registered(self._registry_key) is worker:
            _arm_idle_timer(self._store, self._registry_key, worker, self._idle_seconds)
        self._finish()

    def _set_worker_conversation_id(self, conversation_id: str) -> None:
        worker = self.worker
        if worker is not None and conversation_id and worker.conversation_id is None:
            worker.conversation_id = conversation_id

    def _note_worker_activity(self, proc: subprocess.Popen) -> None:
        return None

    def _on_worker_turn_success(self, stream: AntigravityStream, step_usage: _StepUsage,
                                usage_data: dict[str, Any]) -> None:
        self.commit(stream.reply_message(), step_usage, usage_data)

    # -- lifecycle ----------------------------------------------------------
    def commit(self, reply: dict[str, Any], step_usage: _StepUsage, usage_data: dict[str, Any]) -> None:
        if self.committed or self.finished:
            return
        worker = self.worker
        try:
            if worker is None or not worker.conversation_id:
                raise RuntimeError("no agy conversation id for this turn")
            if step_usage.seen:
                size = step_usage.context_tokens
            else:  # over-estimates on a reused process: refreshes early, never late
                size = int(usage_data.get("input_tokens") or 0) + int(usage_data.get("cache_read_tokens") or 0)
            previous = self._previous or {}
            same_conversation = previous.get("agy_conversation_id") == worker.conversation_id
            now = time.time()
            record = {
                "version": persist.RECORD_VERSION,
                "scope_hash": self._key,
                "agy_conversation_id": worker.conversation_id,
                "last_reply_fp": reply_fingerprint(reply),
                "model": _base_model(self._model),
                "effort": self._effort,
                "tools_hash": self._tools_hash,
                "agy_size_tokens": size,
                "needs_refresh": False,
                "seed_tokens_total": (int(previous.get("seed_tokens_total") or 0) if same_conversation else 0)
                + self.seed_tokens,
                "created_at": previous.get("created_at", now) if same_conversation else now,
                "updated_at": now,
            }
            self._store.save(record)
            self.committed = True
            if previous.get("agy_conversation_id") and not same_conversation:
                self._store.delete_conversation(previous["agy_conversation_id"])
        except Exception:
            logger.warning("Could not record persistent agy turn; next turn refreshes", exc_info=True)
            self._store.mark_needs_refresh(self._key)
        finally:
            self._turn_lock.release()

    def abort(self, reason: str) -> None:
        """Failure, interruption or cancel: kill the worker; refresh next time.

        A turn that already committed keeps its record: agy finished the turn,
        so the conversation on disk is consistent and resumable.
        """
        if self.finished:
            return
        worker = self.worker
        _kill_worker(self._registry_key, worker)
        if not self.committed:
            logger.info("Persistent agy turn aborted (%s); marking for refresh", reason)
            self._store.mark_needs_refresh(self._key)
            # A conversation this turn started was never recorded: discard it.
            previous_id = (self._previous or {}).get("agy_conversation_id")
            if worker is not None and worker.conversation_id and worker.conversation_id != previous_id:
                self._store.delete_conversation(worker.conversation_id)
        self._finish()

    def _finish(self) -> None:
        self.finished = True
        self._turn_lock.release()
        turns = getattr(self._client, "_persistent_turns", None)
        if turns is not None:
            turns.discard(self)

    # -- spawning and seeding ----------------------------------------------
    def spawn(self, conversation_id: str | None) -> _Worker:
        _kill_worker(self._registry_key, _registered(self._registry_key))
        client = self._client
        sync_isolated_token(self._store.home)
        cmd = [client._command, "--input-format", "stream-json", *client._args]
        if conversation_id:
            cmd.extend(["--conversation", conversation_id])
        if self._model:
            cmd.extend(["--model", self._model])
        if self._effort:
            cmd.extend(["--effort", self._effort])
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=str(self._store.workspace),
            env=build_child_env(self._store.home),
            **_own_process_group(),
        )
        start_stderr_drainer(proc)
        register_tracked_process(proc)
        worker = _Worker(proc, self._model, self._effort,
                         read_media_rules(self._store.gemini_dir), conversation_id)
        with _REGISTRY_LOCK:
            _REGISTRY[self._registry_key] = worker
        self.worker = worker
        # A death date from birth, as for the per-client worker: a turn that
        # never completes must not leave the process running forever.
        _arm_idle_timer(self._store, self._registry_key, worker, self._idle_seconds)
        return worker

    def write(self, payload: str, deadline: float) -> list[str]:
        """Write one stdin message; on a fresh process, validate its init."""
        worker = self.worker
        assert worker is not None
        if self._cancelled:
            raise RuntimeError("Antigravity request cancelled.")
        _write_user_event(worker.proc, payload)
        if not worker.awaiting_init:
            return []
        worker.awaiting_init = False
        conversation_id, lines, failure = _await_init(worker.proc, deadline)
        expected = worker.expected_conversation
        if expected is not None:
            if failure or conversation_id != expected:
                # agy may have opened a different conversation instead.
                if conversation_id and conversation_id != expected:
                    _kill_worker(self._registry_key, worker)
                    self._store.delete_conversation(conversation_id)
                raise ResumeRejected(failure or f"init named conversation {conversation_id!r}")
        elif not conversation_id:
            raise RuntimeError(f"Antigravity execution failed: new conversation did not start ({failure})")
        worker.conversation_id = conversation_id
        return lines

    def consume_seed(self, pending: list[str], deadline: float) -> None:
        """Read a context-only part's reply internally and require an ACK."""
        worker = self.worker
        assert worker is not None
        text, status, error = "", "", ""
        saw_result = native_tool = False
        steps = _StepUsage()
        result_usage: dict[str, Any] = {}
        while time.monotonic() < deadline:
            if self._cancelled:
                raise RuntimeError("Antigravity request cancelled.")
            line = _read_line(worker.proc, pending)
            if not line:
                if worker.proc.poll() is not None:
                    break
                time.sleep(0.01)
                continue
            event = _parse(line)
            if event is None:
                continue
            kind = event.get("event")
            if kind == "step_update":
                step = event.get("step_update") or {}
                if step.get("step_type") == "tool":
                    native_tool = True
                    break
                if isinstance(step.get("usage"), dict):
                    steps.add(step["usage"])
                text += str(step.get("text_delta") or "")
            elif kind == "result":
                result = event.get("result") or {}
                saw_result = True
                status = str(result.get("status") or "")
                error = str(result.get("error") or "")
                if not text:
                    text = str(result.get("response") or "")
                if isinstance(result.get("usage"), dict):
                    result_usage = result["usage"]
                break
        if result_usage:
            _delta_worker_usage(result_usage, worker.usage_baseline)
        tokens = (steps.context_tokens + steps.output_sum) if steps.seen else 0
        self.seed_tokens += tokens
        failure = ack_failure(saw_result=saw_result, status=status, error=error, text=text,
                              native_tool=native_tool)
        if failure:
            raise RuntimeError(f"Antigravity execution failed: context restore was not acknowledged ({failure})")
        logger.info("Antigravity context restore acknowledged (%d tokens)", tokens)


def run_persistent_turn(
    client: Any,
    *,
    scope: str,
    model: str,
    effort: str | None,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    timeout: float,
    media_paths: set[str],
    idle_seconds: float,
) -> AntigravityStream | None:
    """Start one persistent turn; None when the lock timed out (use a one-shot)."""
    store = PersistStore(persist.persist_root())
    store.ensure()
    key = scope_key(scope)
    lock = ConversationLock(store.lock_path(key))
    if not lock.acquire(persist.LOCK_TIMEOUT_SECONDS):
        logger.warning("Persistent agy conversation busy for %.0fs; using a one-shot",
                       persist.LOCK_TIMEOUT_SECONDS)
        return None
    record = store.load(key)
    turn = PersistentTurn(client, store, key, lock, model=model, effort=effort,
                          tools_hash_value=tools_hash(tools), previous=record,
                          idle_seconds=idle_seconds)
    turns = getattr(client, "_persistent_turns", None)
    if turns is not None:
        turns.add(turn)
    try:
        return _start_turn(turn, store, key, record, model=model, effort=effort, messages=messages,
                           tools=tools, tool_choice=tool_choice, timeout=timeout,
                           media_paths=media_paths)
    except BaseException as exc:
        turn.abort(f"{type(exc).__name__}: {exc}")
        raise


def _start_turn(turn: PersistentTurn, store: PersistStore, key: str, record: dict[str, Any] | None, *,
                model: str, effort: str | None, messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None, tool_choice: Any, timeout: float,
                media_paths: set[str]) -> AntigravityStream:
    client = turn._client
    deadline = time.monotonic() + timeout
    media_sink = getattr(client, "_materialized_media", None)
    required_rules = frozenset(f"{tool}({path})" for path in media_paths for tool in ("read_file", "view_file"))
    if required_rules and not required_rules <= read_media_rules(store.gemini_dir):
        write_media_read_permissions(store.gemini_dir, media_paths)

    # The resolved id carries the effort suffix; an effort change alone must
    # resume (respawning with --conversation), so compare the base model.
    decision = decide(record, messages, model=_base_model(model), tools_hash=turn._tools_hash)
    payloads: list[str] = []
    pending: list[str] = []
    if decision.resume:
        assert record is not None
        conversation_id = record["agy_conversation_id"]
        payloads = split_delta(decision.delta, pending_user_text=_latest_user_text(messages),
                               media_sink=media_sink)
        worker = _registered(turn._registry_key)
        reusable = (
            worker is not None and worker.alive() and not worker.awaiting_init
            and worker.conversation_id == conversation_id
            and worker.model == model and worker.effort == effort
            and required_rules <= worker.media_rules
        )
        if reusable:
            turn.worker = worker
        else:
            turn.spawn(conversation_id)
        try:
            pending = turn.write(payloads[0], deadline)
        except ResumeRejected as exc:
            logger.info("agy did not resume conversation %s (%s); refreshing", conversation_id, exc)
            decision = persist.Decision(False, f"resume rejected: {exc}")
        else:
            logger.debug("Resuming persistent agy conversation (%d message(s))", len(decision.delta))
    if not decision.resume:
        logger.info("Refreshing persistent agy conversation: %s", decision.reason)
        store.gc()
        payloads = pack_refresh(messages, model=model, tools=tools, tool_choice=tool_choice,
                                media_sink=media_sink)
        turn.spawn(None)
        pending = turn.write(payloads[0], deadline)

    # Each context-only part gets the full timeout, and so does the real turn
    # after the last one: a long rebuild must not starve the answer.
    for payload in payloads[1:]:
        turn.consume_seed(pending, deadline)
        deadline = time.monotonic() + timeout
        pending = turn.write(payload, deadline)

    worker = turn.worker
    assert worker is not None
    return AntigravityStream(
        proc=worker.proc,
        client=turn,
        model=model,
        timeout=max(deadline - time.monotonic(), 1.0),
        tools=tools,
        is_worker=True,
        worker_lock=turn._turn_lock,
        worker_lock_held=True,
        messages=messages,
        usage_baseline=worker.usage_baseline,
        prompt_chars=len(payloads[-1]),
        delivery_watch=None,
        pending_lines=pending,
    )
