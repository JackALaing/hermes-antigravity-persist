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
import os
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
        request_fingerprint,
        scope_key,
        split_delta,
        stored_reply_fingerprint,
        system_hash,
        tools_hash,
        user_fingerprint,
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
    from .custom_agent import custom_agent_args, note_custom_agent_rejection
    from .models import split_effort_suffix
    from .prompt import _latest_user_text
    from .stream import AntigravityStream, _StepUsage, _delta_worker_usage, cache_hit_pct
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
        request_fingerprint,
        scope_key,
        split_delta,
        stored_reply_fingerprint,
        system_hash,
        tools_hash,
        user_fingerprint,
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
    from custom_agent import custom_agent_args, note_custom_agent_rejection
    from models import split_effort_suffix
    from prompt import _latest_user_text
    from stream import AntigravityStream, _StepUsage, _delta_worker_usage, cache_hit_pct

logger = logging.getLogger(__name__)


def _base_model(model: str) -> str:
    return split_effort_suffix(model)[0]


def _record_time(record: dict[str, Any] | None) -> float | None:
    """The record's last commit time: when its conversation's previous turn ended."""
    try:
        value = float((record or {}).get("updated_at"))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


class ResumeRejected(Exception):
    """agy did not reopen the recorded conversation; refresh instead."""


class SeedRejected(RuntimeError):
    """A context-only part was not accepted; rebuild as one message instead.

    ``native_tool`` marks a rejection because agy ran a native tool on the
    part: the model ignored the context-only instruction, which recurs. Other
    rejections (no result, an agy error, a non-SUCCESS status) are transient
    and never move a conversation to single-message rebuilds.
    """

    def __init__(self, message: str, *, native_tool: bool = False):
        super().__init__(message)
        self.native_tool = native_tool


# After this many context-only parts rejected for running a native tool on one
# Hermes conversation, its refreshes skip seeds and send one packed message
# (oldest history omitted). A rebuild whose seeds are all accepted resets the
# count, and single-message mode lapses SEED_FAILURE_TTL_SECONDS after the
# last counted rejection, so seeds are tried again.
SEED_FAILURE_LIMIT = 2
SEED_FAILURE_TTL_SECONDS = 7 * 24 * 3600


def _single_message_rebuilds(record: dict[str, Any] | None, now: float | None = None) -> bool:
    """Whether this conversation's refreshes skip context-only parts."""
    record = record or {}
    if int(record.get("seed_failures") or 0) < SEED_FAILURE_LIMIT:
        return False
    try:
        last = float(record.get("seed_failure_at") or 0)
    except (TypeError, ValueError):
        last = 0.0
    now = time.time() if now is None else now
    # A record without the timestamp (written before it existed) has lapsed:
    # its last commit time moves on every turn, so it would never expire.
    return now - last < SEED_FAILURE_TTL_SECONDS
# A persistent turn whose stream nobody has read for this long is abandoned:
# its worker is killed and the conversation lock released, so later turns do
# not wait for garbage collection. A live consumer is either blocked inside a
# read (never counted) or handling one chunk, which takes far less.
# Override: ANTIGRAVITY_PERSIST_ABANDON_SECONDS; 0 or less disables.
DEFAULT_ABANDON_SECONDS = 180.0


def _abandon_seconds() -> float:
    raw = os.environ.get("ANTIGRAVITY_PERSIST_ABANDON_SECONDS", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return DEFAULT_ABANDON_SECONDS


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
        # The record's commit token after this worker's last committed turn. A
        # live worker is reused only while the record still carries it: another
        # Hermes process may have advanced the conversation since.
        self.commit_token: str | None = None

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
                 previous: dict[str, Any] | None, idle_seconds: float,
                 messages: list[dict[str, Any]] | None = None):
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
        self._messages = list(messages or [])
        self._system_hash = system_hash(self._messages)
        # The plugin's custom agent (custom_agent.py) for this turn's spawns,
        # set by resolve_agent() before the decision and at every spawn; the
        # record keeps it. AntigravityStream reads _custom_agent_named.
        self.agent_args: list[str] = []
        self.agent = ""
        self._custom_agent_named = False
        self.worker: _Worker | None = None
        self.seed_tokens = 0
        # This turn's seed outcome: a native-tool rejection (counted towards
        # single-message rebuilds) or a seeded rebuild fully accepted (resets
        # the count).
        self.seed_failed = False
        self.seeds_accepted = False
        # Hermes-side request size reported as prompt_tokens (see
        # persist.hermes_request_tokens); kept in the record for diagnostics.
        self.hermes_tokens: int | None = None
        self.committed = False
        self.finished = False
        # Abandon watchdog (see DEFAULT_ABANDON_SECONDS).
        self._pulling = False
        self._last_pull = time.monotonic()
        self._watch_stop = threading.Event()

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

    def _note_consumer_pull(self, active: bool) -> None:
        """AntigravityStream calls this around every read its consumer makes."""
        with self._lock:
            self._pulling = active
            self._last_pull = time.monotonic()

    def start_abandon_watchdog(self) -> None:
        grace = _abandon_seconds()
        if grace <= 0 or self.finished or self.committed:
            return
        with self._lock:
            self._last_pull = time.monotonic()
        thread = threading.Thread(target=self._watch_abandon, args=(grace,), daemon=True,
                                  name="agy-persist-abandon")
        thread.start()

    def _watch_abandon(self, grace: float) -> None:
        interval = min(max(grace / 6.0, 0.02), 30.0)
        while not self._watch_stop.wait(interval):
            if self.finished or self.committed:
                return
            with self._lock:
                idle = 0.0 if self._pulling else time.monotonic() - self._last_pull
            if idle >= grace:
                logger.warning(
                    "Persistent agy turn abandoned: its stream was not read for %.0fs; "
                    "stopping the worker and releasing the conversation lock", idle,
                )
                self.abort("stream abandoned")
                return

    def resolve_agent(self) -> str:
        """Name the custom agent for this turn's spawns ('' for agy's default).

        Same rule as the per-client path (AntigravityClient._agent_args): an
        agent named in ANTIGRAVITY_ARGS wins, and a refused agent is dropped
        for the rest of the process. The definition is written into the
        persistent home, where these spawns run.
        """
        client = self._client
        args: list[str] = []
        if not any(arg == "--agent" or arg.startswith("--agent=") for arg in client._args):
            args = custom_agent_args(client._command, self._store.home)
        self.agent_args = list(args)
        self.agent = args[1] if len(args) > 1 else ""
        self._custom_agent_named = bool(args)
        return self.agent

    def note_agent_rejection(self, detail: object, proc: subprocess.Popen | None = None) -> None:
        """Stop naming the agent if agy refused it (its error or stderr says so)."""
        if not self._custom_agent_named:
            return
        tail = ""
        drainer = getattr(proc, "_stderr_drainer", None) if proc is not None else None
        if drainer is not None and hasattr(drainer, "get_tail"):
            try:
                tail = drainer.get_tail(timeout=1.0) or ""
            except Exception:
                tail = ""
        note_custom_agent_rejection(f"{detail}\n{tail}" if tail else detail)

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
            token = os.urandom(8).hex()
            if self.seed_failed:
                seed_failures, seed_failure_at = int(previous.get("seed_failures") or 0) + 1, now
            elif self.seeds_accepted:
                seed_failures, seed_failure_at = 0, None
            else:
                seed_failures = int(previous.get("seed_failures") or 0)
                seed_failure_at = previous.get("seed_failure_at")
            record = {
                "version": persist.RECORD_VERSION,
                "scope_hash": self._key,
                "agy_conversation_id": worker.conversation_id,
                "last_reply_fp": reply_fingerprint(reply),
                "last_reply_stored_fp": stored_reply_fingerprint(reply),
                "last_user_fp": user_fingerprint(self._messages, len(self._messages)),
                "last_request_fp": request_fingerprint(self._messages),
                "system_hash": self._system_hash,
                "commit_token": token,
                "seed_failures": seed_failures,
                "seed_failure_at": seed_failure_at,
                "model": _base_model(self._model),
                "effort": self._effort,
                "tools_hash": self._tools_hash,
                "agent": self.agent,
                "agy_size_tokens": size,
                "hermes_size_tokens": self.hermes_tokens,
                "needs_refresh": False,
                "seed_tokens_total": (int(previous.get("seed_tokens_total") or 0) if same_conversation else 0)
                + self.seed_tokens,
                "created_at": previous.get("created_at", now) if same_conversation else now,
                "updated_at": now,
            }
            self._store.save(record)
            worker.commit_token = token
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
        if self.committed:
            # agy finished the turn and the record names it: a close or cancel
            # after the finish chunk must not cost the next turn a respawn.
            if worker is not None and _registered(self._registry_key) is worker:
                _arm_idle_timer(self._store, self._registry_key, worker, self._idle_seconds)
            self._finish()
            return
        _kill_worker(self._registry_key, worker)
        if not self.committed:
            logger.info("Persistent agy turn aborted (%s); marking for refresh", reason)
            self._store.mark_needs_refresh(self._key)
            # A conversation this turn started was never recorded: discard it.
            previous_id = (self._previous or {}).get("agy_conversation_id")
            if worker is not None and worker.conversation_id and worker.conversation_id != previous_id:
                self._store.delete_conversation(worker.conversation_id)
        self._finish()

    def discard_worker(self) -> None:
        """Kill this turn's worker and drop the conversation it started (not recorded)."""
        worker, self.worker = self.worker, None
        self.seed_tokens = 0  # spent on a conversation that no longer exists
        _kill_worker(self._registry_key, worker)
        previous_id = (self._previous or {}).get("agy_conversation_id")
        if worker is not None and worker.conversation_id and worker.conversation_id != previous_id:
            self._store.delete_conversation(worker.conversation_id)

    def _finish(self) -> None:
        self.finished = True
        self._watch_stop.set()
        self._turn_lock.release()
        turns = getattr(self._client, "_persistent_turns", None)
        if turns is not None:
            turns.discard(self)

    # -- spawning and seeding ----------------------------------------------
    def spawn(self, conversation_id: str | None) -> _Worker:
        _kill_worker(self._registry_key, _registered(self._registry_key))
        client = self._client
        sync_isolated_token(self._store.home)
        # Asked at every spawn, as on the per-client path: an agent agy
        # refused earlier in this turn is no longer named on the retry.
        self.resolve_agent()
        cmd = [client._command, "--input-format", "stream-json", *client._args, *self.agent_args]
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
        if failure:
            # A spawn agy refuses for its agent exits with stderr only.
            self.note_agent_rejection(failure, worker.proc)
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
        status, error = "", ""
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
            elif kind == "result":
                result = event.get("result") or {}
                saw_result = True
                status = str(result.get("status") or "")
                error = str(result.get("error") or "")
                if isinstance(result.get("usage"), dict):
                    result_usage = result["usage"]
                break
        if result_usage:
            _delta_worker_usage(result_usage, worker.usage_baseline)
        tokens = (steps.context_tokens + steps.output_sum) if steps.seen else 0
        self.seed_tokens += tokens
        failure = ack_failure(saw_result=saw_result, status=status, error=error, native_tool=native_tool)
        if failure and not native_tool:
            self.note_agent_rejection(error or failure, worker.proc)
        if failure:
            raise SeedRejected(f"Antigravity execution failed: context restore was not acknowledged ({failure})",
                               native_tool=native_tool)
        logger.info("Antigravity context restore acknowledged (%d tokens; input=%d cache_read=%d hit=%.1f%%)",
                    tokens, steps.input_sum, steps.cache_read_sum,
                    cache_hit_pct(steps.input_sum, steps.cache_read_sum))


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
    """Start one persistent turn; None when the conversation is unavailable
    (lock timeout, unusable state directory): the caller uses a one-shot."""
    store = PersistStore(persist.persist_root())
    key = scope_key(scope)
    lock = ConversationLock(store.lock_path(key))
    try:
        store.ensure()
        acquired = lock.acquire(persist.LOCK_TIMEOUT_SECONDS)
    except Exception:
        logger.warning("Persistent agy state unavailable; using a one-shot", exc_info=True)
        return None
    if not acquired:
        logger.warning("Persistent agy conversation busy for %.0fs; using a one-shot",
                       persist.LOCK_TIMEOUT_SECONDS)
        return None
    record = store.load(key)
    turn = PersistentTurn(client, store, key, lock, model=model, effort=effort,
                          tools_hash_value=tools_hash(tools), previous=record,
                          idle_seconds=idle_seconds, messages=messages)
    turns = getattr(client, "_persistent_turns", None)
    if turns is not None:
        turns.add(turn)
    try:
        return _start_turn(turn, store, key, record, model=model, effort=effort, messages=messages,
                           tools=tools, tool_choice=tool_choice, timeout=timeout,
                           media_paths=media_paths)
    except BaseException as exc:
        proc = turn._worker_proc
        turn.abort(f"{type(exc).__name__}: {exc}")
        if proc is not None and isinstance(exc, Exception):
            try:  # the caller's auth-contention check reads the worker's stderr
                exc.agy_proc = proc  # type: ignore[attr-defined]
            except Exception:
                pass
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
    agent = turn.resolve_agent()
    decision = decide(record, messages, model=_base_model(model), tools_hash=turn._tools_hash,
                      system_hash=turn._system_hash, agent=agent)
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
            and bool(record.get("commit_token")) and worker.commit_token == record.get("commit_token")
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
            # The failed handshake may have spent the whole budget.
            deadline = time.monotonic() + timeout
        else:
            logger.debug("Resuming persistent agy conversation (%d message(s))", len(decision.delta))
    if not decision.resume:
        logger.info("Refreshing persistent agy conversation: %s", decision.reason)
        try:  # housekeeping must never fail the user's turn
            store.gc()
        except Exception:
            logger.warning("Antigravity persistent-state cleanup failed", exc_info=True)
        single = _single_message_rebuilds(record)
        payloads = pack_refresh(messages, model=model, tools=tools, tool_choice=tool_choice,
                                media_sink=media_sink, max_messages=1 if single else None)
        turn.spawn(None)
        pending = turn.write(payloads[0], deadline)

    # Each context-only part gets the full timeout, and so does the real turn
    # after the last one: a long rebuild must not starve the answer.
    try:
        for payload in payloads[1:]:
            turn.consume_seed(pending, deadline)
            deadline = time.monotonic() + timeout
            pending = turn.write(payload, deadline)
        turn.seeds_accepted = not decision.resume and len(payloads) > 1
    except SeedRejected as exc:
        # Never fail the conversation over a context-only part: start again
        # from one packed message (the newest history that fits, with the
        # prefix and any compaction summary).
        logger.warning("%s; rebuilding as a single message", exc)
        turn.discard_worker()
        turn.seed_failed = exc.native_tool
        decision = persist.Decision(False, "context restore rejected")
        payloads = pack_refresh(messages, model=model, tools=tools, tool_choice=tool_choice,
                                media_sink=media_sink, max_messages=1)
        deadline = time.monotonic() + timeout
        turn.spawn(None)
        pending = turn.write(payloads[0], deadline)

    try:
        turn.hermes_tokens = persist.hermes_request_tokens(messages, tools, tool_choice)
    except Exception:  # the stream then reports agy's own size
        logger.warning("Could not estimate the Hermes request size; reporting agy's own "
                       "conversation size to Hermes for this turn", exc_info=True)
    worker = turn.worker
    assert worker is not None
    stream = AntigravityStream(
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
        turn_kind="resume" if decision.resume else "refresh",
        conversation_key=f"persist:{key}",
        previous_turn_end=_record_time(record),
        reported_prompt_tokens=turn.hermes_tokens,
    )
    turn.start_abandon_watchdog()
    return stream
