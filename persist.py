"""Persistent agy conversations: records, locking, fingerprints, packing.

A main-agent Hermes conversation keeps one agy conversation alive across
turns and gateway restarts instead of re-sending its whole history. This
module holds the pure and storage parts of that design; ``session.py`` drives
the agy processes.

State lives in one directory per Hermes profile (``persist_root``):

* ``home/``      the persistent isolated agy HOME (token and keychain links,
                 ``conversations/*.db``). Never purged wholesale.
* ``workspace/`` the agy working directory.
* ``records/``   one JSON record per Hermes conversation, named by the hash of
                 its prompt-cache scope, written atomically with mode 0600.
* ``locks/``     one ``flock`` file per conversation.

Every mismatch between a record and the incoming history resolves to a
*refresh*: a new agy conversation seeded from the history. The worst case is a
slower turn, never a turn answered from the wrong state.

``/new`` starts a new Hermes scope, so it gets a new record and a new agy
conversation. The old conversation's database is left in place and removed by
the 14-day record garbage collection (``PersistStore.gc``).
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

try:
    from .prompt import (
        _PROMPT_PREAMBLE,
        _apply_media_viewing_protocol,
        _delta_parts,
        _format_delta_prompt,
        _format_messages_as_prompt,
        _is_recovery_nudge,
        _latest_user_text,
        _render_transcript_entry,
        _system_section,
        _tool_sections,
    )
except ImportError:
    from prompt import (
        _PROMPT_PREAMBLE,
        _apply_media_viewing_protocol,
        _delta_parts,
        _format_delta_prompt,
        _format_messages_as_prompt,
        _is_recovery_nudge,
        _latest_user_text,
        _render_transcript_entry,
        _system_section,
        _tool_sections,
    )

logger = logging.getLogger(__name__)

RECORD_VERSION = 1
# Only the main agent's own turns persist. Background review, /btw side
# questions and the curator run under other write origins; auxiliary calls
# and delegated agents carry no (or their own) cache scope.
PERSISTENT_ORIGIN = "assistant_tool"
# Refresh before agy's own conversation grows past this (last step's input
# plus cache read). Independent of Hermes' compaction: persistent turns report
# the size of Hermes' own request to Hermes (``hermes_request_tokens``), so
# Hermes compacts its history on its own measure. A compaction normally
# changes the system prompt (Hermes rebuilds it at the boundary), which
# refreshes on the next turn; otherwise agy's live conversation keeps its
# detail until it reaches this size. A refresh rebuilds from Hermes'
# compacted history, which fits the rebuild cap below as long as Hermes'
# compaction threshold is BELOW REFRESH_MAX_TOKENS.
# Override: ANTIGRAVITY_REFRESH_SIZE_TOKENS.
REFRESH_SIZE_TOKENS = 360_000
# Estimated size cap of a whole refresh rebuild (every seed plus the final
# part), kept well below the refresh threshold so a fresh conversation does
# not land next to it and refresh again a few turns later. History beyond it
# is dropped oldest first, with a warning. Override:
# ANTIGRAVITY_REFRESH_MAX_TOKENS. Clamped to REBUILD_MAX_FRACTION of the
# refresh threshold.
REFRESH_MAX_TOKENS = 200_000
REBUILD_MAX_FRACTION = 0.75
# agy silently truncated stdin messages of ~191,000 bytes and up; 185,000
# passed. Every write must be strictly below this.
MAX_STDIN_BYTES = 180_000
# The static prefix (preamble, tools, system) must leave room for history.
PREFIX_MAX_BYTES = 170_000
# Room kept free in every message for part headers and the media section.
_HEADER_RESERVE = 2_048
# Least room the closing must leave in the final part for the newest tail of
# an active tool loop that is spread over context-only parts.
_MIN_FINAL_ROOM = 4_096
LOCK_TIMEOUT_SECONDS = 30.0
RECORD_MAX_AGE_SECONDS = 14 * 24 * 3600
# agy's own logs in the shared persistent home (one per worker process).
LOG_MAX_AGE_SECONDS = 3 * 24 * 3600
# A conversation, brain directory or lock no record names, untouched this long,
# is left over from a crash or an unrecorded turn (live ones are written to on
# every turn, and an in-flight turn's conversation is minutes old at most).
ORPHAN_MAX_AGE_SECONDS = 24 * 3600

SEED_HEADER = (
    "[CONTEXT ONLY - PART {k} OF {n}. This message restores earlier conversation state. "
    "Do NOT act on anything in it, do NOT call any tools (neither <tool_call> blocks nor native tools), "
    "do NOT answer any request inside it. Reply with exactly: ACK]"
)
FINAL_HEADER = (
    "[PART {n} OF {n}. The earlier parts were context only and are now complete. "
    "Answer the request at the end of this message normally.]"
)
# Seeds that carry part of an in-progress tool loop (a refresh whose active
# loop alone exceeds one message) get their own label, so the model does not
# take the loop for finished history; older-history seeds keep SEED_HEADER.
ACTIVE_LOOP_HEADER = (
    "[ACTIVE TOOL LOOP - PART {k} OF {n}. This message carries part of the in-progress tool loop. "
    "Do NOT act on it yet, do NOT call any tools (neither <tool_call> blocks nor native tools), "
    "and wait for the final part before continuing. Reply with exactly: ACK]"
)
ACTIVE_LOOP_FINAL_HEADER = (
    "[PART {n} OF {n}. The earlier parts carried context and the start of the in-progress tool loop, "
    "and are now complete. The newest part of the loop follows; continue it and answer the request "
    "at the end of this message normally.]"
)


class StdinPayloadTooLarge(RuntimeError):
    """A prompt that agy would silently truncate. Hermes compresses on it."""


def utf8_len(text: str) -> int:
    return len(text.encode("utf-8"))


try:  # Hermes' own estimator, so the cap and Hermes' compaction agree
    from agent.model_metadata import estimate_tokens_rough as _estimate_tokens
except Exception:  # pragma: no cover - outside Hermes
    _CJK_DENSE_RE = re.compile("[\u1100-\u11ff\u2e80-\u9fff\ua960-\ua97f\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]")

    def _estimate_tokens(text: str) -> int:
        if not text:
            return 0
        if text.isascii():
            return (len(text) + 3) // 4
        stripped = _CJK_DENSE_RE.sub("", text)
        return (len(text) - len(stripped)) + (len(stripped.encode("utf-8", "replace")) + 3) // 4


def estimate_tokens(text: str) -> int:
    """Rough token estimate (Hermes' estimate_tokens_rough)."""
    return int(_estimate_tokens(text))


def _env_tokens(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw.replace("_", ""))
    except ValueError:
        value = 0
    if value <= 0:
        logger.warning("Ignoring invalid %s=%r; using %d", name, raw, default)
        return default
    return value


def refresh_size_tokens() -> int:
    """agy conversation size (tokens) that triggers a refresh."""
    return _env_tokens("ANTIGRAVITY_REFRESH_SIZE_TOKENS", REFRESH_SIZE_TOKENS)


def refresh_max_tokens() -> int:
    """Estimated-token cap on one refresh rebuild, well below the threshold."""
    cap = _env_tokens("ANTIGRAVITY_REFRESH_MAX_TOKENS", REFRESH_MAX_TOKENS)
    ceiling = int(refresh_size_tokens() * REBUILD_MAX_FRACTION)
    if cap > ceiling:
        logger.warning(
            "Antigravity refresh rebuild cap %d tokens is too close to the %d-token refresh "
            "threshold; using %d", cap, refresh_size_tokens(), ceiling,
        )
        return ceiling
    return cap


def check_stdin_bytes(content: str) -> None:
    """The byte guard every stdin write goes through."""
    size = utf8_len(content)
    if size >= MAX_STDIN_BYTES:
        raise StdinPayloadTooLarge(
            f"Antigravity prompt of {size:,} UTF-8 bytes exceeds agy's stdin limit "
            f"of {MAX_STDIN_BYTES:,} bytes; refusing to send a silently truncated prompt."
        )


# --------------------------------------------------------------------------
# Location, records and locks
# --------------------------------------------------------------------------


def persist_root() -> Path:
    """Per-profile state directory (``$HERMES_HOME/antigravity-persist``)."""
    override = os.environ.get("ANTIGRAVITY_PERSIST_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    try:
        from hermes_constants import get_hermes_home

        hermes_home = Path(get_hermes_home())
    except Exception:
        hermes_home = Path(os.path.expanduser("~")) / ".hermes"
    return hermes_home / "antigravity-persist"


def scope_key(scope: str) -> str:
    return hashlib.sha256(scope.encode("utf-8")).hexdigest()[:32]


def is_persistent_request(scope: Any, origin: Any, model: Any = None) -> bool:
    """Main-agent turns persist; Claude routes keep the per-client path.

    agy checkpoints Claude conversations at about 239,616 tokens, below the
    360,000-token refresh threshold, so a persistent Claude conversation would
    be compacted by agy before the plugin refreshed it.
    """
    if isinstance(model, str) and "claude" in model.lower():
        return False
    return bool(isinstance(scope, str) and scope.strip()) and origin == PERSISTENT_ORIGIN


class PersistStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.home = self.root / "home"
        self.gemini_dir = self.home / ".gemini" / "antigravity-cli"
        self.workspace = self.root / "workspace"
        self.records_dir = self.root / "records"
        self.locks_dir = self.root / "locks"

    def ensure(self) -> None:
        """Create the layout and the token/keychain links (idempotent)."""
        try:
            from .process import setup_isolated_home
        except ImportError:
            from process import setup_isolated_home
        for directory in (self.root, self.workspace, self.records_dir, self.locks_dir):
            directory.mkdir(parents=True, exist_ok=True)
            with contextlib.suppress(OSError):
                os.chmod(directory, 0o700)
        setup_isolated_home(self.root)

    def record_path(self, key: str) -> Path:
        return self.records_dir / f"{key}.json"

    def lock_path(self, key: str) -> Path:
        return self.locks_dir / f"{key}.lock"

    def load(self, key: str) -> dict[str, Any] | None:
        path = self.record_path(key)
        try:
            with open(path, encoding="utf-8") as handle:
                record = json.load(handle)
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            logger.warning("Unreadable Antigravity conversation record %s; refreshing", path.name)
            return None
        return record if isinstance(record, dict) else None

    def save(self, record: dict[str, Any]) -> None:
        """Atomic write: temp file, fsync, replace, then fsync the directory."""
        key = record["scope_hash"]
        self.records_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=self.records_dir)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(record, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.record_path(key))
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        with contextlib.suppress(OSError):
            dir_fd = os.open(self.records_dir, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)

    def mark_needs_refresh(self, key: str) -> None:
        record = self.load(key)
        if record is None or record.get("needs_refresh"):
            return
        record["needs_refresh"] = True
        record["updated_at"] = time.time()
        try:
            self.save(record)
        except OSError:
            logger.warning("Could not mark Antigravity record for refresh", exc_info=True)

    def delete_conversation(self, conversation_id: str | None) -> None:
        """Delete one agy conversation and its per-conversation artefacts.

        agy keeps the conversation database under ``conversations/``, a
        transcript and scratch directory under ``brain/<id>`` and annotations
        under ``annotations/``. Symlinks are never followed.
        """
        if not conversation_id or not re.fullmatch(r"[A-Za-z0-9_-]+", conversation_id):
            return
        for sub, pattern in (("conversations", f"{conversation_id}.db*"),
                             ("annotations", f"{conversation_id}.*")):
            for entry in (self.gemini_dir / sub).glob(pattern):
                with contextlib.suppress(OSError):
                    if entry.is_file() and not entry.is_symlink():
                        entry.unlink()
        brain = self.gemini_dir / "brain" / conversation_id
        if brain.is_dir() and not brain.is_symlink():
            shutil.rmtree(brain, ignore_errors=True)

    def gc(self, now: float | None = None) -> int:
        """Housekeeping for the shared persistent home; returns records removed.

        Drops records idle for over 14 days with their conversations, agy logs
        over 3 days old, implicit-context and crash files over 14 days old, and
        conversations, brain directories and locks that no record names and
        nobody has touched for a day.
        """
        now = time.time() if now is None else now
        removed = 0
        live_ids: set[str] = set()
        live_keys: set[str] = set()
        if self.records_dir.is_dir():
            for path in self.records_dir.iterdir():
                age = _age(path, now)
                if age is None:
                    continue
                if path.name.startswith(".tmp-"):
                    if age > 3600:
                        with contextlib.suppress(OSError):
                            path.unlink()
                    continue
                if path.suffix != ".json":
                    continue
                record = self.load(path.stem) or {}
                if age <= RECORD_MAX_AGE_SECONDS or not self._expire_record(path, now):
                    live_keys.add(path.stem)
                    if record.get("agy_conversation_id"):
                        live_ids.add(str(record["agy_conversation_id"]))
                    continue
                removed += 1
        self._gc_orphans(now, live_ids, live_keys)
        return removed

    def _expire_record(self, path: Path, now: float) -> bool:
        """Delete an idle record and its conversation under its lock.

        False (kept) when a turn holds the lock or the record was written
        again meanwhile: that turn is resuming or replacing it right now.
        """
        lock = ConversationLock(self.lock_path(path.stem))
        try:
            if not lock.acquire(0):
                return False
        except OSError:
            return False
        try:
            age = _age(path, now)
            if age is None or age <= RECORD_MAX_AGE_SECONDS:
                return False
            record = self.load(path.stem) or {}
            self.delete_conversation(record.get("agy_conversation_id"))
            path.unlink()
            return True
        except OSError:
            return False
        finally:
            lock.release()

    def _gc_orphans(self, now: float, live_ids: set[str], live_keys: set[str]) -> None:
        def expire(directory: Path, pattern: str, max_age: float) -> None:
            for entry in directory.glob(pattern):
                age = _age(entry, now)
                if age is not None and age > max_age and entry.is_file() and not entry.is_symlink():
                    with contextlib.suppress(OSError):
                        entry.unlink()

        expire(self.gemini_dir / "log", "cli-*.log", LOG_MAX_AGE_SECONDS)
        expire(self.gemini_dir / "implicit", "*.pb", RECORD_MAX_AGE_SECONDS)
        expire(self.gemini_dir / "crashes", "*", RECORD_MAX_AGE_SECONDS)

        orphans: set[str] = set()
        for entry in (self.gemini_dir / "conversations").glob("*.db"):
            if entry.stem not in live_ids and (_age(entry, now) or 0) > ORPHAN_MAX_AGE_SECONDS:
                orphans.add(entry.stem)
        brain_dir = self.gemini_dir / "brain"
        if brain_dir.is_dir():
            for entry in brain_dir.iterdir():
                if (entry.is_dir() and not entry.is_symlink() and entry.name not in live_ids
                        and re.fullmatch(r"[0-9a-f-]{36}", entry.name)
                        and (_tree_age(entry, now) or 0) > ORPHAN_MAX_AGE_SECONDS):
                    orphans.add(entry.name)
        for conversation_id in orphans:
            logger.info("Removing orphaned Antigravity conversation %s", conversation_id[:8])
            self.delete_conversation(conversation_id)

        if self.locks_dir.is_dir():
            for entry in self.locks_dir.glob("*.lock"):
                if entry.stem in live_keys or (_age(entry, now) or 0) <= ORPHAN_MAX_AGE_SECONDS:
                    continue
                lock = ConversationLock(entry)
                if lock.acquire(0):  # never unlink a lock someone holds
                    try:
                        with contextlib.suppress(OSError):
                            entry.unlink()
                    finally:
                        lock.release()


class ConversationLock:
    """Cross-process exclusive lock on one conversation (``flock``).

    flock locks belong to the open file description, so two acquisitions in
    the same process through separate instances also exclude each other.
    ``release`` is idempotent, which lets a stream release it exactly once
    from whichever cleanup path runs first.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._fd: int | None = None

    def acquire(self, timeout: float = LOCK_TIMEOUT_SECONDS) -> bool:
        if self._fd is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        deadline = time.monotonic() + max(timeout, 0.0)
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    return False
                time.sleep(0.05)
                continue
            except OSError:
                os.close(fd)
                raise
            if self._still_linked(fd):
                self._fd = fd
                return True
            # The file was unlinked (orphan-lock cleanup) while we waited on
            # it: a lock on it excludes nobody. Lock the current file instead.
            os.close(fd)
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)

    def _still_linked(self, fd: int) -> bool:
        try:
            held, current = os.fstat(fd), os.stat(self.path)
        except OSError:
            return False
        return (held.st_dev, held.st_ino) == (current.st_dev, current.st_ino)

    def locked(self) -> bool:
        return self._fd is not None

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        with contextlib.suppress(OSError):
            os.close(fd)

    def __del__(self) -> None:  # an abandoned stream must not hold it forever
        with contextlib.suppress(Exception):
            self.release()


def _age(path: Path, now: float) -> float | None:
    try:
        return now - path.stat().st_mtime
    except OSError:
        return None


def _tree_age(directory: Path, now: float) -> float | None:
    """Age of the newest entry in a directory tree (agy writes into brain/<id>)."""
    newest = None
    try:
        for root, dirs, files in os.walk(directory):
            for name in [*dirs, *files, "."]:
                with contextlib.suppress(OSError):
                    mtime = os.lstat(os.path.join(root, name)).st_mtime
                    newest = mtime if newest is None else max(newest, mtime)
    except OSError:
        return None
    return None if newest is None else now - newest


# --------------------------------------------------------------------------
# Fingerprints
# --------------------------------------------------------------------------

_REASONING_BLOCK_RE = re.compile(
    r"<(think|thinking|reasoning|thought)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL
)
_TOOL_CALL_FRAME_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)


def _flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return content["text"]
    return str(content)


def visible_text(content: Any) -> str:
    """Reply text as compaction keeps it: no reasoning, no call frames, one-spaced."""
    text = _REASONING_BLOCK_RE.sub(" ", _flatten_content(content))
    text = _TOOL_CALL_FRAME_RE.sub(" ", text)
    return " ".join(text.split())


def _canonical_arguments(arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError:
            return arguments.strip()
    return json.dumps(arguments, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _hermes_dedupe_key(name: str, arguments: Any) -> tuple[str, str]:
    """The key Hermes drops repeated calls by (``AIAgent._deduplicate_tool_calls``).

    Valid JSON arguments are canonicalised; anything else is compared as is.
    """
    try:
        arguments = json.dumps(json.loads(arguments), separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError):
        pass
    return name, arguments if isinstance(arguments, str) else repr(arguments)


def reply_fingerprint(message: dict[str, Any]) -> str:
    """sha256 of the visible text plus ordered (name, canonical args) calls.

    Tool-call ids are excluded on purpose: Hermes renames reused ids
    (``<id>_d<n>``) and the model's own ids are arbitrary. Repeated calls
    (same name and arguments) count once, first one kept: Hermes drops them
    before it stores the reply, and agy repeats a call when a turn runs a
    second model step.
    """
    calls = []
    seen: set[tuple[str, str]] = set()
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        name = str(function.get("name") or "")
        arguments = function.get("arguments", {})
        key = _hermes_dedupe_key(name, arguments)
        if key in seen:
            continue
        seen.add(key)
        calls.append([name, _canonical_arguments(arguments)])
    payload = json.dumps(
        {"text": visible_text(message.get("content")), "calls": calls},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Hermes' own helpers for the reply as it stores it, imported on first use.
# The plugin runs inside Hermes, so these are the running Hermes' functions.
_HERMES_HELPER_SOURCES = (
    ("strip_think_blocks", "agent.agent_runtime_helpers", "strip_think_blocks"),
    ("redact", "agent.redact", "redact_sensitive_text"),
    ("expand_batches", "agent.tool_call_batches", "expand_local_tool_batches"),
    ("ToolCall", "agent.transports.types", "ToolCall"),
    ("max_children", "tools.delegate_tool", "_get_max_concurrent_children"),
)
_hermes_helpers_cache: Optional[dict[str, Any]] = None
_hermes_helpers_lock = threading.Lock()
_hermes_call_failures: set[str] = set()


def _hermes_helpers() -> dict[str, Any]:
    """Hermes' storage helpers that import, by key; one WARNING lists any that do not.

    A missing helper's step is skipped, so the stored fingerprint stays nearer
    the streamed reply: a turn Hermes rewrote that way refreshes, never resumes
    wrongly.
    """
    global _hermes_helpers_cache
    with _hermes_helpers_lock:
        if _hermes_helpers_cache is None:
            helpers, missing = {}, []
            for key, module, name in _HERMES_HELPER_SOURCES:
                try:
                    helpers[key] = getattr(importlib.import_module(module), name)
                except Exception as exc:
                    missing.append(f"{module}.{name} ({type(exc).__name__}: {exc})")
            if missing:
                logger.warning(
                    "Hermes helpers unavailable, so replies Hermes rewrites that way will refresh "
                    "the persistent agy conversation: %s", "; ".join(missing))
            _hermes_helpers_cache = helpers
        return _hermes_helpers_cache


def _hermes_call(key: str, *args: Any, **kwargs: Any) -> Any:
    """Call Hermes helper ``key``; raise LookupError when it is missing or fails.

    A failure (Hermes changed the helper) logs one WARNING per helper.
    """
    helper = _hermes_helpers().get(key)
    if helper is None:
        raise LookupError(key)
    try:
        return helper(*args, **kwargs)
    except Exception as exc:
        if key not in _hermes_call_failures:
            _hermes_call_failures.add(key)
            logger.warning("Hermes helper %s failed (%s: %s); skipping that step of the stored-reply "
                           "fingerprint", key, type(exc).__name__, exc)
        raise LookupError(key) from exc


def hermes_stored_content(content: Any) -> str:
    """Reply text as Hermes stores it, using Hermes' own helpers.

    Hermes strips reasoning blocks and redacts credentials before the reply
    enters its history (``_assistant_content_for_storage``), so the next
    request can carry, say, ``ROMA_API_KEY=***`` where agy wrote
    ``ROMA_API_KEY=...``. Runs in Hermes' process with its redaction policy;
    a helper that is missing or fails leaves the text as it is.
    """
    text = _flatten_content(content)
    with contextlib.suppress(LookupError):
        text = _hermes_call("strip_think_blocks", None, text).strip()
    if text:
        with contextlib.suppress(LookupError):
            text = _hermes_call("redact", text)
    return text


def _validated_arguments(arguments: Any) -> str:
    """Arguments after Hermes' tool-call validation (``validate_tool_calls``).

    Objects become JSON, other non-strings ``str()``, and an empty or blank
    string (or none) ``"{}"``. Repeated here: Hermes does this inside
    ``validate_tool_calls``, which needs a live agent.
    """
    if isinstance(arguments, (dict, list)):
        return json.dumps(arguments)
    if arguments is not None and not isinstance(arguments, str):
        arguments = str(arguments)
    if not arguments or not arguments.strip():
        return "{}"
    return arguments


def _expanded_batches(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hermes' ``expand_local_tool_batches``: one ``tool_call`` batch of local
    tools becomes one call per entry."""
    tool_call = _hermes_helpers().get("ToolCall")
    if tool_call is None:
        return calls
    try:
        objects = [tool_call(id=call.get("id"), name=call["function"]["name"],
                             arguments=call["function"]["arguments"]) for call in calls]
        expanded = _hermes_call("expand_batches", objects, provider_data=None)
        return [{"id": c.id, "type": "function",
                 "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in expanded]
    except (LookupError, TypeError, AttributeError):
        return calls


def _capped_delegate_calls(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Hermes' ``AIAgent._cap_delegate_task_calls``: delegate_task calls past
    ``max_concurrent_children`` are dropped, others kept.

    Its limit is Hermes' own (``_get_max_concurrent_children``); the trimming is
    repeated here because Hermes' method logs a warning each time it trims.
    """
    count = sum(1 for call in calls if call["function"]["name"] == "delegate_task")
    if not count:
        return calls
    try:
        limit = int(_hermes_call("max_children"))
    except (LookupError, TypeError, ValueError):
        return calls
    if count <= limit:
        return calls
    kept, capped = 0, []
    for call in calls:
        if call["function"]["name"] == "delegate_task":
            if kept >= limit:
                continue
            kept += 1
        capped.append(call)
    return capped


def hermes_stored_tool_calls(tool_calls: Sequence[Any] | None) -> list[dict[str, Any]]:
    """Tool calls as Hermes stores them, in the order its turn applies each step
    (``agent/turn_tool_round.py``): validation's argument normalisation, then
    batch splitting, then the delegate_task cap. Its last step, dropping
    repeated calls, is reply_fingerprint's own.
    """
    calls = []
    for call in tool_calls or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        calls.append({"id": call.get("id"), "type": "function",
                      "function": {"name": str(function.get("name") or ""),
                                   "arguments": _validated_arguments(function.get("arguments"))}})
    return _capped_delegate_calls(_expanded_batches(calls))


def stored_reply_fingerprint(message: dict[str, Any]) -> str:
    """Fingerprint of ``message`` as Hermes stores it (hermes_stored_content,
    hermes_stored_tool_calls)."""
    stored = dict(message, content=hermes_stored_content(message.get("content")))
    if message.get("tool_calls"):
        stored["tool_calls"] = hermes_stored_tool_calls(message["tool_calls"])
    return reply_fingerprint(stored)


def tools_hash(tools: Sequence[Any] | None) -> str:
    payload = json.dumps(list(tools or []), sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Hermes writes the active model and provider into the system prompt; a
# fallback switch or an effort change must not count as an instruction change
# (the model is compared on its own).
_VOLATILE_SYSTEM_LINE_RE = re.compile(r"^[ \t]*(?:Model|Provider)[ \t]*:.*$", re.MULTILINE)


def system_hash(messages: Sequence[Any]) -> str:
    """sha256 of the system messages without their Model:/Provider: lines."""
    parts = [_flatten_content(m.get("content")) for m in messages if _role(m) == "system"]
    text = _VOLATILE_SYSTEM_LINE_RE.sub("", "\n\n".join(parts))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def refresh_on_system_change() -> bool:
    """Whether a changed system prompt forces a refresh (default on)."""
    raw = os.environ.get("ANTIGRAVITY_REFRESH_ON_SYSTEM_CHANGE", "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def user_fingerprint(messages: Sequence[Any], before: int) -> str:
    """sha256 of the latest real user message before index ``before`` ('' if none).

    Paired with the reply fingerprint, it tells an undo back to an identical
    earlier reply ("Done.") from the turn agy actually saw. Hermes replays the
    exact API content of earlier user turns (memory and plugin context
    included), so the text is stable from one turn to the next.
    """
    for index in range(min(before, len(messages)) - 1, -1, -1):
        message = messages[index]
        if _role(message) == "user" and not _is_recovery_nudge(message):
            return hashlib.sha256(visible_text(message.get("content")).encode("utf-8")).hexdigest()
    return ""


def request_fingerprint(messages: Sequence[Any]) -> str:
    """sha256 of a request after its system prompt: how many messages it holds,
    its last assistant reply (reply_fingerprint) and every message after that
    reply, by role, call id and text.

    Recorded with each turn. Hermes sends the same request again when it does
    not keep agy's reply: it retries a tool call whose arguments it cannot
    parse or repair without storing the reply (``agent/turn_truncation.py``).
    agy's conversation then holds a reply Hermes never kept, so that request
    must refresh. A next request that kept the reply holds more messages
    (that reply and what follows it), so it never fingerprints the same.
    """
    index = last_assistant_index(messages)
    anchor = reply_fingerprint(messages[index]) if index is not None else ""
    start = index + 1 if index is not None else 0
    tail = [[_role(m), str(m.get("tool_call_id") or ""), visible_text(m.get("content"))]
            for m in messages[start:] if isinstance(m, dict) and _role(m) != "system"]
    count = sum(1 for m in messages if isinstance(m, dict) and _role(m) != "system")
    payload = json.dumps({"count": count, "anchor": anchor, "tail": tail}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _role(message: Any) -> str:
    return str(message.get("role") or "").strip().lower() if isinstance(message, dict) else ""


def last_assistant_index(messages: Sequence[Any]) -> int | None:
    for index in range(len(messages) - 1, -1, -1):
        if _role(messages[index]) == "assistant":
            return index
    return None


# --------------------------------------------------------------------------
# Decision rules
# --------------------------------------------------------------------------


@dataclass
class Decision:
    resume: bool
    reason: str
    delta: list[dict[str, Any]] = field(default_factory=list)


def decide(
    record: dict[str, Any] | None,
    messages: Sequence[dict[str, Any]],
    *,
    model: str,
    tools_hash: str,
    system_hash: str | None = None,
    agent: str | None = None,
) -> Decision:
    """Resume the recorded agy conversation, or refresh, and why.

    A resume is only safe when agy's conversation ends exactly where the
    incoming history's LAST assistant message ends: any assistant message
    after ours (a fallback provider's answer, a fork's reply, a placeholder
    row) is something agy never saw. An effort change alone still resumes;
    the caller respawns the worker with ``--conversation``. The user request
    that reply answered must match too (an undo back to an identical earlier
    reply), and so must the system prompt (instructions deployed while the
    conversation lives) unless ``ANTIGRAVITY_REFRESH_ON_SYSTEM_CHANGE=0``.
    The very request agy last answered, sent again, refreshes: Hermes did not
    keep that reply (see request_fingerprint).
    """
    if record is None:
        return Decision(False, "no record")
    if record.get("version") != RECORD_VERSION:
        return Decision(False, "record version")
    if record.get("needs_refresh"):
        return Decision(False, "needs refresh")
    if not record.get("agy_conversation_id"):
        return Decision(False, "no conversation id")
    if record.get("model") != model:
        return Decision(False, "model changed")
    if record.get("tools_hash") != tools_hash:
        return Decision(False, "tools changed")
    # The agent sets agy's own system prompt and native tools for the whole
    # conversation; a record from before the agent field ran agy's default.
    if agent is not None and (record.get("agent") or "") != agent:
        return Decision(False, "agent changed")
    if system_hash is not None and refresh_on_system_change() and record.get("system_hash") != system_hash:
        return Decision(False, "system prompt changed")
    if int(record.get("agy_size_tokens") or 0) > refresh_size_tokens():
        return Decision(False, "conversation size")
    if record.get("last_request_fp") == request_fingerprint(messages):
        # agy already answered this exact request; Hermes did not keep that reply.
        return Decision(False, "last reply not kept (same request re-sent)")
    index = last_assistant_index(messages)
    # The reply as agy streamed it, or as Hermes stored it (see hermes_stored_content).
    expected = {fp for fp in (record.get("last_reply_fp"), record.get("last_reply_stored_fp")) if fp}
    if index is None or reply_fingerprint(messages[index]) not in expected:
        return Decision(False, "fingerprint mismatch")
    if user_fingerprint(messages, index) != record.get("last_user_fp"):
        return Decision(False, "earlier request changed")
    delta = [m for m in messages[index + 1:] if isinstance(m, dict)]
    if not delta:
        return Decision(False, "empty delta")
    return Decision(True, "resume", delta)


# --------------------------------------------------------------------------
# Packing
# --------------------------------------------------------------------------


def hermes_request_tokens(messages: Sequence[Any], tools: list[dict[str, Any]] | None,
                          tool_choice: Any) -> int:
    """Estimated tokens of a fresh send of the whole Hermes request.

    This is the size a persistent turn reports to Hermes as ``prompt_tokens``
    instead of agy's accumulated conversation, so Hermes' compaction measures
    its own history: after a compaction the next turn reports the reduced
    size (and the system prompt Hermes rebuilds at a compaction normally
    refreshes agy's conversation from it on that next turn). It counts the preamble, tool schemas, system prompt and the
    whole transcript, tool output in full, so it measures what Hermes' own
    estimate, LCM-X and native compression measure. A rebuild only ever
    sends less (old tool output cut to a stub, the cap), so a compaction
    threshold below the rebuild cap still guarantees a rebuild that fits.
    It reads text only: rendering would resolve media references (writing
    data URLs to the cache, downloading remote images) on every turn.
    """
    sections = [*_PROMPT_PREAMBLE, *_tool_sections(tools, tool_choice)]
    history: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        if _role(message) == "system":
            sections.append(_flatten_content(message.get("content")))
        else:
            history.append(message)
    for message in history:
        sections.append(_flatten_content(message.get("content")))
        for call in message.get("tool_calls") or []:
            if isinstance(call, dict):
                function = call.get("function") or {}
                sections.append(f"{function.get('name') or ''} {function.get('arguments') or ''}")
    return estimate_tokens("\n\n".join(s for s in sections if s))


def _split_text_bytes(text: str, size: int) -> list[str]:
    """Cut text into pieces of at most ``size`` UTF-8 bytes on char boundaries."""
    data = text.encode("utf-8")
    if len(data) <= size:
        return [text]
    pieces = []
    start = 0
    while start < len(data):
        end = min(start + size, len(data))
        while end < len(data) and (data[end] & 0xC0) == 0x80:
            end -= 1
        pieces.append(data[start:end].decode("utf-8"))
        start = end
    return pieces


def _presplit(entries: list[str], capacity: int) -> list[str]:
    out: list[str] = []
    for entry in entries:
        if utf8_len(entry) > capacity:
            pieces = _split_text_bytes(entry, max(capacity // 2, 1))
            total = len(pieces)
            out.extend(f"[message continued, piece {i} of {total}]\n{p}" if i > 1 else p
                       for i, p in enumerate(pieces, start=1))
        else:
            out.append(entry)
    return out


def _joined_len(parts: Sequence[str]) -> int:
    if not parts:
        return 0
    return sum(utf8_len(p) for p in parts) + 2 * (len(parts) - 1)


def _fill_from_end(entries: list[str], capacity: int, base: int) -> list[str]:
    """Move the newest entries that fit (contiguously) off ``entries``."""
    taken: list[str] = []
    size = base
    while entries:
        add = utf8_len(entries[-1]) + 2
        if size + add > capacity:
            break
        taken.insert(0, entries.pop())
        size += add
    return taken


def _join(sections: Sequence[str]) -> str:
    return "\n\n".join(s.strip() for s in sections if s and s.strip())


def _final_message(header: str | None, entries: list[str], closing: list[str]) -> str:
    sections = [header] if header else []
    if entries:
        sections.append("Conversation transcript (most recent part):\n\n" + "\n\n".join(entries))
    sections.extend(closing)
    return _join(_apply_media_viewing_protocol([s for s in sections if s]))


# Messages that carry a compactor's summary or preserved objective. Both
# compactors put them at the oldest end of the history (native: a
# "[CONTEXT COMPACTION ..." handoff row, or a merged "[PRIOR CONTEXT ..."
# carrier; LCM-X: summary blocks and the preserved objective in the first user
# message), which is exactly what an oldest-first cap would drop first. They
# are mandatory in a rebuild, kept right after the static prefix. Hermes
# strips its private ``_compressed_summary`` key before providers see the
# request, so detection is by the markers the compactors write, anchored at
# the start of the message: a message that merely quotes a header is ordinary.
_SUMMARY_PREFIXES = (
    "[CONTEXT COMPACTION",
    "[CONTEXT SUMMARY]:",
    "[PRIOR CONTEXT",
    "[Current user objective preserved from compacted history]",
    "[Your active task list was preserved",
    "[Planning state preserved",
)
_SUMMARY_HEADER_RE = re.compile(r"\[(?:Recent|Session Arc|Durable|Depth-\d+) Summary \(d\d+, node \d+\)\]")


def is_summary_message(message: Any) -> bool:
    """A compaction summary / preserved-objective message (kept in every rebuild)."""
    if _role(message) not in ("user", "assistant"):
        return False
    text = _flatten_content(message.get("content")).lstrip()
    return text.startswith(_SUMMARY_PREFIXES) or bool(_SUMMARY_HEADER_RE.match(text))


# How many of the newest tool results keep their full output in a split
# rebuild, tried in order until the active tool loop fits its message. The
# first step matches the canonical one-message prompt (prompt.py keeps the 8
# newest tool results and cuts older ones to a stub).
TOOL_RETENTION_STEPS = (8, 4, 1, 0)
_PRUNED_TOOL_MAX_CHARS = 300


def _prune_tool_results(history: list[dict[str, Any]], keep: int) -> list[dict[str, Any]]:
    """Cut every tool result but the ``keep`` newest to a stub, as prompt.py does."""
    tool_indices = [i for i, m in enumerate(history) if _role(m) == "tool"]
    retained = set(tool_indices[-keep:]) if keep > 0 else set()
    out = []
    for index, message in enumerate(history):
        if _role(message) == "tool" and index not in retained:
            text = _flatten_content(message.get("content"))
            if len(text) > _PRUNED_TOOL_MAX_CHARS:
                message = dict(message, content=(
                    text[:200] + f"\n[... tool output truncated: {len(text)} chars ...]\n" + text[-100:]
                ))
        out.append(message)
    return out


def _split_history(history: list[dict[str, Any]], media_sink: Optional[set]):
    """(summary entries, older entries, active entries, closing sections).

    ``history`` holds the non-system messages. The active part is the latest
    user request plus its whole tool loop: everything from the last real user
    message on. It must survive packing, and so must the summary messages
    before it (``is_summary_message``).
    """
    loop_start = len(history) - 1 if history else 0
    for index in range(len(history) - 1, -1, -1):
        if _role(history[index]) == "user" and not _is_recovery_nudge(history[index]):
            loop_start = index
            break
    summaries: list[str] = []
    older: list[str] = []
    for message in history[:loop_start]:
        entry = _render_transcript_entry(message)
        if entry:
            (summaries if is_summary_message(message) else older).append(entry)
    # The closing restates the request in full: a rebuild sends it once, and
    # it is the anchor the final part answers.
    active, closing = _delta_parts(history[loop_start:], _latest_user_text(history), media_sink,
                                   full_pending_request=True)
    return summaries, older, active, closing


def _prefix_sections(messages, tools, tool_choice, media_sink) -> list[str]:
    sections = [*_PROMPT_PREAMBLE, *_tool_sections(tools, tool_choice)]
    system = _system_section(messages, media_sink)
    if system:
        sections.append(system)
    prefix_size = _joined_len(sections)
    if prefix_size > PREFIX_MAX_BYTES:
        raise StdinPayloadTooLarge(
            f"Antigravity prompt prefix (preamble, tools and system instructions) is "
            f"{prefix_size:,} UTF-8 bytes, over the {PREFIX_MAX_BYTES:,}-byte limit; "
            "refusing to send a truncated prompt."
        )
    return sections


_SUMMARY_TRUNCATED = "\n[summary truncated to fit]"


def _omission_marker(count: int) -> str:
    return f"[{count} earlier message{'s' if count != 1 else ''} omitted]"


class _ActiveLoopTooLarge(StdinPayloadTooLarge):
    """The request and its active tool loop do not fit one message."""


def _require_active(omitted: int, older_count: int) -> None:
    if omitted > older_count:
        raise _ActiveLoopTooLarge(
            "Antigravity prompt: the latest request and its active tool loop alone "
            f"exceed agy's stdin limit of {MAX_STDIN_BYTES:,} bytes per message."
        )


def _fill_forward(entries: list[str], capacity: int, base: int) -> list[str]:
    """Move the oldest entries that fit (contiguously) off ``entries``."""
    taken: list[str] = []
    size = base
    while entries:
        add = utf8_len(entries[0]) + 2
        if size + add > capacity:
            break
        taken.append(entries.pop(0))
        size += add
    return taken


def _cap_history(older: list[str], fixed_tokens: int, cap: int) -> tuple[list[str], int, int]:
    """Keep the newest older entries that fit the token cap.

    Returns (kept entries, omitted message count, omitted estimated tokens).
    """
    budget = cap - fixed_tokens
    kept_from = len(older)
    used = 0
    for index in range(len(older) - 1, -1, -1):
        tokens = estimate_tokens(older[index])
        if used + tokens > budget:
            break
        used += tokens
        kept_from = index
    dropped = older[:kept_from]
    return older[kept_from:], len(dropped), sum(estimate_tokens(e) for e in dropped)


def _single_message(prefix: list[str], marker: list[str], entries: list[str], closing: list[str]) -> str:
    sections = list(prefix)
    transcript = marker + entries
    if transcript:
        sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))
    sections.extend(closing)
    return _join(_apply_media_viewing_protocol(sections))


def pack_refresh(
    messages: Sequence[dict[str, Any]],
    *,
    model: str | None,
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    media_sink: Optional[set] = None,
    max_messages: int | None = None,
    max_tokens: int | None = None,
) -> list[str]:
    """Pack a full history into stdin messages for a fresh agy conversation.

    One message when the canonical full prompt fits. Otherwise as many
    context-only seeds (each answered internally) as the history needs, then a
    final part. The final part carries the latest request and its whole tool
    loop, filled backwards with the newest history; the first seed carries
    the static prefix and any compaction summary; the seeds between carry the
    rest of the history in order. Every message stays under agy's stdin byte
    limit.

    The whole rebuild is capped at ``max_tokens`` estimated tokens (default
    ``refresh_max_tokens()``). The prefix, compaction summaries and the active
    loop are mandatory; only ordinary history beyond the cap is dropped,
    oldest first, behind an "N earlier messages omitted" marker, with a
    warning. Old tool output is cut to a stub as in the canonical prompt, and
    more of it when the active loop would not fit otherwise
    (``TOOL_RETENTION_STEPS``). A loop that fits no single message even then
    is spread over the last seeds at the canonical retention; the final part
    keeps its newest entries beside the restated request.

    ``max_messages=1`` (one-shot) instead keeps the newest history that fits
    one message and omits the rest, with a warning.
    """
    full = _format_messages_as_prompt(
        list(messages), model=model, tools=tools, tool_choice=tool_choice, media_sink=media_sink
    )
    prefix = _prefix_sections(messages, tools, tool_choice, media_sink)
    cap = refresh_max_tokens() if max_tokens is None else max_tokens
    single = max_messages is not None and max_messages <= 1
    if utf8_len(full) < MAX_STDIN_BYTES and (single or estimate_tokens(full) <= cap):
        return [full]

    history = [m for m in messages if isinstance(m, dict) and _role(m) != "system"]
    failure: StdinPayloadTooLarge | None = None
    for keep in TOOL_RETENTION_STEPS:
        pruned = _prune_tool_results(history, keep)
        try:
            if single:
                payloads = _pack_single(pruned, prefix, media_sink)
            else:
                payloads = _pack_split(pruned, prefix, cap, media_sink)
        except _ActiveLoopTooLarge as exc:
            failure = exc
            continue
        if keep != TOOL_RETENTION_STEPS[0]:
            logger.warning(
                "Antigravity rebuild: the active tool loop only fits with the %d newest tool "
                "result(s) in full; older tool output was cut to a stub", keep,
            )
        return payloads
    assert failure is not None
    if single:
        raise failure
    # Even with every tool result cut to a stub the loop does not fit one
    # message (a long request restated in the closing, large tool-call
    # arguments). Spill its oldest entries into context-only parts instead,
    # at the canonical retention, so the final part keeps the newest tool
    # results in full.
    return _pack_split(_prune_tool_results(history, TOOL_RETENTION_STEPS[0]), prefix, cap,
                       media_sink, spill_active=True)


def _pack_single(history: list[dict[str, Any]], prefix: list[str], media_sink: Optional[set]) -> list[str]:
    """One message: prefix, summaries, the newest history that fits, the request."""
    summaries, older, active, closing = _split_history(history, media_sink)
    capacity = MAX_STDIN_BYTES - _HEADER_RESERVE
    summary_pieces = _presplit(summaries, capacity // 2)
    older_pieces = _presplit(older, capacity // 2)
    active_pieces = _presplit(active, capacity // 2)
    marker_reserve = utf8_len(_omission_marker(len(older_pieces) + len(summary_pieces))) + 64
    base = _joined_len(prefix) + _joined_len(closing) + marker_reserve + 64
    # Summaries come before ordinary history: they take the room the active
    # loop leaves (the newest summary pieces first, if not all fit), and
    # ordinary history only what is left after them.
    dropped_summaries = list(summary_pieces)
    summary_room = capacity - _joined_len(active_pieces) - 2
    head = _fill_from_end(dropped_summaries, summary_room, base)
    if head:
        base += _joined_len(head) + 2
    truncated = 0
    room = summary_room - base - utf8_len(_SUMMARY_TRUNCATED) - 2
    if dropped_summaries and room >= 1024:
        # The next summary piece does not fit whole: keep its beginning
        # rather than give its room to ordinary history.
        piece = dropped_summaries.pop()
        cut = _split_text_bytes(piece, room)[0] + _SUMMARY_TRUNCATED
        truncated = utf8_len(piece) - utf8_len(cut)
        head = [cut] + head
        base += utf8_len(cut) + 2
    entries = older_pieces + active_pieces
    tail = _fill_from_end(entries, capacity, base)
    _require_active(len(entries), len(older_pieces))
    omitted = dropped_summaries + entries
    marker = [_omission_marker(len(omitted))] if omitted else []
    if omitted:
        logger.warning(
            "Antigravity one-shot prompt omits the %d oldest history piece(s) (~%d estimated tokens, "
            "%d of them summary pieces) to fit agy's %d-byte stdin limit; kept %d and %d summary piece(s)",
            len(omitted), sum(estimate_tokens(e) for e in omitted), len(dropped_summaries),
            MAX_STDIN_BYTES, len(tail), len(head),
        )
    if truncated:
        logger.warning("Antigravity one-shot prompt cut the oldest kept summary piece by %d bytes "
                       "to fit agy's %d-byte stdin limit", truncated, MAX_STDIN_BYTES)
    return [_single_message(prefix, head + marker, tail, closing)]


def _pack_split(history: list[dict[str, Any]], prefix: list[str], cap: int,
                media_sink: Optional[set], spill_active: bool = False) -> list[str]:
    """Context-only seeds, then a final part with the closing.

    The final part normally carries the whole active loop. With
    ``spill_active`` the loop's oldest entries may go to the last seeds
    instead: the final part then holds the newest tail of the loop that fits
    beside the closing, which restates the request the loop serves. Seeds
    that carry any of the loop are labelled ``ACTIVE_LOOP_HEADER`` (and the
    final part ``ACTIVE_LOOP_FINAL_HEADER``); seeds of older history keep
    ``SEED_HEADER``.
    """
    summaries, older, active, closing = _split_history(history, media_sink)
    capacity = MAX_STDIN_BYTES - _HEADER_RESERVE
    final_base = _joined_len(closing) + 64
    final_room = capacity - final_base
    if spill_active and final_room < _MIN_FINAL_ROOM:
        # The closing is the request itself (restated verbatim): it cannot be
        # split across parts without the model losing what it has to answer.
        # The wording must keep "exceeds agy's stdin limit": Hermes compacts
        # on it (__init__._classify_antigravity_error).
        raise StdinPayloadTooLarge(
            f"Antigravity prompt: the latest request and its instructions "
            f"({_joined_len(closing):,} UTF-8 bytes) plus the tool loop it serves exceed agy's "
            f"stdin limit of {MAX_STDIN_BYTES:,} bytes per message, leaving no room for the loop "
            "in the final part; refusing to send a truncated prompt."
        )

    # Token cap over the whole rebuild: the prefix, the summaries, the request
    # and its active loop are mandatory; older history fills the rest, newest
    # first.
    fixed = (estimate_tokens(_join(prefix)) + estimate_tokens(_join(closing))
             + sum(estimate_tokens(e) for e in summaries)
             + sum(estimate_tokens(e) for e in active) + 256)
    kept, omitted, omitted_tokens = _cap_history(older, fixed, cap)
    if omitted:
        logger.warning(
            "Antigravity refresh rebuild exceeds the %d-token cap: dropped the %d oldest "
            "message(s) (~%d estimated tokens); kept %d and %d summary message(s)",
            cap, omitted, omitted_tokens, len(kept), len(summaries),
        )
    marker = [_omission_marker(omitted)] if omitted else []

    head_pieces = _presplit(summaries, capacity // 2) + marker
    older_pieces = _presplit(kept, capacity // 2)
    # When spilling, no active piece may exceed the room beside the closing,
    # so the final part always carries at least the loop's newest piece.
    # (2 bytes for the separator _fill_from_end counts per entry).
    active_pieces = _presplit(active, min(capacity // 2, final_room - 2) if spill_active else capacity // 2)
    entries = head_pieces + older_pieces + active_pieces
    final_entries = _fill_from_end(entries, capacity, final_base)
    if not spill_active:
        # The active loop must sit in the final part.
        _require_active(len(entries), len(head_pieces) + len(older_pieces))
    elif len(entries) > len(head_pieces) + len(older_pieces):
        spilled = len(entries) - len(head_pieces) - len(older_pieces)
        logger.warning(
            "Antigravity refresh rebuild: the active tool loop does not fit one message; its %d "
            "oldest piece(s) go to active-tool-loop parts and the final part keeps the newest %d",
            spilled, len(active_pieces) - spilled,
        )

    if not entries and _joined_len(prefix) + _joined_len(final_entries) \
            + _joined_len(closing) + 256 < capacity:
        # Only the cap forced a split: everything that is left fits one message.
        return [_single_message(prefix, [], final_entries, closing)]

    # Entries past this index are pieces of the active loop (only present in
    # the seeds when spilling).
    history_count = len(head_pieces) + len(older_pieces)
    chunks: list[list[str]] = [_fill_forward(entries, capacity, _joined_len(prefix) + 128)]
    while entries:
        chunk = _fill_forward(entries, capacity, 128)
        if not chunk:  # cannot happen after _presplit; never loop forever
            raise StdinPayloadTooLarge("Antigravity prompt: history entry too large to split.")
        chunks.append(chunk)

    total = len(chunks) + 1
    payloads = []
    consumed = 0
    loop_spilled = False
    for k, chunk in enumerate(chunks, start=1):
        consumed += len(chunk)
        carries_loop = consumed > history_count
        loop_spilled = loop_spilled or carries_loop
        header = ACTIVE_LOOP_HEADER if carries_loop else SEED_HEADER
        sections = [header.format(k=k, n=total)]
        if k == 1:
            sections.extend(prefix)
            label = "Conversation transcript (earlier part):"
        else:
            label = "Conversation transcript (continued):"
        if chunk:
            sections.append(label + "\n\n" + "\n\n".join(chunk))
        payloads.append(_join(sections))
    final_header = ACTIVE_LOOP_FINAL_HEADER if loop_spilled else FINAL_HEADER
    payloads.append(_final_message(final_header.format(n=total), final_entries, closing))
    logger.info(
        "Antigravity refresh rebuild: %d message(s), ~%d estimated tokens, %d history message(s) omitted",
        len(payloads), sum(estimate_tokens(p) for p in payloads), omitted,
    )
    return payloads


def pack_oneshot(
    messages: Sequence[dict[str, Any]],
    *,
    model: str | None,
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    media_sink: Optional[set] = None,
) -> str:
    """A single prompt under the byte limit (oldest history omitted if needed)."""
    return pack_refresh(
        messages, model=model, tools=tools, tool_choice=tool_choice,
        media_sink=media_sink, max_messages=1,
    )[0]


def split_delta(
    delta: Sequence[dict[str, Any]],
    *,
    pending_user_text: str = "",
    media_sink: Optional[set] = None,
) -> list[str]:
    """A resume delta as stdin messages, each under the byte limit.

    Normally one message. An oversized delta becomes context-only seeds
    (answered with ACK) followed by a final part that carries the request.
    """
    full = _format_delta_prompt(delta, pending_user_text=pending_user_text, media_sink=media_sink)
    if utf8_len(full) < MAX_STDIN_BYTES:
        return [full]
    entries, closing = _delta_parts(delta, pending_user_text, media_sink)
    capacity = MAX_STDIN_BYTES - _HEADER_RESERVE
    if _joined_len(closing) + 64 > capacity:
        raise StdinPayloadTooLarge(
            f"Antigravity prompt: the latest request alone exceeds agy's stdin limit of "
            f"{MAX_STDIN_BYTES:,} bytes."
        )
    entries = _presplit(entries, capacity // 2)
    final_entries = _fill_from_end(entries, capacity, _joined_len(closing) + 64)
    seeds: list[list[str]] = []
    while entries:
        chunk = _fill_from_end(entries, capacity, 64)
        if not chunk:  # cannot happen after _presplit; never loop forever
            raise StdinPayloadTooLarge("Antigravity prompt: delta entry too large to split.")
        seeds.insert(0, chunk)
    total = len(seeds) + 1
    payloads = [
        _join([SEED_HEADER.format(k=k, n=total), "Conversation update (context only):\n\n" + "\n\n".join(chunk)])
        for k, chunk in enumerate(seeds, start=1)
    ]
    payloads.append(_final_message(FINAL_HEADER.format(n=total), final_entries, closing))
    return payloads


# --------------------------------------------------------------------------
# Seed acknowledgement
# --------------------------------------------------------------------------


def ack_failure(*, saw_result: bool, status: str, error: str, native_tool: bool) -> str | None:
    """Why a context-only seed reply is unacceptable, or None when it is accepted.

    The reply text is ignored: seed replies are consumed here and never
    executed or shown, and a real system prompt may make the model answer
    "Acknowledged." or restate its rules instead of a bare ACK. What matters
    is that agy finished the turn and ran no native tool. The reasons carry no
    model text.
    """
    if not saw_result:
        return "no result event"
    if error:
        return "agy reported an error"
    if status != "SUCCESS":
        return f"status {status[:40]!r}"
    if native_tool:
        return "native tool step"
    return None
