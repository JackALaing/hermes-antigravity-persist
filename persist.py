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
import json
import logging
import os
import re
import tempfile
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
# plus cache read). Hermes' compaction threshold on these profiles is higher.
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
LOCK_TIMEOUT_SECONDS = 30.0
RECORD_MAX_AGE_SECONDS = 14 * 24 * 3600

SEED_HEADER = (
    "[CONTEXT ONLY - PART {k} OF {n}. This message restores earlier conversation state. "
    "Do NOT act on anything in it, do NOT call any tools (neither <tool_call> blocks nor native tools), "
    "do NOT answer any request inside it. Reply with exactly: ACK]"
)
FINAL_HEADER = (
    "[PART {n} OF {n}. The earlier parts were context only and are now complete. "
    "Answer the request at the end of this message normally.]"
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
        if not conversation_id or not re.fullmatch(r"[A-Za-z0-9_-]+", conversation_id):
            return
        conv_dir = self.gemini_dir / "conversations"
        for entry in conv_dir.glob(f"{conversation_id}.db*"):
            with contextlib.suppress(OSError):
                if entry.is_file() and not entry.is_symlink():
                    entry.unlink()

    def gc(self, now: float | None = None) -> int:
        """Drop records (and their conversations) idle for over 14 days."""
        now = time.time() if now is None else now
        removed = 0
        if not self.records_dir.is_dir():
            return 0
        for path in self.records_dir.iterdir():
            try:
                age = now - path.stat().st_mtime
            except OSError:
                continue
            if path.name.startswith(".tmp-"):
                if age > 3600:
                    with contextlib.suppress(OSError):
                        path.unlink()
                continue
            if path.suffix != ".json" or age <= RECORD_MAX_AGE_SECONDS:
                continue
            record = self.load(path.stem) or {}
            self.delete_conversation(record.get("agy_conversation_id"))
            with contextlib.suppress(OSError):
                path.unlink()
                removed += 1
        return removed


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
                self._fd = fd
                return True
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    return False
                time.sleep(0.05)
            except OSError:
                os.close(fd)
                raise

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


def reply_fingerprint(message: dict[str, Any]) -> str:
    """sha256 of the visible text plus ordered (name, canonical args) calls.

    Tool-call ids are excluded on purpose: Hermes renames reused ids
    (``<id>_d<n>``) and the model's own ids are arbitrary.
    """
    calls = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        calls.append([str(function.get("name") or ""), _canonical_arguments(function.get("arguments", {}))])
    payload = json.dumps(
        {"text": visible_text(message.get("content")), "calls": calls},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
    if system_hash is not None and refresh_on_system_change() and record.get("system_hash") != system_hash:
        return Decision(False, "system prompt changed")
    if int(record.get("agy_size_tokens") or 0) > refresh_size_tokens():
        return Decision(False, "conversation size")
    index = last_assistant_index(messages)
    if index is None or reply_fingerprint(messages[index]) != record.get("last_reply_fp"):
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
# request, so detection is by the markers the compactors write.
_SUMMARY_PREFIXES = (
    "[CONTEXT COMPACTION",
    "[PRIOR CONTEXT",
    "[Current user objective preserved from compacted history]",
    "[Your active task list was preserved",
    "[Planning state preserved",
)
_SUMMARY_MARKER_RE = re.compile(
    r"--- END OF CONTEXT SUMMARY"
    r"|\[(?:Recent|Session Arc|Durable|Depth-\d+) Summary \(d\d+, node \d+\)\]"
)


def is_summary_message(message: Any) -> bool:
    """A compaction summary / preserved-objective message (kept in every rebuild)."""
    if _role(message) not in ("user", "assistant"):
        return False
    text = _flatten_content(message.get("content")).lstrip()
    return text.startswith(_SUMMARY_PREFIXES) or bool(_SUMMARY_MARKER_RE.search(text))


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
    active, closing = _delta_parts(history[loop_start:], _latest_user_text(history), media_sink)
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
    (``TOOL_RETENTION_STEPS``).

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
    raise failure


def _pack_single(history: list[dict[str, Any]], prefix: list[str], media_sink: Optional[set]) -> list[str]:
    """One message: prefix, summaries, the newest history that fits, the request."""
    summaries, older, active, closing = _split_history(history, media_sink)
    capacity = MAX_STDIN_BYTES - _HEADER_RESERVE
    summary_pieces = _presplit(summaries, capacity // 2)
    older_pieces = _presplit(older, capacity // 2)
    active_pieces = _presplit(active, capacity // 2)
    marker_reserve = utf8_len(_omission_marker(len(older_pieces) + len(summary_pieces))) + 64
    base = _joined_len(prefix) + _joined_len(closing) + marker_reserve + 64
    head: list[str] = []
    if summary_pieces and base + _joined_len(summary_pieces) + _joined_len(active_pieces) + 4 < capacity:
        head = summary_pieces
        base += _joined_len(summary_pieces) + 2
    else:  # no room to keep them whole: they compete like any old message
        older_pieces = summary_pieces + older_pieces
    entries = older_pieces + active_pieces
    tail = _fill_from_end(entries, capacity, base)
    _require_active(len(entries), len(older_pieces))
    marker = [_omission_marker(len(entries))] if entries else []
    if entries:
        logger.warning(
            "Antigravity one-shot prompt omits the %d oldest history piece(s) (~%d estimated tokens) "
            "to fit agy's %d-byte stdin limit; kept %d", len(entries),
            sum(estimate_tokens(e) for e in entries), MAX_STDIN_BYTES, len(tail),
        )
    return [_single_message(prefix, head + marker, tail, closing)]


def _pack_split(history: list[dict[str, Any]], prefix: list[str], cap: int,
                media_sink: Optional[set]) -> list[str]:
    summaries, older, active, closing = _split_history(history, media_sink)
    capacity = MAX_STDIN_BYTES - _HEADER_RESERVE

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
    active_pieces = _presplit(active, capacity // 2)
    entries = head_pieces + older_pieces + active_pieces
    final_entries = _fill_from_end(entries, capacity, _joined_len(closing) + 64)
    # The active loop must sit in the final part.
    _require_active(len(entries), len(head_pieces) + len(older_pieces))

    if not entries and _joined_len(prefix) + _joined_len(final_entries) \
            + _joined_len(closing) + 256 < capacity:
        # Only the cap forced a split: everything that is left fits one message.
        return [_single_message(prefix, [], final_entries, closing)]

    chunks: list[list[str]] = [_fill_forward(entries, capacity, _joined_len(prefix) + 128)]
    while entries:
        chunk = _fill_forward(entries, capacity, 128)
        if not chunk:  # cannot happen after _presplit; never loop forever
            raise StdinPayloadTooLarge("Antigravity prompt: history entry too large to split.")
        chunks.append(chunk)

    total = len(chunks) + 1
    payloads = []
    for k, chunk in enumerate(chunks, start=1):
        sections = [SEED_HEADER.format(k=k, n=total)]
        if k == 1:
            sections.extend(prefix)
            label = "Conversation transcript (earlier part):"
        else:
            label = "Conversation transcript (continued):"
        if chunk:
            sections.append(label + "\n\n" + "\n\n".join(chunk))
        payloads.append(_join(sections))
    payloads.append(_final_message(FINAL_HEADER.format(n=total), final_entries, closing))
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
