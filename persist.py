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
REFRESH_SIZE_TOKENS = 360_000
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


def is_persistent_request(scope: Any, origin: Any) -> bool:
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
) -> Decision:
    """Resume the recorded agy conversation, or refresh, and why.

    A resume is only safe when agy's conversation ends exactly where the
    incoming history's LAST assistant message ends: any assistant message
    after ours (a fallback provider's answer, a fork's reply, a placeholder
    row) is something agy never saw. An effort change alone still resumes;
    the caller respawns the worker with ``--conversation``.
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
    if int(record.get("agy_size_tokens") or 0) > REFRESH_SIZE_TOKENS:
        return Decision(False, "conversation size")
    index = last_assistant_index(messages)
    if index is None or reply_fingerprint(messages[index]) != record.get("last_reply_fp"):
        return Decision(False, "fingerprint mismatch")
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


def _split_history(messages: Sequence[dict[str, Any]], media_sink: Optional[set]):
    """(prefix sections, older entries, active entries, closing sections).

    The active part is the latest user request plus its whole tool loop:
    everything from the last real user message on. It must survive packing.
    """
    history = [m for m in messages if isinstance(m, dict) and _role(m) != "system"]
    loop_start = len(history) - 1 if history else 0
    for index in range(len(history) - 1, -1, -1):
        if _role(history[index]) == "user" and not _is_recovery_nudge(history[index]):
            loop_start = index
            break
    older = [e for e in (_render_transcript_entry(m) for m in history[:loop_start]) if e]
    active, closing = _delta_parts(history[loop_start:], _latest_user_text(history), media_sink)
    return history, older, active, closing


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


def _require_active(omitted: int, older_count: int) -> None:
    if omitted > older_count:
        raise StdinPayloadTooLarge(
            "Antigravity prompt: the latest request and its active tool loop alone "
            f"exceed agy's stdin limit of {MAX_STDIN_BYTES:,} bytes per message."
        )


def pack_refresh(
    messages: Sequence[dict[str, Any]],
    *,
    model: str | None,
    tools: list[dict[str, Any]] | None,
    tool_choice: Any,
    media_sink: Optional[set] = None,
    max_messages: int = 2,
) -> list[str]:
    """Pack a full history into at most ``max_messages`` stdin messages.

    One message when the canonical full prompt fits. Otherwise the last
    message holds the latest request and its whole tool loop, filled
    backwards with the newest history; the first holds the static prefix and
    the older history that fits behind it, and the oldest messages beyond
    that are dropped behind an "N earlier messages omitted" marker. With two
    messages the first is a context-only seed answered with ACK.
    """
    full = _format_messages_as_prompt(
        list(messages), model=model, tools=tools, tool_choice=tool_choice, media_sink=media_sink
    )
    prefix = _prefix_sections(messages, tools, tool_choice, media_sink)
    if utf8_len(full) < MAX_STDIN_BYTES:
        return [full]

    _history, older, active, closing = _split_history(messages, media_sink)
    capacity = MAX_STDIN_BYTES - _HEADER_RESERVE
    entries = _presplit(older + active, capacity // 2)
    older_count = len(entries) - len(_presplit(active, capacity // 2))
    marker_reserve = utf8_len(_omission_marker(len(entries))) + 64

    if max_messages <= 1:
        base = _joined_len(prefix) + _joined_len(closing) + marker_reserve + 64
        tail = _fill_from_end(entries, capacity, base)
        _require_active(len(entries), older_count)
        sections = list(prefix)
        transcript = ([_omission_marker(len(entries))] if entries else []) + tail
        if transcript:
            sections.append("Conversation transcript:\n\n" + "\n\n".join(transcript))
        sections.extend(closing)
        return [_join(_apply_media_viewing_protocol(sections))]

    final_entries = _fill_from_end(entries, capacity, _joined_len(closing) + 64)
    first_entries = _fill_from_end(entries, capacity, _joined_len(prefix) + marker_reserve + 64)
    _require_active(len(entries), older_count)
    seed_sections = [SEED_HEADER.format(k=1, n=2), *prefix]
    transcript = ([_omission_marker(len(entries))] if entries else []) + first_entries
    if transcript:
        seed_sections.append("Conversation transcript (earlier part):\n\n" + "\n\n".join(transcript))
    seed = _join(seed_sections)
    final = _final_message(FINAL_HEADER.format(n=2), final_entries, closing)
    return [seed, final]


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


def ack_failure(
    *, saw_result: bool, status: str, error: str, text: str, native_tool: bool
) -> str | None:
    """Why a context-only seed reply is unacceptable, or None when it is an ACK."""
    if not saw_result:
        return "no result event"
    if error:
        return f"error {error!r}"
    if status != "SUCCESS":
        return f"status {status!r}"
    if native_tool:
        return "native tool step"
    if "<tool_call>" in text:
        return "tool call"
    if not text.strip().startswith("ACK"):
        return f"reply {text.strip()[:40]!r}"
    return None
