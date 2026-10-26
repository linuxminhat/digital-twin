"""Evidence-card chunker.

Splits a markdown knowledge base strictly on the ``===END EVIDENCE CARD===``
delimiter and extracts structured metadata from each card header.

A card looks like::

    ## E02 — Project: ITSmartHire
    **Domain:** software_engineering
    **Source Type:** resume_verified
    **Tags:** project, itsmarthire, fastapi
    **Confidence:** High

    - body bullet ...
    ===END EVIDENCE CARD===

An optional YAML front-matter block (``---`` ... ``---``) directly after the
heading is also supported and takes precedence over ``**Field:**`` lines.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator

DELIMITER = "===END EVIDENCE CARD==="
MAX_CHUNK_CHARS = 2000

_HEADING_RE = re.compile(r"^##\s+(E\d{2,})\s*[—\-:]+\s*(.+?)\s*$", re.MULTILINE)
_FIELD_RE = re.compile(r"^\*\*(?P<key>[A-Za-z][A-Za-z ]*):\*\*\s*(?P<value>.*?)\s*$", re.MULTILINE)
_FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
# The delimiter only counts when it stands on its own line, so documentation that
# mentions it inline (e.g. in backticks) is never split.
_DELIMITER_LINE_RE = re.compile(rf"^[ \t]*{re.escape(DELIMITER)}[ \t]*$", re.MULTILINE)


class ChunkValidationError(ValueError):
    """Raised when a chunk violates the evidence-card contract."""


class EvidenceCard(BaseModel):
    card_id: str
    title: str
    domain: str
    tags: list[str] = Field(default_factory=list)
    source_type: str
    confidence: str
    source: str | None = None
    timeline: str | None = None
    text: str

    @field_validator("confidence")
    @classmethod
    def _normalise_confidence(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in {"high", "medium", "low"}:
            raise ValueError(f"confidence must be high|medium|low, got {v!r}")
        return v

    @property
    def metadata(self) -> dict:
        return self.model_dump(exclude={"text"})


def _snake(key: str) -> str:
    return key.strip().lower().replace(" ", "_")


def _parse_header(body: str) -> tuple[dict, str]:
    """Return (metadata, body_without_front_matter)."""
    meta: dict = {}
    fm = _FRONT_MATTER_RE.match(body)
    if fm:
        loaded = yaml.safe_load(fm.group(1)) or {}
        if not isinstance(loaded, dict):
            raise ChunkValidationError("front matter must be a mapping")
        meta.update({_snake(k): v for k, v in loaded.items()})
        body = body[fm.end() :]
    for m in _FIELD_RE.finditer(body):
        meta.setdefault(_snake(m.group("key")), m.group("value"))
    tags = meta.get("tags", [])
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    meta["tags"] = tags
    return meta, body


def _validate_self_contained(raw: str, card_id: str) -> None:
    headings = _HEADING_RE.findall(raw)
    if len(headings) != 1:
        raise ChunkValidationError(
            f"{card_id}: chunk must contain exactly one card heading, found {len(headings)} "
            "(missing delimiter → text bleeding between cards)"
        )
    if _DELIMITER_LINE_RE.search(raw):
        raise ChunkValidationError(f"{card_id}: delimiter leaked into chunk body")
    if len(raw) > MAX_CHUNK_CHARS:
        raise ChunkValidationError(f"{card_id}: {len(raw)} chars exceeds limit of {MAX_CHUNK_CHARS}")


def parse_cards(markdown: str) -> list[EvidenceCard]:
    """Split ``markdown`` into validated :class:`EvidenceCard` objects.

    Text before the first card heading (the retrieval contract preamble) is
    discarded. Every segment that contains a card heading must be closed by the
    delimiter; an unterminated trailing card raises ``ChunkValidationError``.
    """
    segments = _DELIMITER_LINE_RE.split(markdown)
    trailing = segments.pop()  # text after the last delimiter
    if _HEADING_RE.search(trailing):
        raise ChunkValidationError("trailing evidence card is not terminated by the delimiter")

    cards: list[EvidenceCard] = []
    seen: set[str] = set()
    for seg in segments:
        m = _HEADING_RE.search(seg)
        if not m:
            raise ChunkValidationError("segment between delimiters has no evidence card heading")
        raw = seg[m.start() :].strip()
        card_id, title = m.group(1), m.group(2)
        _validate_self_contained(raw, card_id)
        if card_id in seen:
            raise ChunkValidationError(f"duplicate card id {card_id}")
        seen.add(card_id)

        body = raw[m.end() - m.start() :].lstrip("\n")
        meta, _ = _parse_header(body)
        missing = [k for k in ("domain", "source_type", "confidence") if not meta.get(k)]
        if missing:
            raise ChunkValidationError(f"{card_id}: missing header fields {missing}")
        cards.append(
            EvidenceCard(
                card_id=card_id,
                title=title,
                domain=str(meta["domain"]).strip(),
                tags=meta["tags"],
                source_type=str(meta["source_type"]).strip(),
                confidence=str(meta["confidence"]),
                source=meta.get("source"),
                timeline=meta.get("timeline"),
                text=raw,
            )
        )
    return cards


def load_cards(path: str | Path) -> list[EvidenceCard]:
    return parse_cards(Path(path).read_text(encoding="utf-8"))


if __name__ == "__main__":  # pragma: no cover
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "knowledge-base/minh-primary-evidence.md"
    for c in load_cards(target):
        print(f"{c.card_id:4} {c.domain:24} {c.source_type:18} {len(c.text):5} chars  {c.title}")
