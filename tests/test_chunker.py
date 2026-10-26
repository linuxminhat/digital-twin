from pathlib import Path

import pytest

from src.chunker import DELIMITER, ChunkValidationError, load_cards, parse_cards

KB = Path(__file__).resolve().parent.parent / "knowledge-base" / "minh-primary-evidence.md"

CARD = """## E{n} — Card {n}

**Domain:** iot
**Source Type:** github_portfolio
**Tags:** a, b, c
**Confidence:** High

- body line for card {n}
"""


def test_knowledge_base_parses_into_atomic_cards():
    cards = load_cards(KB)
    assert [c.card_id for c in cards] == ["E00", "E01", "E02", "E03", "E04", "E05"]
    for c in cards:
        assert c.text.startswith(f"## {c.card_id}")
        assert all(line.strip() != DELIMITER for line in c.text.splitlines())
        assert c.confidence == "high"
        assert c.domain and c.source_type and c.tags


def test_metadata_extraction():
    card = next(c for c in load_cards(KB) if c.card_id == "E02")
    assert card.domain == "software_engineering"
    assert card.source_type == "resume_verified"
    assert "itsmarthire" in card.tags
    assert card.timeline == "04/2025 – 07/2025"


def test_inline_delimiter_mention_does_not_split():
    md = f"Preamble mentions `{DELIMITER}` inline.\n\n" + CARD.format(n="01") + DELIMITER + "\n"
    assert len(parse_cards(md)) == 1


def test_missing_delimiter_is_detected_as_bleeding():
    md = CARD.format(n="01") + CARD.format(n="02") + DELIMITER + "\n"
    with pytest.raises(ChunkValidationError, match="exactly one card heading"):
        parse_cards(md)


def test_unterminated_trailing_card_rejected():
    md = CARD.format(n="01") + DELIMITER + "\n" + CARD.format(n="02")
    with pytest.raises(ChunkValidationError, match="not terminated"):
        parse_cards(md)


def test_missing_required_header_field():
    md = CARD.format(n="01").replace("**Domain:** iot\n", "") + DELIMITER
    with pytest.raises(ChunkValidationError, match="domain"):
        parse_cards(md)


def test_duplicate_ids_rejected():
    md = (CARD.format(n="01") + DELIMITER + "\n") * 2
    with pytest.raises(ChunkValidationError, match="duplicate"):
        parse_cards(md)


def test_yaml_front_matter_takes_precedence():
    md = (
        "## E09 — YAML card\n---\ndomain: profile\nsource_type: resume_verified\n"
        "tags: [x, y]\nconfidence: medium\n---\n**Domain:** iot\n- body\n" + DELIMITER
    )
    (card,) = parse_cards(md)
    assert card.domain == "profile"
    assert card.tags == ["x", "y"]
    assert card.confidence == "medium"


def test_oversized_card_rejected():
    md = CARD.format(n="01") + ("- filler\n" * 400) + DELIMITER
    with pytest.raises(ChunkValidationError, match="exceeds"):
        parse_cards(md)
