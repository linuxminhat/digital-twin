from pathlib import Path

import pytest

from src.chunker import load_cards
from src.cli import REFUSAL, compose_answer, main
from src.hybrid_search import HybridSearchEngine

KB = Path(__file__).resolve().parent.parent / "knowledge-base" / "minh-primary-evidence.md"


@pytest.fixture(scope="module")
def engine():
    return HybridSearchEngine(load_cards(KB))


def test_answer_contains_citation(engine):
    answer = compose_answer(engine.search("Which embedding model was fine-tuned?"))
    assert "all-mpnet-base-v2" in answer
    assert "[E02]" in answer


def test_refusal_message(engine):
    assert compose_answer(engine.search("What was Minh's salary at Google?")) == REFUSAL


def test_main_single_query(capsys):
    assert main(["--query", "What hardware does smartIOTproject use?", "--debug"]) == 0
    out = capsys.readouterr().out
    assert "Arduino UNO" in out and "[E04]" in out
