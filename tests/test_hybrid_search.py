from pathlib import Path

import numpy as np
import pytest

from src.chunker import load_cards
from src.hybrid_search import HashingEmbedder, HybridSearchEngine, reciprocal_rank_fusion, tokenize

KB = Path(__file__).resolve().parent.parent / "knowledge-base" / "minh-primary-evidence.md"


@pytest.fixture(scope="module")
def engine():
    return HybridSearchEngine(load_cards(KB))


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[0, 1, 2], [1, 0, 2]], k=60)
    assert fused[0] == pytest.approx(fused[1])
    assert fused[0] > fused[2]
    assert fused[2] == pytest.approx(2 / 63)


def test_tokenizer_splits_identifiers_and_folds_accents():
    toks = tokenize("Spring_Boot Đại học streaming")
    assert {"spring_boot", "spring", "boot", "dai", "hoc", "stream"} <= set(toks)


def test_hashing_embedder_is_normalised_and_deterministic():
    emb = HashingEmbedder()
    a, b = emb.encode(["hybrid search"]), emb.encode(["hybrid search"])
    assert np.allclose(a, b)
    assert np.linalg.norm(a[0]) == pytest.approx(1.0, rel=1e-5)


def test_bm25_finds_exact_repository_name(engine):
    top = engine.bm25_rank("smartIOTproject")[0]
    assert engine.cards[top].card_id == "E04"


def test_dense_ranks_semantic_query(engine):
    top = engine.dense_rank("recruitment platform resume screening")[0]
    assert engine.cards[top].card_id == "E02"


@pytest.mark.parametrize(
    ("query", "card"),
    [
        ("What is ITSmartHire?", "E02"),
        ("Which university did Minh attend?", "E00"),
        ("How does Minh approach ambiguous engineering problems?", "E05"),
    ],
)
def test_end_to_end_retrieval(engine, query, card):
    res = engine.search(query)
    assert not res.refused
    assert res.card_ids[0] == card
    assert len(res.chunks) <= 5


@pytest.mark.parametrize(
    "query",
    [
        "What was Minh's salary?",
        "Did Minh work at Microsoft?",
        "Which Kubernetes clusters has Minh managed?",
    ],
)
def test_guardrail_refuses_out_of_domain(engine, query):
    res = engine.search(query)
    assert res.refused
    assert res.refusal_reason


def test_empty_index_rejected():
    with pytest.raises(ValueError):
        HybridSearchEngine([])
