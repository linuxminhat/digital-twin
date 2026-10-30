"""Hybrid retrieval engine: BM25 + dense vectors, fused with RRF, then reranked.

Pipeline::

    query ──► BM25 (sparse, exact identifiers) ──┐
          └─► Dense cosine (semantic)          ──┴─► Reciprocal Rank Fusion
                                                       │
                                                       ▼
                                          Cross-encoder style reranker
                                                       │
                                                       ▼
                                       score-threshold pruning (Top-K ≤ 5)
                                                       │
                                                       ▼
                                         grounding guardrail (refuse?)

Backends are pluggable. The defaults are fully offline and deterministic so the
benchmark is reproducible in CI:

* ``HashingEmbedder`` — word + character n-gram feature hashing into a fixed
  dense vector (L2 normalised). Swap in ``SentenceTransformerEmbedder`` with
  ``DT_EMBEDDER=sentence-transformers`` for learned embeddings.
* ``LexicalCrossReranker`` — scores each (query, card) pair jointly using
  IDF-weighted term coverage, phrase overlap and metadata hits, emulating what a
  cross-encoder such as ``rerank-english-v3.0`` / ``bge-reranker`` contributes.
  Swap in ``BGECrossReranker`` with ``DT_RERANKER=bge``.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from rank_bm25 import BM25Okapi

from src.chunker import EvidenceCard

RRF_K = 60
CANDIDATE_POOL = 10
TOP_K = 5

STOPWORDS = frozenset(
    """a an the and or of to in on for with by at from as is are was were be been being
    do does did done has have had what which who whom whose when where why how this that
    these those it its his her he she they them their you your i me my we our us can could
    would should will shall may might must about into over under any all some tell me
    give list describe explain show please much many there here than then so if not no yes
    s minh minh's nhat dang
    co o khong la gi nao cua va nhung ban anh ay the nhu sao duoc da dang voi cho ve hay""".split()
)  # last line: accent-folded Vietnamese function words

# Words that commonly appear in questions but carry no factual claim. They are
# never treated as "unsupported" by the grounding guardrail.
QUESTION_VOCAB = frozenset(
    """use used using built build work worked working project projects tech technology
    stack technologies role kind type approach approaches think thinks way ways know knows
    information info detail details main key core part parts system systems purpose
    goal goals did does handle handled choose chooses pick picks get got make made""".split()
)


def strip_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


# Query-side glossary: maps question vocabulary onto the terms the knowledge base
# actually uses (classic thesaurus-style query expansion).
GLOSSARY: dict[str, str] = {
    "cv": "resume",
    "jd": "job description",
    "database": "mongodb",
    "db": "mongodb",
    "username": "github",
    "handle": "github",
    "college": "university education",
    "school": "university education",
    "attend": "education university",
    "study": "education university",
    "degree": "education",
    "prize": "award",
    "certificate": "certification",
    "certified": "certification",
    "video": "meet",
    "microcontroller": "arduino esp32",
    "board": "arduino esp32",
    "hardware": "arduino esp32",
    "frontend": "reactjs",
    "hallucination": "hallucination guardrails",
    "decomposition": "decouple",
    "split": "decouple",
}

# Attributes the twin must never assert unless the evidence explicitly contains
# them: compensation, business metrics, personal data, employment and degrees.
# A query term from this set that is absent from the KB triggers a refusal.
CLAIM_BEARING = frozenset(
    """salary compensation pay wage income bonus equity revenue profit funding valuation
    price cost budget user customer download star traffic mau dau monthly daily
    address phone birthday age married family religion password ssn passport
    company employer employ hire salary manager boss promotion fire
    phd master thesis gpa grade rank ranking patent publication paper""".split()
)


def _stem(tok: str) -> str:
    if len(tok) <= 4:
        return tok
    if tok.endswith("ies"):
        return tok[:-3] + "y"
    if tok.endswith("ing") and len(tok) >= 7:
        stem = tok[:-3]
        if len(stem) > 3 and stem[-1] == stem[-2] and stem[-1] not in "lsz":
            stem = stem[:-1]  # cutting → cut, mapping → map
        return stem
    if tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    """Lowercase, accent-fold, split identifiers (``spring_boot`` → spring, boot, spring_boot)."""
    text = strip_accents(text.lower())
    out: list[str] = []
    for raw in re.findall(r"[a-z0-9]+(?:[_\-][a-z0-9]+)*", text):
        parts = re.split(r"[_\-]", raw)
        if len(parts) > 1:
            out.append(raw.replace("-", "_"))
        out.extend(_stem(p) for p in parts if p)
    return out


def content_terms(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS and len(t) > 1]


# --------------------------------------------------------------------------- embedders


class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic dense embedder via signed feature hashing (no model download)."""

    def __init__(self, dim: int = 1024, ngram_range: tuple[int, int] = (3, 5)):
        self.dim = dim
        self.ngram_range = ngram_range

    def _features(self, text: str) -> Counter:
        feats: Counter = Counter()
        for tok in content_terms(text):
            feats[f"w:{tok}"] += 2.0
            padded = f"<{tok}>"
            lo, hi = self.ngram_range
            for n in range(lo, hi + 1):
                for i in range(max(0, len(padded) - n + 1)):
                    feats[f"c:{padded[i : i + n]}"] += 1.0
        return feats

    def encode(self, texts: list[str]) -> np.ndarray:
        mat = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, count in self._features(text).items():
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
                sign = 1.0 if (h >> 63) & 1 else -1.0
                mat[row, h % self.dim] += sign * (1.0 + math.log(count))
            norm = np.linalg.norm(mat[row])
            if norm:
                mat[row] /= norm
        return mat


class SentenceTransformerEmbedder:  # pragma: no cover - optional heavy dependency
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)


# --------------------------------------------------------------------------- rerankers


class Reranker(Protocol):
    def score(self, query: str, cards: list[EvidenceCard]) -> list[float]: ...


class LexicalCrossReranker:
    """Joint (query, document) scorer emulating a cross-encoder with interpretable features."""

    def __init__(self, idf: dict[str, float]):
        self.idf = idf
        self.default_idf = max(idf.values(), default=1.0)

    def coverage(self, query: str, card: EvidenceCard) -> float:
        # Glossary keys are represented by their expansions, so they are not scored themselves.
        q = set(content_terms(query)) - GLOSSARY.keys()
        if not q:
            return 0.0
        doc = set(tokenize(card.text))
        total = sum(self.idf.get(t, self.default_idf) for t in q)
        hit = sum(self.idf.get(t, self.default_idf) for t in q if t in doc)
        return hit / total if total else 0.0

    def score(self, query: str, cards: list[EvidenceCard]) -> list[float]:
        q_terms = content_terms(query)
        q_bigrams = set(zip(q_terms, q_terms[1:], strict=False))
        scores = []
        for card in cards:
            cov = self.coverage(query, card)
            d_terms = content_terms(card.text)
            d_bigrams = set(zip(d_terms, d_terms[1:], strict=False))
            phrase = len(q_bigrams & d_bigrams) / len(q_bigrams) if q_bigrams else 0.0
            meta_tokens = set(tokenize(" ".join(card.tags) + " " + card.title + " " + card.domain))
            meta = len(set(q_terms) & meta_tokens) / len(set(q_terms)) if q_terms else 0.0
            scores.append(0.6 * cov + 0.2 * phrase + 0.2 * meta)
        return scores


class BGECrossReranker:  # pragma: no cover - optional heavy dependency
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def score(self, query: str, cards: list[EvidenceCard]) -> list[float]:
        logits = self.model.predict([(query, c.text) for c in cards])
        return [float(1 / (1 + math.exp(-x))) for x in logits]


# --------------------------------------------------------------------------- engine


@dataclass
class RetrievedChunk:
    card: EvidenceCard
    score: float
    bm25_rank: int | None
    dense_rank: int | None
    rrf_score: float


@dataclass
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    refused: bool
    refusal_reason: str | None = None
    unsupported_terms: list[str] = field(default_factory=list)

    @property
    def card_ids(self) -> list[str]:
        return [c.card.card_id for c in self.chunks]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K) -> dict[int, float]:
    """Fuse several ranked lists of doc indices: score(d) = Σ 1 / (k + rank_i(d))."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return fused


class HybridSearchEngine:
    def __init__(
        self,
        cards: list[EvidenceCard],
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        top_k: int = TOP_K,
        relative_threshold: float = 0.6,
        min_support: float = 0.35,
    ):
        if not cards:
            raise ValueError("cannot build an index over zero cards")
        self.cards = cards
        self.top_k = top_k
        self.relative_threshold = relative_threshold
        self.min_support = min_support

        self._doc_tokens = [tokenize(c.text) for c in cards]
        self.bm25 = BM25Okapi(self._doc_tokens)
        self.vocab = {t for toks in self._doc_tokens for t in toks}
        n = len(cards)
        df = Counter(t for toks in self._doc_tokens for t in set(toks))
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

        self.embedder = embedder or _default_embedder()
        self._doc_vecs = self.embedder.encode([c.text for c in cards])
        self.reranker = reranker or _default_reranker(self.idf)
        self._coverage = LexicalCrossReranker(self.idf)

    # -- individual retrievers ------------------------------------------------
    def bm25_rank(self, query: str) -> list[int]:
        scores = self.bm25.get_scores(content_terms(query))
        return [i for i in np.argsort(-scores) if scores[i] > 0][:CANDIDATE_POOL]

    def dense_rank(self, query: str) -> list[int]:
        q = self.embedder.encode([query])[0]
        sims = self._doc_vecs @ q
        return [int(i) for i in np.argsort(-sims) if sims[i] > 0][:CANDIDATE_POOL]

    # -- guardrail helpers ----------------------------------------------------
    @staticmethod
    def expand(query: str) -> str:
        extra = [GLOSSARY[t] for t in tokenize(query) if t in GLOSSARY]
        return f"{query} {' '.join(extra)}" if extra else query

    def unsupported_terms(self, query: str) -> list[str]:
        """Content terms of ``query`` that the knowledge base never mentions (after expansion)."""
        return [
            t
            for t in dict.fromkeys(content_terms(query))
            if t not in self.vocab and t not in QUESTION_VOCAB and t not in GLOSSARY
        ]

    def unknown_entities(self, query: str) -> list[str]:
        """Proper nouns / identifiers in the query that never occur in the knowledge base.

        A capitalised word (other than the first), an ALL-CAPS acronym, or a token
        containing digits is treated as a named entity or identifier.
        """
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9_\-]*", strip_accents(query))
        out = []
        for i, w in enumerate(words):
            entity_like = (i > 0 and w[0].isupper()) or any(ch.isdigit() for ch in w) or (w.isupper() and len(w) > 1)
            if not entity_like:
                continue
            toks = [t for t in tokenize(w) if t not in STOPWORDS]
            if toks and not any(t in self.vocab or t in GLOSSARY for t in toks):
                out.append(w)
        return out

    def guardrail(self, query: str, chunks: list[RetrievedChunk]) -> tuple[bool, str | None, list[str]]:
        unsupported = self.unsupported_terms(query)
        claims = [t for t in unsupported if t in CLAIM_BEARING]
        if claims:
            return True, f"asks for unverified attributes: {', '.join(claims)}", unsupported
        entities = self.unknown_entities(query)
        if entities:
            return True, f"unknown entities: {', '.join(entities)}", unsupported
        content = content_terms(query)
        if content and len(unsupported) / len(set(content)) > 0.5:
            return True, f"most query terms are absent from the knowledge base: {', '.join(unsupported)}", unsupported
        expanded = self.expand(query)
        support = max((self._coverage.coverage(expanded, c.card) for c in chunks), default=0.0)
        if support < self.min_support:
            return True, f"best evidence covers only {support:.0%} of the query", unsupported
        return False, None, unsupported

    # -- full pipeline ---------------------------------------------------------
    def search(self, query: str) -> RetrievalResult:
        original, query = query, self.expand(query)
        sparse, dense = self.bm25_rank(query), self.dense_rank(query)
        fused = reciprocal_rank_fusion([sparse, dense])
        pool = sorted(fused, key=fused.get, reverse=True)[:CANDIDATE_POOL]
        if not pool:
            return RetrievalResult(original, [], True, "no lexical or semantic match", self.unsupported_terms(original))

        rerank_scores = self.reranker.score(query, [self.cards[i] for i in pool])
        order = sorted(range(len(pool)), key=lambda j: rerank_scores[j], reverse=True)
        best = rerank_scores[order[0]]
        chunks = [
            RetrievedChunk(
                card=self.cards[pool[j]],
                score=rerank_scores[j],
                bm25_rank=sparse.index(pool[j]) + 1 if pool[j] in sparse else None,
                dense_rank=dense.index(pool[j]) + 1 if pool[j] in dense else None,
                rrf_score=fused[pool[j]],
            )
            for j in order[: self.top_k]
            if best > 0 and rerank_scores[j] >= self.relative_threshold * best
        ]

        refused, reason, unsupported = self.guardrail(original, chunks)
        return RetrievalResult(original, chunks, refused, reason, unsupported)


def _default_embedder() -> Embedder:
    if os.getenv("DT_EMBEDDER") == "sentence-transformers":  # pragma: no cover
        return SentenceTransformerEmbedder()
    return HashingEmbedder()


def _default_reranker(idf: dict[str, float]) -> Reranker:
    if os.getenv("DT_RERANKER") == "bge":  # pragma: no cover
        return BGECrossReranker()
    return LexicalCrossReranker(idf)
