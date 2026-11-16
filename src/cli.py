"""Interactive CLI for the Digital Twin retrieval engine.

Examples::

    python -m src.cli --query "What did Minh build in ITSmartHire?"
    python -m src.cli --query "..." --debug     # show BM25 / dense / RRF / rerank scores
    python -m src.cli                           # interactive REPL

The CLI answers *extractively*: it selects the evidence lines that best match the
question and attaches card citations (``[E02]``). It never generates text that is
not present in the knowledge base. Generative synthesis is delegated to the LLM
node in the Dify chatflow (``dify/digital_twin_chatflow.yml``), which receives
the same Top-K evidence.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.chunker import load_cards
from src.hybrid_search import HybridSearchEngine, RetrievalResult, content_terms

DEFAULT_KB = Path(__file__).resolve().parent.parent / "knowledge-base" / "minh-primary-evidence.md"
REFUSAL = "I do not have verified evidence regarding that specific detail in my knowledge base."
MAX_LINES = 4

_BULLET_RE = re.compile(r"^\s*-\s+(.*)$", re.MULTILINE)


def _clean(line: str) -> str:
    return re.sub(r"\*\*|`|\[([^\]]+)\]\([^)]+\)", lambda m: m.group(1) or "", line).strip()


def compose_answer(result: RetrievalResult) -> str:
    """Build a cited, extractive answer from the retrieved evidence."""
    if result.refused or not result.chunks:
        return REFUSAL
    q = set(content_terms(HybridSearchEngine.expand(result.query)))
    scored: list[tuple[float, int, str, str]] = []
    for rank, chunk in enumerate(result.chunks):
        for line in _BULLET_RE.findall(chunk.card.text):
            overlap = len(q & set(content_terms(line)))
            if overlap:
                scored.append((overlap * chunk.score, -rank, _clean(line), chunk.card.card_id))
    if not scored:
        top = result.chunks[0].card
        return f"{top.title} [{top.card_id}]"
    scored.sort(reverse=True)
    lines, seen = [], set()
    for _, _, text, cid in scored:
        if text not in seen:
            seen.add(text)
            lines.append(f"- {text} [{cid}]")
        if len(lines) == MAX_LINES:
            break
    return "\n".join(lines)


def render(console: Console, result: RetrievalResult, latency_ms: float, debug: bool) -> None:
    style = "red" if result.refused else "green"
    console.print(Panel(compose_answer(result), title=f"Q: {result.query}", border_style=style))
    if result.refused:
        console.print(f"[dim]guardrail: {result.refusal_reason}[/dim]")
    if debug and result.chunks:
        t = Table("card", "title", "bm25 rank", "dense rank", "rrf", "rerank")
        for c in result.chunks:
            t.add_row(
                c.card.card_id,
                c.card.title[:48],
                str(c.bm25_rank or "-"),
                str(c.dense_rank or "-"),
                f"{c.rrf_score:.4f}",
                f"{c.score:.3f}",
            )
        console.print(t)
    console.print(f"[dim]latency: {latency_ms:.1f} ms[/dim]")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Query Minh's Digital Twin evidence base.")
    parser.add_argument("--query", "-q", help="question to ask; omit for interactive mode")
    parser.add_argument("--kb", default=str(DEFAULT_KB), help="path to the evidence-card markdown")
    parser.add_argument("--debug", action="store_true", help="print retrieval score breakdown")
    args = parser.parse_args(argv)

    console = Console()
    engine = HybridSearchEngine(load_cards(args.kb))

    def ask(q: str) -> None:
        t0 = time.perf_counter()
        res = engine.search(q)
        render(console, res, (time.perf_counter() - t0) * 1000, args.debug)

    if args.query:
        ask(args.query)
        return 0
    console.print("[bold]Minh's Digital Twin[/bold] — type a question, or 'exit'.")
    while True:
        try:
            q = console.input("[cyan]? [/cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            return 0
        if q.lower() in {"exit", "quit"}:
            return 0
        if q:
            ask(q)


if __name__ == "__main__":
    sys.exit(main())
