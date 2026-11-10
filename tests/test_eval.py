import json
from pathlib import Path

from evaluation.eval_runner import DEFAULT_DATASET, render_markdown, run

KB = Path(__file__).resolve().parent.parent / "knowledge-base" / "minh-primary-evidence.md"


def test_dataset_shape():
    data = json.loads(DEFAULT_DATASET.read_text(encoding="utf-8"))
    cases = data["cases"]
    assert len(cases) == 30
    assert len({c["id"] for c in cases}) == 30
    for c in cases:
        assert c["category"] in {"A", "B", "C"}
        assert c["expect_refusal"] == (c["category"] == "C")
        assert bool(c["relevant_cards"]) != c["expect_refusal"]


def test_benchmark_meets_quality_gate():
    summary, results = run()
    assert summary["recall_at_5"] >= 0.9
    assert summary["hallucination_rejection_rate"] == 1.0
    assert summary["false_refusal_rate"] <= 0.1
    report = render_markdown(summary, results, {})
    assert "Recall@5" in report and "| A01 |" in report
