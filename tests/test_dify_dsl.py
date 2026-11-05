from pathlib import Path

import yaml

DSL = Path(__file__).resolve().parent.parent / "dify" / "digital_twin_chatflow.yml"


def _load():
    return yaml.safe_load(DSL.read_text(encoding="utf-8"))


def test_dsl_is_advanced_chat_app():
    doc = _load()
    assert doc["kind"] == "app"
    assert doc["app"]["mode"] == "advanced-chat"


def test_graph_is_linear_pipeline_in_expected_order():
    graph = _load()["workflow"]["graph"]
    nodes = {n["id"]: n["data"]["type"] for n in graph["nodes"]}
    nxt = {e["source"]: e["target"] for e in graph["edges"]}
    order, cur = ["start"], "start"
    while cur in nxt:
        cur = nxt[cur]
        order.append(cur)
    assert [nodes[n] for n in order] == ["start", "knowledge-retrieval", "template-transform", "llm", "code", "answer"]
    for e in graph["edges"]:
        assert e["data"]["sourceType"] == nodes[e["source"]]
        assert e["data"]["targetType"] == nodes[e["target"]]


def test_retrieval_uses_rerank_top5():
    node = next(n for n in _load()["workflow"]["graph"]["nodes"] if n["id"] == "retrieval")
    cfg = node["data"]["multiple_retrieval_config"]
    assert cfg["top_k"] == 5 and cfg["reranking_enable"] is True


def test_citation_formatter_code_runs():
    node = next(n for n in _load()["workflow"]["graph"]["nodes"] if n["id"] == "citations")
    scope: dict = {}
    exec(node["data"]["code"], scope)
    evidence = [{"content": "## E02 — Project: ITSmartHire\n- body"}]
    out = scope["main"]("Built ITSmartHire [E02]. Also X [E09].", evidence)["result"]
    assert "[E09]" not in out
    assert "**Sources**\n- [E02] Project: ITSmartHire" in out
