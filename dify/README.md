# Dify Chatflow

`digital_twin_chatflow.yml` is a Dify DSL (advanced-chat) export of the twin:

```
Start → Evidence Retrieval (Hybrid + Rerank, Top-5) → System Prompt (Minh OS) → LLM Generation → Citation Formatter → End (Answer)
```

## Import

1. **Knowledge Base** — Dify → Knowledge → Create → upload `knowledge-base/minh-primary-evidence.md`.
   - Chunk setting: *Custom*, delimiter `===END EVIDENCE CARD===`, max length 2000, overlap 0.
   - Index: *High Quality*; retrieval: *Hybrid Search* with *Rerank* enabled, Top-K 5.
2. **App** — Studio → Create App → *Import DSL file* → choose `digital_twin_chatflow.yml`.
3. **Bind** — open the *Evidence Retrieval* node and add the Knowledge Base from step 1
   (`dataset_ids` is intentionally empty in the export: dataset IDs are workspace-specific).
4. **Models** — the export references the Anthropic (`claude-sonnet-5-5`) and Cohere
   (`rerank-english-v3.0`) plugins. Install them, or pick any chat / rerank model you have.
5. **Publish** and test with the questions in `evaluation/benchmark_dataset.json`.

The *Citation Formatter* code node removes `[Exx]` citations that point at cards that were not
retrieved, then appends a **Sources** list — the same contract the local CLI enforces.

The *System Prompt (Minh OS)* node is a condensed copy of `prompts/minh-operating-system.md` with
added grounding/citation rules. Update both when the persona changes.
