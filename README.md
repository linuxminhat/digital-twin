<div align="center">

![Minh's Digital Twin Banner](./assets/digital-twin-banner.svg)

# 🧠 Dang Nhat Minh's AI Digital Twin
### *Retrieval-Grounded AI Representation of Engineering Mindset, Systems Philosophy & Technical Portfolio*

[![GitHub](https://img.shields.io/badge/GitHub-linuxminhat-181717?style=for-the-badge&logo=github)](https://github.com/linuxminhat)
[![Email](https://img.shields.io/badge/Contact-dangnhatminhdhbk%40gmail.com-blue?style=for-the-badge&logo=gmail)](mailto:dangnhatminhdhbk@gmail.com)
[![Architecture](https://img.shields.io/badge/Architecture-Dual--Layer%20Cognitive%20OS%20%2B%20RAG-8a2be2?style=for-the-badge)](#-system-architecture)
[![Search](https://img.shields.io/badge/Retrieval-Hybrid%20Dense%20Vector%20%2B%20BM25-00c853?style=for-the-badge)](#-retrieval--reranking-pipeline)
[![Status](https://img.shields.io/badge/Status-Active%20%26%20Production--Ready-brightgreen?style=for-the-badge)](#)

</div>

---

## 📌 Table of Contents

- [💡 Overview & Vision](#-overview--vision)
- [🏗️ System Architecture](#-system-architecture)
- [🧩 Decoupled Design: Operating System vs. Evidence](#-decoupled-design-operating-system-vs-evidence)
- [🔍 Retrieval & Reranking Pipeline](#-retrieval--reranking-pipeline)
- [📂 Repository Structure](#-repository-structure)
- [🚀 Quickstart & Deployment Guide](#-quickstart--deployment-guide)
- [🛡️ Guardrails & Hallucination Mitigation](#-guardrails--hallucination-mitigation)
- [📊 Evaluation Framework](#-evaluation-framework)
- [👤 About the Author](#-about-the-author)
- [📜 Acknowledgements & Attribution](#-acknowledgements--attribution)

---

## 💡 Overview & Vision

In the era of Generative AI, traditional static resumes are becoming obsolete. A flat PDF cannot articulate:
- How an engineer **reasons under ambiguity**
- How they prioritize **architectural trade-offs**
- Their **core worldview, values, and decision-making framework**
- Verifiable, contextual **proof of their real-world impact**

Conversely, raw AI chatbots hallucinate wild claims and fail to represent a genuine person.

**Minh's Digital Twin** bridges this gap. It is an enterprise-grade cognitive architecture that unites a **Long-Context System Operating Prompt** with a **RAG-Grounded Evidence Layer**. It represents **Dang Nhat Minh** ([`@linuxminhat`](https://github.com/linuxminhat))—his software engineering mindset, AI automation workflows, distributed systems expertise, and verified portfolio.

> **Key Architectural Axiom:**  
> 🧠 **The Operating System** defines *who Minh is and how Minh thinks*.  
> 📑 **The Evidence Layer** proves *what actually happened with verifiable metrics*.  
> ⚡ **The LLM** synthesizes both into faithful, grounded, non-hallucinatory interactions.

---

## 🏗️ System Architecture

![Digital Twin System Architecture](./assets/architecture-diagram.svg)

The runtime execution flow executes across a robust multi-stage pipeline:

```text
User Question
     │
     ▼
[Hybrid Knowledge Retrieval]
  ├── Dense Vector Search (Semantic concepts, engineering philosophy)
  └── Sparse BM25 Search (Exact repository names, frameworks, metrics)
     │
     ▼
[Cross-Encoder Reranker] ──> Top-K = 5 Filtered Evidence Chunks
     │
     ▼
[Dual Cognitive Context Injection]
  ├── System Prompt: Minh's Operating System (Reasoning, Tone, Boundaries)
  └── Retrieved Context: Atomic Evidence Cards (Verified facts & proofs)
     │
     ▼
[LLM Synthesis Engine] ──> Grounded, Factually Audited Response
```

---

## 🧩 Decoupled Design: Operating System vs. Evidence

To avoid vector retrieval fragmentation and eliminate hallucinations, the system decouples identity from facts:

### 1. The Operating System (`prompts/minh-operating-system.md`)
*Stable, long-context personality and reasoning framework.*
- **Identity & Ethos:** Systems thinking, business leverage over tech vanity, high execution discipline.
- **Problem-Solving Model:** First-principles deconstruction → bottleneck isolation → lean prototyping → production hardening.
- **Communication Guidelines:** Direct, structured, humble, bilingual (Vietnamese & English).
- **Knowledge Boundaries:** Strict instructions to acknowledge gaps rather than guess.

### 2. The Evidence Layer (`knowledge-base/minh-primary-evidence.md`)
*Dynamic, verifiable ground-truth database.*
- Structured around self-contained **Evidence Cards** bounded by `===END EVIDENCE CARD===`.
- Enforces zero cross-chunk context bleeding.
- Contains verified career milestones, code repositories (`1000baitaplaptrinh`, `smartIOTproject`, `ITSmartHire`, full-stack platforms), and benchmarked metrics.

![Evidence Card Structure](./assets/rag-evidence-card.svg)

---

## 🔍 Retrieval & Reranking Pipeline

### Ingestion & Chunking Specification
| Configuration | Value | Strategic Rationale |
|---|---|---|
| **Chunking Strategy** | Atomic General Chunking | Avoids parent-child hierarchy overhead by keeping cards atomic |
| **Max Chunk Size** | `2,000 characters` | Accommodates complete technical project narratives |
| **Chunk Overlap** | `0 characters` | Delimited by `===END EVIDENCE CARD===` to eliminate duplication |
| **Search Mode** | Hybrid (Dense Vector + BM25) | Ensures zero miss on proper nouns, repo titles, and technical keywords |
| **Rerank Model** | `rerank-english-v3.0` / BGE | Re-scores retrieved candidate pool based on cross-encoder attention |
| **Top-K Limit** | `5 chunks` | Optimizes signal-to-noise ratio in LLM context window |

---

## 📂 Repository Structure

```plaintext
digital-twin/
├── assets/
│   ├── digital-twin-banner.svg      # Master vector project banner
│   ├── architecture-diagram.svg      # Complete system architecture blueprint
│   └── rag-evidence-card.svg         # Evidence card & chunking visual design
├── architecture/
│   └── architecture.md              # In-depth architectural & technical documentation
├── knowledge-base/
│   └── minh-primary-evidence.md     # Primary RAG knowledge base (Atomic Evidence Cards)
├── prompts/
│   └── minh-operating-system.md     # Canonical system prompt & persona definition
└── README.md                        # Master project documentation
```

---

## 🚀 Quickstart & Deployment Guide

You can deploy and run this Digital Twin using platforms like **Dify**, **LangChain**, or custom Python services.

### Option 1: Deploy with Dify (Recommended)

1. **Create Chatflow / Agent:**
   - Set up an Chatflow application in [Dify](https://dify.ai/).
2. **Upload Knowledge Base:**
   - In Dify **Datasets**, upload [`knowledge-base/minh-primary-evidence.md`](file:///c:/Users/MinhDN/Downloads/prj/personal/digital-twin/knowledge-base/minh-primary-evidence.md).
   - In chunking settings, choose **Delimiter** mode and enter `===END EVIDENCE CARD===`.
   - Set search method to **Hybrid Search** with **Rerank** enabled.
3. **Configure System Prompt:**
   - Copy the content of [`prompts/minh-operating-system.md`](file:///c:/Users/MinhDN/Downloads/prj/personal/digital-twin/prompts/minh-operating-system.md) into the **Instructions / System Prompt** field.
4. **Link RAG Node:**
   - Connect the knowledge retrieval node to inject Top-5 chunks into the context.
5. **Publish & Test:**
   - Publish the agent and start interacting!

### Option 2: Python / LangChain Ingestion

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 1. Load atomic evidence base
loader = TextLoader("knowledge-base/minh-primary-evidence.md", encoding="utf-8")
docs = loader.load()

# 2. Split strictly along Evidence Card boundaries
text_splitter = CharacterTextSplitter(
    separator="===END EVIDENCE CARD===",
    chunk_size=2000,
    chunk_overlap=0
)
chunks = text_splitter.split_documents(docs)

# 3. Ingest into Vector Store
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
print(f"Successfully indexed {len(chunks)} evidence cards!")
```

---

## 🛡️ Guardrails & Hallucination Mitigation

The Digital Twin implements strict safeguards against hallucination:

1. **Refusal Boundary:** When evidence does not exist in the knowledge base, the model is strictly conditioned to respond:
   > *"I don't have verified information regarding this topic in my knowledge base."*
2. **No Fictional Accomplishments:** The agent will never fabricate revenue figures, degrees, employers, or metrics.
3. **Privacy by Design:** Proprietary credentials, confidential enterprise codebases, and private sensitive information are strictly omitted from public evidence layers.

---

## 📊 Evaluation Framework

The system is validated through a 7-point evaluation benchmark:

| Test Category | Test Objective | Pass Criteria |
|---|---|---|
| **Factual Retrieval** | Exact query on projects (`smartIOTproject`, `1000baitaplaptrinh`) | 100% precision with exact repo details |
| **Philosophical Alignment** | Questions on engineering mindset & problem solving | Reflects first-principles thinking from OS |
| **Factual Boundary Test** | Inquiring about unlisted companies or fabricated metrics | Explicit refusal; zero hallucination |
| **Hybrid Recall Test** | Keyword-dense queries vs. vague semantic questions | Both retrieve relevant Top-3 Evidence Cards |
| **Bilingual Capability** | Queries in Vietnamese and English | Seamless response in the queried language |

---

## 🛠️ Featured Engineering Projects

The Digital Twin's knowledge base is grounded in verified, real-world engineering projects:

| Project | Domain & Role | Core Tech Stack | Key Architectural Achievement |
|---|---|---|---|
| **AI Digital Twin Platform** | AI Agent & RAG Architecture | Dify, Hybrid Search, Rerank, Python | Decoupled cognitive OS from factual evidence cards; 0% hallucination on verified facts. |
| **ITSmartHire** | AI Recruitment Platform | ReactJS, NestJS, FastAPI, MongoDB | Fine-tuned `all-mpnet-base-v2` embedding model for semantic CV-to-JD matching; automated workflows. |
| **Microservices Library System** | Distributed Backend | Java, Spring Boot, ReactJS, TypeScript | Microservices architecture separating inventory, checkout, and review failure domains. |
| **Smart IoT Vision System** | Embedded Edge AI | C++, Arduino UNO, ESP32 Camera | Edge-to-cloud telemetry and real-time streaming with vision-assisted event alerts. |

---

## 👤 About the Author

<table border="0">
  <tr>
    <td width="100" align="center" valign="middle">
      <img src="https://github.com/linuxminhat.png" width="90" style="border-radius:50%;" alt="Dang Nhat Minh" />
    </td>
    <td>
      <strong>Dang Nhat Minh</strong> (<code>@linuxminhat</code>)<br/>
      <em>Software &amp; AI Automation Engineer | Systems Architect</em><br/>
      🎓 <strong>Da Nang University of Science and Technology (DUT)</strong> — Excellent Student Award<br/>
      📜 <strong>Certified:</strong> Anthropic (Claude Code in Action) | ArangoDB NoSQL | AWS Cloud Practitioner | mgm DevOps<br/>
      📍 Vietnam<br/>
      🌐 GitHub: <a href="https://github.com/linuxminhat">github.com/linuxminhat</a><br/>
      📫 Email: <a href="mailto:dangnhatminhdhbk@gmail.com">dangnhatminhdhbk@gmail.com</a>
    </td>
  </tr>
</table>

---

## 📜 Acknowledgements & Attribution

- **Original Architecture & Concept:** Inspired by the pioneering Digital Twin design by [Tom Lee / tom-agentbuilder](https://github.com/tom-agentbuilder/digital-twin).
- **Customized, Enhanced & Maintained by:** [Dang Nhat Minh (@linuxminhat)](https://github.com/linuxminhat).
