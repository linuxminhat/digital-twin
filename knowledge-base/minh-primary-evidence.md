# Dang Nhat Minh — Primary Evidence / RAG Knowledge Base — Canonical v2.1

## Retrieval Contract & Ingestion Guidelines

- **Unit of Retrieval:** Each `## E...` section represents a self-contained **Evidence Card**.
- **Context Preservation:** Ingestion pipelines MUST treat each card as an atomic retrieval unit separated by `===END EVIDENCE CARD===`.
- **Card Header Schema:** every card declares the following header fields as `**Field:** value` lines, parsed by `src/chunker.py`:
  - `Domain`: `profile` | `ai_automation` | `software_engineering` | `microservices` | `iot` | `engineering_philosophy`
  - `Source Type`: `resume_verified` | `github_portfolio` | `operating_system`
  - `Tags`: comma-separated retrieval keywords
  - `Confidence`: `high` | `medium` | `low`
- **Public Visibility:** every card in this file is approved for public retrieval.

---

## E00 — Canonical Identity, Education & Credentials

**Domain:** profile  
**Source Type:** resume_verified  
**Tags:** profile, identity, education, bach_khoa, dut, university, award, certifications  
**Source:** Verified Resume (Dang Nhat Minh)  
**Confidence:** High  

- **Full Name:** Dang Nhat Minh
- **GitHub:** [https://github.com/linuxminhat](https://github.com/linuxminhat)
- **Email:** dangnhatminhdhbk@gmail.com
- **Education:** Da Nang University of Science and Technology (DUT - Đại học Bách Khoa Đà Nẵng).
- **Academic Honors:** Excellent Student Award (09/2023 – 10/2024) for outstanding academic performance.
- **Key Industry Certifications:**
  - **Claude Code in Action** — Anthropic (01/2026)
  - **NoSQL Certification for Professionals** — ArangoDB (01/2025)
  - **AWS Cloud Practitioner Essentials** — AWS Skill Builder (08/2024)
  - **Fundamental DevOps Skills Training** — mgm technology partners (04/2024)

===END EVIDENCE CARD===

## E01 — Project: AI Digital Twin Platform (Dify & Hybrid RAG)

**Domain:** ai_automation  
**Source Type:** github_portfolio  
**Tags:** project, digital_twin, dify, rag, llm, hybrid_search, bm25, rerank, guardrails  
**Source:** Verified Personal Project (`linuxminhat/digital-twin`)  
**Timeline:** 10/2025 – Present  
**Tech Stack:** Dify (Chatflow / Workflow), Long-Context LLMs, Hybrid Search (Dense Vector + BM25), Cross-Encoder Reranker, Python, FastAPI, SVG Architecture Modeling.  
**Confidence:** High  

- **Architectural Paradigm:** Designed a decoupled cognitive agent separating the Long-Context Operating System (persona, reasoning patterns, guardrails) from the RAG Evidence Layer (verifiable facts, metrics).
- **Knowledge Engineering:** Formatted career and engineering data into self-contained **Evidence Cards** bounded by `===END EVIDENCE CARD===`, eliminating context loss and chunk fragmentation.
- **Retrieval Pipeline:** Configured Hybrid Search merging semantic dense vector retrieval with sparse BM25 keyword matching, followed by Cross-Encoder reranking (Top-K = 5) to guarantee precise factual recall.
- **Guardrails & Reliability:** Enforced zero-hallucination constraints and explicit knowledge boundaries, instructing the agent to refuse unsupported queries rather than speculate.
- **Outcome:** Produced an interactive, citation-backed digital twin serving as a live cognitive alternative to static resumes.

===END EVIDENCE CARD===

## E02 — Project: ITSmartHire — AI-Powered Recruitment Platform

**Domain:** software_engineering  
**Source Type:** resume_verified  
**Tags:** project, itsmarthire, recruitment, ats, embedding, fastapi, reactjs, nestjs, mongodb  
**Source:** Verified Resume Project (04/2025 – 07/2025)  
**Timeline:** 04/2025 – 07/2025  
**Tech Stack:** ReactJS, TypeScript, Node.js, NestJS, Python (FastAPI), MongoDB, Firebase (Notification, Authentication Service).  
**Confidence:** High  

- **Role-Based Architecture:** Developed a comprehensive recruitment platform with dedicated workflows for administrators, candidates, and HR staff.
- **Core Business Workflows:** Implemented end-to-end modules for job lifecycle management, applicant tracking (ATS), automated candidate screening, and interview scheduling.
- **Third-Party Integration:** Integrated Google Meet API for seamless remote interview scheduling.
- **AI / NLP Engineering:** Fine-tuned the `all-mpnet-base-v2` embedding model to automatically calculate semantic relevance scores between candidate resumes and job descriptions (JDs).
- **Integrated Ecosystem:** Connected two supporting sub-applications for AI-driven mock technical interviews and intelligent resume creation.
- **Outcome:** Delivered a unified, automated recruitment system that significantly reduced manual candidate screening overhead.

===END EVIDENCE CARD===

## E03 — Project: Microservices Library Management System

**Domain:** microservices  
**Source Type:** resume_verified  
**Tags:** project, microservices, library, java, spring_boot, reactjs, typescript  
**Source:** Verified Resume Project (08/2025 – 10/2025)  
**Timeline:** 08/2025 – 10/2025  
**Tech Stack:** ReactJS, TypeScript, Java, Spring Boot, Microservices Architecture.  
**Confidence:** High  

- **System Purpose:** Built a full-fledged web application to digitize library operations, book circulation, and inventory management.
- **Service Decomposition:** Applied microservices design principles to decouple independent business domains: inventory management, user circulation/checkout, inquiry processing, and community reviews.
- **Frontend & APIs:** Created high-performance backend REST services in Java Spring Boot paired with a responsive ReactJS/TypeScript web UI.
- **Outcome:** Enhanced system maintainability, decoupled failure domains, and improved horizontal scalability.

===END EVIDENCE CARD===

## E04 — Embedded AI & Smart IoT Systems

**Domain:** iot  
**Source Type:** github_portfolio  
**Tags:** project, iot, smartiotproject, esp32, arduino, camera, computer_vision, edge_ai  
**Source:** Repository: `smartIOTproject`  
**Confidence:** High  

- **Implementation:** Built an integrated IoT system utilizing Arduino UNO and ESP32 Camera hardware.
- **Capabilities:** Real-time camera streaming, sensor data telemetry, edge computing inference, and automated event notifications.
- **Outcome:** Proved capability to bridge physical embedded hardware with software backends and AI models.

===END EVIDENCE CARD===

## E05 — Engineering Operating Principles & Problem-Solving Framework

**Domain:** engineering_philosophy  
**Source Type:** operating_system  
**Tags:** mindset, philosophy, first_principles, trade_offs, problem_solving, architecture, decision_making  
**Source:** Canonical Operating System (`prompts/minh-operating-system.md`, Section 3)  
**Confidence:** High  

- **Operating Pattern:** Understand business context → model the data & system → isolate bottlenecks → prototype a lean MVP → benchmark & validate → harden for production.
- **First-Principles Thinking:** Deconstructs ambiguous problems to their foundational constraints before selecting tools or frameworks.
- **Pragmatic Technology Trade-offs:** Chooses battle-tested technology where reliability is paramount; selectively applies LLMs, RAG and agentic orchestration where they create outsized leverage.
- **Execution Discipline:** Prefers clean modular code, automated CI/CD pipelines and explicit observability over quick-and-dirty patches.
- **Communication Style:** Direct, structured, humble and evidence-oriented; bilingual (Vietnamese & English).

===END EVIDENCE CARD===
