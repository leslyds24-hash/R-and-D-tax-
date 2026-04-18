# R&D Tax Credit AI Platform: Product Roadmap & Analysis

## 🏛️ Project Architecture Analysis
The current codebase (MVP) demonstrates a high level of sophistication in AI-driven financial compliance. Key strengths include:
*   **Tiered Intelligence Cascade**: A cost-efficient "Rule-out → Heuristic → LLM" pipeline that optimizes for speed and accuracy.
*   **Agentic Multi-Step Workflow**: Uses `LangGraph` to separate concerns between eligibility assessment, expense classification, and narrative generation.
*   **Audit-Ready Infrastructure**: Built-in `ImmutableTraceLogger` and ZIP-based audit packages ensure all AI decisions are defensible.
*   **Expert-in-the-Loop (HITL)**: A confidence-based queue that flags borderline cases for manual review, a critical feature for high-stakes tax filing.

---

## 🚀 Roadmap to a Market-Ready Product

### 1. Enterprise Foundation (Hardening)
*   **Persistent Data Storage**: Migrate from in-memory caching to a relational database (**PostgreSQL** with SQLModel/SQLAlchemy) for multi-tenancy and data history.
*   **Authentication & RBAC**: Implement **OAuth2/JWT** with roles (Admin, Company Executive, Tax Reviewer, Auditor).
*   **Asynchronous Task Queue**: Use **Celery/Redis** to handle LLM calls and PDF generation in the background, keeping the API responsive for large project batches.

### 2. Deep Intelligence & "Evidence Engine"
*   **Document-Based RAG**: Allow users to upload PDFs (Project Charters, Test Plans) and use a vector database (**Pinecone/Chroma**) to extract verbatim "Four-Part Test" evidence.
*   **SDLC Integrations**: Develop connectors for **Jira**, **GitHub**, and **Linear** to automatically find evidence of "Technological Uncertainty" and "Process of Experimentation" in commit logs and tickets.
*   **Automated Conflict Detection**: Flag contradictions between internal project documents and the AI-generated tax claim to prevent IRS audit flags.

### 3. Modern UX/UI & Experience
*   **Bespoke Web Interface**: Move from Streamlit to a **Next.js/TS** frontend with dedicated dashboards for project tracking and evidence mapping.
*   **Interactive Narrative Editor**: A Google Docs-style editor for the IRS narrative where the AI provides real-time "Defensibility Scores" and suggests technical improvements for weak sections.
*   **Visual Audit Timeline**: An interactive timeline showing the project's evolution, highlighting key experimental phases and failures (the "Uncertainty Path").

### 4. Commercialization & Compliance
*   **Direct E-Filing Integration**: Connect with IRS-authorized software providers to populate and file **Form 6765** directly from the platform.
*   **Compliance & Security**: Pursue **SOC 2 Type II** and **ISO 27001** certification, as data privacy is the primary blocker for enterprise tax clients.
*   **Auditor "Safe-Room"**: A secure, read-only interface for outside auditors to review the "Trace Map" without accessing internal source code or private project repositories.

---

## 🛠️ Performance & Scalability Goals
*   **Sub-10s Batch Analysis**: Optimize the agentic pipeline to sweep through 100+ projects in under 10 seconds using parallel inference.
*   **99.9% Audit Pass Rate**: Achieve a benchmark where AI-justified projects withstand mock-IRS audits with minimal human rework.
*   **Cost Efficiency**: Maintain a "Cost-per-Review" under $0.50 by utilizing smaller local models (Qwen, Llama 3) for the heuristic and rule-out tiers.

---

## ✅ Immediate Next Steps
1.  **Schema Definition**: Design the PostgreSQL schema for Projects, Evidence, and User Roles.
2.  **RAG Prototype**: Implement a basic document ingestion tool to parse raw project documentation (PDF/DOCX).
3.  **Refined Dashboard**: Create high-fidelity mockups of the "Expert Review Queue" to visualize a premium SaaS experience.
