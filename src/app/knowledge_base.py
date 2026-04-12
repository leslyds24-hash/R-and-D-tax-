"""
Vector database integration for IRS Section 41 knowledge base.

Enables RAG (Retrieval-Augmented Generation) for:
- IRS regulations and case law
- Technical precedents and examples

NOTE: ChromaDB is optional. If not installed, RAG features are disabled
gracefully and the system falls back to LLM-only classification.
"""
import os
from typing import List, Dict, Any, Optional

# Graceful ChromaDB import — optional dependency
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    chromadb = None  # type: ignore
    CHROMA_AVAILABLE = False

# Graceful OpenAI import for embeddings
try:
    from openai import OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    _embedding_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except ImportError:
    _embedding_client = None


class KnowledgeBase:
    """
    RAG knowledge base for IRS Section 41 compliance.
    Falls back gracefully when ChromaDB is not available.
    """

    def __init__(self, persist_directory: str = "./chroma_db"):
        self._available = CHROMA_AVAILABLE
        self.collection = None

        if not self._available:
            return

        try:
            self.chroma_client = chromadb.PersistentClient(path=persist_directory)
            self.collection = self.chroma_client.get_or_create_collection(
                name="irs_section_41_knowledge",
                metadata={"description": "IRS Section 41 regulations, case law, and precedents"}
            )
        except Exception:
            self._available = False

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a document to the knowledge base with embeddings."""
        if not self._available or not _embedding_client:
            return

        response = _embedding_client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding

        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}]
        )

    def search(
        self,
        query: str,
        top_k: int = 3,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[Dict, Any]]:
        """Semantic search over knowledge base."""
        if not self._available or not _embedding_client or not self.collection:
            return []

        response = _embedding_client.embeddings.create(
            input=query,
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata,
        )

        documents = []
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                documents.append({
                    "id": results['ids'][0][i],
                    "text": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results.get('distances') else 0.0,
                })

        return documents

    def load_irs_regulations(self) -> None:
        """Load IRS Section 41 regulations into knowledge base."""
        if not self._available:
            return

        regulations = [
            {
                "id": "irs-41-permitted-purpose",
                "text": """IRS Section 41(d)(1)(A) - Permitted Purpose

Qualified research must be undertaken for the purpose of discovering information
that is technological in nature and the application of which is intended to be
useful in the development of a new or improved business component.

Key requirements:
- Information must be technological in nature
- Must relate to new or improved business component
- Must be useful in development (not commercialization)
- Business component includes products, processes, software, techniques, formulas

Disqualifying purposes:
- Research after commercial production begins
- Adapting existing business components to particular customer requirements
- Duplicating existing business components
- Surveys, studies, or market research""",
                "metadata": {
                    "category": "regulation",
                    "section": "41(d)(1)(A)",
                    "axis": "permitted_purpose"
                }
            },
            {
                "id": "irs-41-uncertainty",
                "text": """IRS Section 41(d)(1)(B) - Elimination of Uncertainty

Activities must constitute elements of a process of experimentation relating to
elimination of uncertainty concerning the development or improvement of a business
component. The uncertainty must concern the capability, method, or appropriate design.

Technical uncertainty exists when:
- Information available does not establish capability or method
- Appropriate design is uncertain
- Multiple technical approaches need evaluation

NOT technical uncertainty:
- Cost or efficiency improvements without technical uncertainty
- Business or market viability questions
- Aesthetic or cosmetic design choices
- Routine testing or debugging""",
                "metadata": {
                    "category": "regulation",
                    "section": "41(d)(1)(B)",
                    "axis": "elimination_of_uncertainty"
                }
            },
            {
                "id": "irs-41-experimentation",
                "text": """IRS Section 41(d)(1)(C) - Process of Experimentation

Substantially all activities must constitute a process of experimentation for
a qualified purpose. This process fundamentally relies on principles of physical
or biological sciences, engineering, or computer science.

Evidence of experimentation includes:
- Systematic trial and error testing
- Evaluation of alternatives
- Modeling, simulation, or systematic testing
- Iterative refinement based on test results
- Documentation of failed attempts
- Comparative analysis of design alternatives

NOT a process of experimentation:
- Activities after uncertainty is eliminated
- Routine application of existing knowledge
- Quality control, efficiency surveys
- Management studies or consumer surveys
- Cosmetic or style changes""",
                "metadata": {
                    "category": "regulation",
                    "section": "41(d)(1)(C)",
                    "axis": "process_of_experimentation"
                }
            },
            {
                "id": "irs-41-technological",
                "text": """IRS Section 41(d)(1)(D) - Technological in Nature

Research must fundamentally rely on principles of physical or biological sciences,
engineering, or computer science.

Qualifying technical disciplines:
- Computer science and software engineering
- Mechanical, electrical, chemical engineering
- Physical sciences (physics, chemistry)
- Biological sciences
- Material sciences

Evidence of technological nature:
- Application of engineering principles
- Scientific method and hypothesis testing
- Technical documentation and specifications
- Use of scientific or engineering tools
- Involvement of technical personnel with appropriate degrees

NOT technological in nature:
- Social sciences or business management techniques
- Aesthetic or artistic improvements
- Market research or competitive analysis""",
                "metadata": {
                    "category": "regulation",
                    "section": "41(d)(1)(D)",
                    "axis": "technological_in_nature"
                }
            },
            {
                "id": "case-law-routine-work",
                "text": """IRS Guidance: Routine vs. Qualified Development

Courts and IRS have consistently held that routine, conventional, or ordinary
activities do NOT qualify, even if they have uncertainty:

Examples of ROUTINE work (non-qualifying):
- Adapting existing technology without significant modification
- Minor design improvements or upgrades
- Cosmetic changes or style improvements
- Efficiency improvements using known techniques
- Testing after development is complete
- Quality control or standard debugging
- Configuration of existing software packages

Examples of QUALIFIED work:
- Developing new algorithms or methods
- Novel applications of technology to new problems
- Significant improvements beyond state-of-the-art
- Systematic experimentation to overcome technical challenges
- Prototype development requiring technical innovation

The key distinction: Innovation vs. Application of known techniques.""",
                "metadata": {
                    "category": "case_law",
                    "risk_type": "routine_development"
                }
            },
        ]

        for reg in regulations:
            try:
                self.add_document(
                    doc_id=reg["id"],
                    text=reg["text"],
                    metadata=reg["metadata"]
                )
            except Exception as e:
                print(f"Warning: Failed to add document {reg['id']}: {e}")

    def get_relevant_context(
        self,
        project_description: str,
        top_k: int = 3,
    ) -> str:
        """
        Retrieve relevant IRS regulations/case law for a project.
        Returns formatted context string for LLM prompt augmentation.
        """
        if not self._available:
            return ""

        try:
            results = self.search(project_description, top_k=top_k)

            if not results:
                return ""

            context_parts = ["**Relevant IRS Section 41 Regulations:**\n"]
            for i, doc in enumerate(results, 1):
                context_parts.append(f"\n{i}. {doc['text'].strip()}")

            return "\n".join(context_parts)
        except Exception as e:
            print(f"Warning: RAG context retrieval failed: {e}")
            return ""


# Singleton instance
_kb_instance: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    """Get or create singleton knowledge base instance."""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = KnowledgeBase()
        try:
            if _kb_instance._available and _kb_instance.collection and _kb_instance.collection.count() == 0:
                print("Loading IRS Section 41 regulations into knowledge base...")
                _kb_instance.load_irs_regulations()
                print(f"Loaded {_kb_instance.collection.count()} regulations")
        except Exception as e:
            print(f"Warning: Failed to load IRS regulations: {e}")
    return _kb_instance


def augment_prompt_with_rag(
    project_description: str,
    base_prompt: str,
) -> str:
    """
    Augment LLM prompt with relevant context from knowledge base.
    """
    try:
        kb = get_knowledge_base()
        context = kb.get_relevant_context(project_description, top_k=3)

        if context:
            return f"{base_prompt}\n\n{context}\n\nProject to analyze: {project_description}"
        else:
            return base_prompt
    except Exception as e:
        print(f"RAG augmentation failed: {e}")
        return base_prompt
