"""
Pytest shared fixtures for Customer Support Agentic RAG test suite.

IMPORTANT: llm_guard loads ONNX models at module-level instantiation in
question_check_node.py and answer_check_node.py. We patch llm_guard in
sys.modules BEFORE any src.graph module is collected/imported so ONNX
models never load during testing.

We do NOT mock torch — it is installed in the venv and works fine.
Mocking torch breaks transformers' importlib.util.find_spec("torch") call.
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# Patch llm_guard BEFORE any src.graph imports happen.
# This prevents ONNX model loading during test collection.
# ---------------------------------------------------------------------------

def _patch_llm_guard() -> None:
    """
    Inject fake llm_guard scanner modules into sys.modules before any import.
    Only patches if not already done (safe to call multiple times).
    """
    if "llm_guard.input_scanners" in sys.modules and isinstance(
        sys.modules["llm_guard.input_scanners"], MagicMock
    ):
        return  # Already patched

    def _make_mock_scanner(*args, **kwargs):
        return MagicMock()

    # Create mock llm_guard top-level
    mock_llm_guard = MagicMock()
    mock_llm_guard.scan_prompt.return_value = ("sanitized", {}, {})
    mock_llm_guard.scan_output.return_value = ("sanitized", {}, {})

    # Create mock input_scanners submodule
    mock_input_scanners = MagicMock()
    mock_input_scanners.PromptInjection = MagicMock(side_effect=_make_mock_scanner)
    mock_input_scanners.Toxicity = MagicMock(side_effect=_make_mock_scanner)
    mock_input_scanners.TokenLimit = MagicMock(side_effect=_make_mock_scanner)

    # Create mock output_scanners submodule
    mock_output_scanners = MagicMock()
    mock_output_scanners.LanguageSame = MagicMock(side_effect=_make_mock_scanner)
    mock_output_scanners.Relevance = MagicMock(side_effect=_make_mock_scanner)
    mock_output_scanners.Sentiment = MagicMock(side_effect=_make_mock_scanner)

    sys.modules["llm_guard"] = mock_llm_guard
    sys.modules["llm_guard.input_scanners"] = mock_input_scanners
    sys.modules["llm_guard.output_scanners"] = mock_output_scanners


# Patch at conftest import time — before pytest collects any test modules
_patch_llm_guard()


# ---------------------------------------------------------------------------
# Shared document fixtures
# ---------------------------------------------------------------------------

SAMPLE_DOCS_METADATA = [
    {
        "question": "How do I cancel my order?",
        "answer": "You can cancel your order within 24 hours by visiting the orders page.",
    },
    {
        "question": "How do I request a refund?",
        "answer": "To request a refund, go to your order history and click 'Request Refund'.",
    },
    {
        "question": "How do I track my shipment?",
        "answer": "You can track your shipment using the tracking number sent to your email.",
    },
]


@pytest.fixture
def sample_docs():
    """Return a list of LangChain Document objects with customer support metadata."""
    return [
        Document(page_content=d["question"], metadata=d)
        for d in SAMPLE_DOCS_METADATA
    ]


# ---------------------------------------------------------------------------
# Mock FAISS retriever
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_retriever(sample_docs):
    """
    Mock LangChain FAISS retriever that returns sample documents.
    Also exposes a mock vectorstore with docstore for evaluation tests.
    """
    retriever = MagicMock()
    retriever.invoke.return_value = sample_docs
    retriever.get_relevant_documents.return_value = sample_docs

    # Mock vectorstore.docstore._dict for evaluation pipeline
    docstore = MagicMock()
    docstore._dict = {str(i): doc for i, doc in enumerate(sample_docs)}
    retriever.vectorstore = MagicMock()
    retriever.vectorstore.docstore = docstore

    return retriever


# ---------------------------------------------------------------------------
# Mock LLM responses
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_groq_llm():
    """Mock Groq ChatGroq LLM that returns a canned answer."""
    mock = MagicMock()
    mock.invoke.return_value = MagicMock(
        content="This is a mock customer support answer."
    )
    mock.with_structured_output.return_value = mock
    return mock


# ---------------------------------------------------------------------------
# FastAPI TestClient (with mocked graph and FAISS)
# ---------------------------------------------------------------------------


@pytest.fixture
def test_client(mock_retriever):
    """
    FastAPI TestClient with FAISS loading and graph workflow fully mocked.
    The graph returns a canned final state so no LLM/FAISS calls happen.
    """
    mock_state = {
        "question": "How do I cancel my order?",
        "question_valid": True,
        "on_topic": "Yes",
        "llm_output": "You can cancel your order within 24 hours.",
        "documents": SAMPLE_DOCS_METADATA,
        "answer_valid": True,
        "question_status": [0, 0, 0],
        "answer_status": [0, 0, 0],
        "prompt": "Answer the question...",
        "chat_history": [
            {"role": "user", "content": "How do I cancel my order?"},
            {"role": "assistant", "content": "You can cancel your order within 24 hours."},
        ],
    }

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value=mock_state)
    mock_graph.invoke = MagicMock(return_value=mock_state)

    with (
        patch("src.graph.utils.load_faiss_index", return_value=mock_retriever),
        patch("src.graph.graph.create_workflow", return_value=mock_graph),
    ):
        from fastapi.testclient import TestClient

        from src.api.main import app

        with TestClient(app) as client:
            yield client
