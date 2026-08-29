"""
Unit tests for LangGraph node logic: topic classification, document grading,
context formatting, and graph compilation.

llm_guard/torch are pre-mocked via conftest.py sys.modules patching.
LLM calls are mocked per-test at the ChatGroq/ChatOllama level.
"""

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Topic classifier tests
# ---------------------------------------------------------------------------

class TestTopicClassifier:
    """Tests for topic_check_node.topic_classifier."""

    def _mock_grade_topic(self, score: str):
        return score

    def test_on_topic_question_returns_yes(self):
        with patch("src.graph.topic_check_node.classify_topic") as mock_classify:
            mock_classify.return_value = self._mock_grade_topic("Yes")
            from src.graph.topic_check_node import topic_classifier
            result = topic_classifier({"question": "How do I return a product?"})
            assert result == {"on_topic": "Yes"}

    def test_off_topic_question_returns_no_with_message(self):
        with patch("src.graph.topic_check_node.classify_topic") as mock_classify:
            mock_classify.return_value = self._mock_grade_topic("No")
            from src.graph.topic_check_node import topic_classifier
            result = topic_classifier({"question": "What is the capital of France?"})
            assert result["on_topic"] == "No"
            assert "llm_output" in result
            assert len(result["llm_output"]) > 0


# ---------------------------------------------------------------------------
# Document grader tests
# ---------------------------------------------------------------------------

class TestDocumentGrader:
    """Tests for docs_grader_node.grade_documents_node."""

    def test_relevant_docs_are_kept(self):
        with patch("src.graph.docs_grader_node.retrieval_grader", return_value="yes"):
            from src.graph.docs_grader_node import grade_documents_node
            docs = [
                {"question": "How do I cancel?", "answer": "Go to orders page."},
                {"question": "What is the refund policy?", "answer": "Within 30 days."},
            ]
            result = grade_documents_node({"documents": docs, "question": "Cancel order"})
            assert len(result["documents"]) == 2

    def test_fallback_returns_all_docs_when_none_pass(self):
        """Regression: if all docs are graded 'no', the original list is returned."""
        with patch("src.graph.docs_grader_node.retrieval_grader", return_value="no"):
            from src.graph.docs_grader_node import grade_documents_node
            docs = [
                {"question": "Unrelated Q1", "answer": "A1"},
                {"question": "Unrelated Q2", "answer": "A2"},
            ]
            result = grade_documents_node({"documents": docs, "question": "Cancel order"})
            assert result["documents"] == docs


class TestFormatDoc:
    """Tests for the _format_doc helper in docs_grader_node."""

    def test_dict_doc_formats_correctly(self):
        from src.graph.docs_grader_node import _format_doc
        doc = {"question": "How do I cancel?", "answer": "Go to orders page."}
        formatted = _format_doc(doc)
        assert "How do I cancel?" in formatted
        assert "Go to orders page." in formatted

    def test_string_doc_passes_through(self):
        from src.graph.docs_grader_node import _format_doc
        result = _format_doc("Some plain text document.")
        assert result == "Some plain text document."

    def test_dict_without_question_key_still_formats(self):
        from src.graph.docs_grader_node import _format_doc
        doc = {"answer": "Just an answer."}
        formatted = _format_doc(doc)
        assert "Just an answer." in formatted


# ---------------------------------------------------------------------------
# Answer node context formatting tests
# ---------------------------------------------------------------------------

class TestFormatContext:
    """Tests for the _format_context helper in answer_node."""

    def test_dict_documents_produce_reference_blocks(self):
        from src.graph.answer_node import _format_context
        docs = [
            {"question": "How do I cancel?", "answer": "Go to orders."},
            {"question": "How do I refund?", "answer": "Click refund button."},
        ]
        result = _format_context(docs)
        assert "[Reference 1]" in result
        assert "[Reference 2]" in result
        assert "How do I cancel?" in result
        assert "Click refund button." in result

    def test_string_documents_produce_reference_blocks(self):
        from src.graph.answer_node import _format_context
        docs = ["First document text.", "Second document text."]
        result = _format_context(docs)
        assert "[Reference 1]" in result
        assert "First document text." in result

    def test_empty_documents_returns_empty_string(self):
        from src.graph.answer_node import _format_context
        result = _format_context([])
        assert result == ""


# ---------------------------------------------------------------------------
# Answer node chat history formatting tests
# ---------------------------------------------------------------------------

class TestFormatChatHistory:
    """Tests for the _format_chat_history helper in answer_node."""

    def test_empty_chat_history_returns_no_prior_conversation(self):
        from src.graph.answer_node import _format_chat_history
        assert _format_chat_history([]) == "No prior conversation."
        assert _format_chat_history(None) == "No prior conversation."

    def test_formatted_turns_render_customer_and_assistant(self):
        from src.graph.answer_node import _format_chat_history
        history = [
            {"role": "user", "content": "I want to return an item."},
            {"role": "assistant", "content": "You can return it within 30 days."},
        ]
        result = _format_chat_history(history)
        assert "Customer: I want to return an item." in result
        assert "Assistant: You can return it within 30 days." in result


# ---------------------------------------------------------------------------
# Graph compilation test
# ---------------------------------------------------------------------------

class TestGraphCompilation:
    """Tests for graph.create_workflow compilation."""

    def test_graph_compiles_without_error(self, mock_retriever):
        """Verify the LangGraph workflow compiles correctly with a mock retriever."""
        from src.graph.graph import create_workflow
        graph = create_workflow(mock_retriever)
        assert graph is not None

    def test_graph_compiles_stateless_when_checkpointer_false(self, mock_retriever):
        """Verify the graph can compile statelessly when checkpointer=False."""
        from src.graph.graph import create_workflow
        graph = create_workflow(mock_retriever, checkpointer=False)
        assert graph is not None

    def test_graph_has_expected_nodes(self, mock_retriever):
        """Verify the compiled graph contains all expected node names."""
        from src.graph.graph import create_workflow
        graph = create_workflow(mock_retriever)
        node_names = set(graph.get_graph().nodes.keys())
        expected_nodes = {
            "scan_prompt_injection",
            "scan_toxicity",
            "scan_token_limit",
            "question_check_node",
            "topic_classifier",
            "retrieve_docs",
            "docs_grader",
            "generate_answer",
            "check_language_same",
            "check_relevance",
            "check_sentiment",
            "answer_check_node",
        }
        assert expected_nodes.issubset(node_names)
