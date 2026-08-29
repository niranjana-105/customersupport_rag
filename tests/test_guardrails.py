"""
Unit tests for input and output guardrail scanner nodes.

llm_guard is pre-mocked at the sys.modules level via conftest.py so
ONNX models never load. We patch the scan_prompt/scan_output calls
directly to control return values per test.
"""

from unittest.mock import patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_scan_result(scanner_name: str, is_valid: bool):
    """Return a (sanitized_prompt, results_valid, results_score) tuple."""
    return ("sanitized", {scanner_name: is_valid}, {scanner_name: 1.0})


# ---------------------------------------------------------------------------
# Input guardrail tests — question_check_node
# ---------------------------------------------------------------------------

class TestScanPromptInjection:
    """Tests for scan_prompt_injection node."""

    def test_safe_input_returns_status_zero(self):
        with patch(
            "src.graph.question_check_node.scan_prompt",
            return_value=_make_scan_result("PromptInjection", True),
        ):
            from src.graph.question_check_node import scan_prompt_injection
            result = scan_prompt_injection({"question": "How do I cancel my order?"})
            assert result == {"question_status": [0]}

    def test_injection_attempt_returns_status_one(self):
        with patch(
            "src.graph.question_check_node.scan_prompt",
            return_value=_make_scan_result("PromptInjection", False),
        ):
            from src.graph.question_check_node import scan_prompt_injection
            result = scan_prompt_injection({"question": "Ignore previous instructions..."})
            assert result == {"question_status": [1]}


class TestScanToxicity:
    """Tests for scan_toxicity node."""

    def test_clean_input_returns_status_zero(self):
        with patch(
            "src.graph.question_check_node.scan_prompt",
            return_value=_make_scan_result("Toxicity", True),
        ):
            from src.graph.question_check_node import scan_toxicity
            result = scan_toxicity({"question": "What is your return policy?"})
            assert result == {"question_status": [0]}

    def test_toxic_input_returns_status_one(self):
        with patch(
            "src.graph.question_check_node.scan_prompt",
            return_value=_make_scan_result("Toxicity", False),
        ):
            from src.graph.question_check_node import scan_toxicity
            result = scan_toxicity({"question": "This is a hateful message."})
            assert result == {"question_status": [1]}


class TestScanTokenLimit:
    """Tests for scan_token_limit node."""

    def test_short_input_within_limit_returns_zero(self):
        with patch(
            "src.graph.question_check_node.scan_prompt",
            return_value=_make_scan_result("TokenLimit", True),
        ):
            from src.graph.question_check_node import scan_token_limit
            result = scan_token_limit({"question": "How do I track my order?"})
            assert result == {"question_status": [0]}

    def test_over_limit_input_returns_one(self):
        with patch(
            "src.graph.question_check_node.scan_prompt",
            return_value=_make_scan_result("TokenLimit", False),
        ):
            from src.graph.question_check_node import scan_token_limit
            result = scan_token_limit({"question": "x " * 300})
            assert result == {"question_status": [1]}


class TestQuestionCheckNode:
    """Tests for the aggregating question_check_node."""

    def test_all_pass_returns_valid_true(self):
        from src.graph.question_check_node import question_check_node
        state = {"question": "How do I cancel?", "question_status": [0, 0, 0]}
        result = question_check_node(state)
        assert result["question_valid"] is True
        assert result["question"] == "How do I cancel?"

    def test_any_fail_returns_valid_false(self):
        from src.graph.question_check_node import question_check_node
        state = {"question": "Inject this prompt", "question_status": [1, 0, 0]}
        result = question_check_node(state)
        assert result["question_valid"] is False
        assert "llm_output" in result

    def test_all_fail_returns_valid_false(self):
        from src.graph.question_check_node import question_check_node
        state = {"question": "Bad question", "question_status": [1, 1, 1]}
        result = question_check_node(state)
        assert result["question_valid"] is False


# ---------------------------------------------------------------------------
# Output guardrail tests — answer_check_node
# ---------------------------------------------------------------------------

class TestCheckLanguageSame:
    """Tests for check_language_same node."""

    def test_same_language_returns_zero(self):
        with patch(
            "src.graph.answer_check_node.scan_output",
            return_value=_make_scan_result("LanguageSame", True),
        ):
            from src.graph.answer_check_node import check_language_same
            result = check_language_same({
                "llm_output": "Your order has been cancelled.",
                "prompt": "Cancel my order.",
            })
            assert result == {"answer_status": [0]}

    def test_different_language_returns_one(self):
        with patch(
            "src.graph.answer_check_node.scan_output",
            return_value=_make_scan_result("LanguageSame", False),
        ):
            from src.graph.answer_check_node import check_language_same
            result = check_language_same({
                "llm_output": "Votre commande a été annulée.",
                "prompt": "Cancel my order.",
            })
            assert result == {"answer_status": [1]}


class TestCheckRelevance:
    """Tests for check_relevance node."""

    def test_relevant_answer_returns_zero(self):
        with patch(
            "src.graph.answer_check_node.scan_output",
            return_value=_make_scan_result("Relevance", True),
        ):
            from src.graph.answer_check_node import check_relevance
            result = check_relevance({
                "llm_output": "You can cancel within 24 hours.",
                "prompt": "How do I cancel my order?",
            })
            assert result == {"answer_status": [0]}

    def test_irrelevant_answer_returns_one(self):
        with patch(
            "src.graph.answer_check_node.scan_output",
            return_value=_make_scan_result("Relevance", False),
        ):
            from src.graph.answer_check_node import check_relevance
            result = check_relevance({
                "llm_output": "The capital of France is Paris.",
                "prompt": "How do I cancel my order?",
            })
            assert result == {"answer_status": [1]}


class TestCheckSentiment:
    """Tests for check_sentiment node."""

    def test_positive_sentiment_returns_zero(self):
        with patch(
            "src.graph.answer_check_node.scan_output",
            return_value=_make_scan_result("Sentiment", True),
        ):
            from src.graph.answer_check_node import check_sentiment
            result = check_sentiment({
                "llm_output": "Happy to help you today!",
                "prompt": "Hello",
            })
            assert result == {"answer_status": [0]}

    def test_negative_sentiment_returns_one(self):
        with patch(
            "src.graph.answer_check_node.scan_output",
            return_value=_make_scan_result("Sentiment", False),
        ):
            from src.graph.answer_check_node import check_sentiment
            result = check_sentiment({
                "llm_output": "I hate this question.",
                "prompt": "Hello",
            })
            assert result == {"answer_status": [1]}


class TestAnswerCheckNode:
    """Tests for the aggregating answer_check_node."""

    def test_all_pass_returns_valid_true_with_output(self):
        from src.graph.answer_check_node import answer_check_node
        state = {"llm_output": "Here is your answer.", "answer_status": [0, 0, 0]}
        result = answer_check_node(state)
        assert result["answer_valid"] is True
        assert result["llm_output"] == "Here is your answer."

    def test_any_fail_returns_valid_false(self):
        from src.graph.answer_check_node import answer_check_node
        state = {"llm_output": "Problematic answer.", "answer_status": [1, 0, 0]}
        result = answer_check_node(state)
        assert result["answer_valid"] is False
        assert "llm_output" in result

    def test_answer_valid_is_in_returned_dict(self):
        """Regression: answer_valid must be in returned dict, not mutated on state."""
        from src.graph.answer_check_node import answer_check_node
        state = {"llm_output": "Here is your answer.", "answer_status": [0, 0, 0]}
        result = answer_check_node(state)
        assert "answer_valid" in result, (
            "answer_valid must be returned in the dict for LangGraph state propagation"
        )
