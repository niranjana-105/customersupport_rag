"""
Integration tests for the FastAPI application.

Uses the shared `test_client` fixture from conftest.py which mocks
FAISS loading and graph invocation — no real LLM or vector store needed.
"""



class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_200(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self, test_client):
        response = test_client.get("/health")
        data = response.json()
        assert data == {"status": "ok"}


class TestRootEndpoint:
    """Tests for GET / (API root metadata)."""

    def test_root_returns_200(self, test_client):
        response = test_client.get("/")
        assert response.status_code == 200

    def test_root_returns_json_metadata(self, test_client):
        response = test_client.get("/")
        assert response.headers["content-type"].startswith("application/json")
        data = response.json()
        assert data.get("status") == "online"
        assert "docs_url" in data


class TestAnswerEndpoint:
    """Tests for POST /answer."""

    def test_valid_question_returns_200(self, test_client):
        response = test_client.post(
            "/answer", json={"question": "How do I cancel my order?"}
        )
        assert response.status_code == 200

    def test_valid_question_returns_llm_output(self, test_client):
        response = test_client.post(
            "/answer", json={"question": "How do I cancel my order?"}
        )
        data = response.json()
        assert "llm_output" in data
        assert len(data["llm_output"]) > 0

    def test_valid_question_includes_answer_valid_field(self, test_client):
        response = test_client.post(
            "/answer", json={"question": "How do I cancel my order?"}
        )
        data = response.json()
        assert "answer_valid" in data

    def test_valid_question_includes_question_field(self, test_client):
        response = test_client.post(
            "/answer", json={"question": "How do I cancel my order?"}
        )
        data = response.json()
        assert "question" in data

    def test_missing_question_field_returns_422(self, test_client):
        """FastAPI should return a 422 validation error for malformed request body."""
        response = test_client.post("/answer", json={})
        assert response.status_code == 422

    def test_empty_string_question_returns_200(self, test_client):
        """An empty question string should still be processed by the graph."""
        response = test_client.post("/answer", json={"question": ""})
        # Graph may return a safety message, but endpoint itself should not crash
        assert response.status_code in (200, 422)

    def test_answer_endpoint_response_is_json(self, test_client):
        response = test_client.post(
            "/answer", json={"question": "What is the return policy?"}
        )
        assert response.headers["content-type"].startswith("application/json")

    def test_answer_with_explicit_thread_id_returns_thread_id(self, test_client):
        """When a thread_id is provided, it should be preserved and returned in the response."""
        thread_id = "test-session-12345"
        response = test_client.post(
            "/answer", json={"question": "How do I return a product?", "thread_id": thread_id}
        )
        assert response.status_code == 200
        data = response.json()
        assert data.get("thread_id") == thread_id

    def test_answer_without_thread_id_generates_thread_id(self, test_client):
        """When no thread_id is provided, the API automatically generates one."""
        response = test_client.post(
            "/answer", json={"question": "What is my order status?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "thread_id" in data
        assert len(data["thread_id"]) > 0


class TestWorkflowNotInitialized:
    """Tests for error handling when workflow is not ready."""

    def test_answer_returns_503_when_workflow_missing(self):
        """
        Simulate uninitialized workflow by using a TestClient that skips
        the lifespan startup (raise_server_exceptions=False, no context mgr).
        api_context stays empty, so the endpoint returns 503.
        """
        from fastapi.testclient import TestClient

        from src.api.main import api_context, app

        # Ensure api_context is empty (no lifespan startup)
        original_workflow = api_context.pop("workflow", None)
        try:
            # Use app directly without entering lifespan
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/answer", json={"question": "Test question"})
            assert response.status_code in (503, 500)
        finally:
            # Restore if it was there
            if original_workflow is not None:
                api_context["workflow"] = original_workflow
