"""
Tests for centralized configuration settings.
"""

import os
from pathlib import Path
from unittest.mock import patch

from src.config import Settings, settings


class TestDefaultSettings:
    """Verify default values are correctly set."""

    def test_use_local_llm_defaults_to_false(self):
        assert settings.USE_LOCAL_LLM is False

    def test_groq_model_name_default(self):
        assert settings.GROQ_MODEL_NAME == "llama-3.3-70b-versatile" or settings.GROQ_MODEL_NAME == "openai/gpt-oss-20b"

    def test_ollama_model_name_default(self):
        assert settings.OLLAMA_MODEL_NAME == "llama3.2:3b"

    def test_embeddings_model_name_default(self):
        assert settings.EMBEDDINGS_MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2"

    def test_faiss_top_k_default(self):
        assert settings.FAISS_TOP_K == 5

    def test_llm_temperature_default(self):
        assert settings.LLM_TEMPERATURE == 0

    def test_llm_max_tokens_default(self):
        assert settings.LLM_MAX_TOKENS == 1024

    def test_debug_defaults_to_false(self):
        assert settings.DEBUG is False

    def test_evaluation_sample_size_default(self):
        assert settings.EVALUATION_SAMPLE_SIZE == 10

    def test_evaluation_random_seed_default(self):
        assert settings.EVALUATION_RANDOM_SEED == 123


class TestPathResolution:
    """Verify directory paths are resolved correctly relative to BASE_DIR."""

    def test_base_dir_is_path(self):
        assert isinstance(settings.BASE_DIR, Path)

    def test_data_dir_is_under_base(self):
        assert settings.DATA_DIR == settings.BASE_DIR / "data"

    def test_index_dir_is_under_data(self):
        assert settings.INDEX_DIR == settings.DATA_DIR / "indexes"

    def test_faiss_index_path_is_string(self):
        assert isinstance(settings.FAISS_INDEX_PATH, str)
        assert "faiss_index.faiss" in settings.FAISS_INDEX_PATH

    def test_evaluation_output_dir_contains_evaluation_results(self):
        assert "evaluation_results" in settings.EVALUATION_OUTPUT_DIR

    def test_logging_file_contains_preprocessing(self):
        assert "preprocessing.log" in settings.LOGGING_FILE


class TestEnvironmentVariableOverride:
    """Verify env vars correctly override defaults."""

    def test_use_local_llm_override(self):
        with patch.dict(os.environ, {"USE_LOCAL_LLM": "true"}):
            s = Settings()
            assert s.USE_LOCAL_LLM is True

    def test_groq_model_name_override(self):
        with patch.dict(os.environ, {"GROQ_MODEL_NAME": "custom-model"}):
            s = Settings()
            assert s.GROQ_MODEL_NAME == "custom-model"

    def test_debug_override(self):
        with patch.dict(os.environ, {"DEBUG": "true"}):
            s = Settings()
            assert s.DEBUG is True

    def test_faiss_top_k_override(self):
        with patch.dict(os.environ, {"FAISS_TOP_K": "10"}):
            s = Settings()
            assert s.FAISS_TOP_K == 10
