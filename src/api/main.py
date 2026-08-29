"""
This module contains the FastAPI application that serves the RAG Graph API.
"""

import os
import warnings
from contextlib import asynccontextmanager

import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from src.graph.graph import create_workflow
from src.graph.utils import load_faiss_index

warnings.filterwarnings("ignore")


class Question(BaseModel):
    question: str
    thread_id: Optional[str] = None


api_context = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Async context manager to handle the lifespan events of the FastAPI application."""
    try:
        # Load the FAISS index
        faiss_index = load_faiss_index()
        # Create the workflow
        logger.info("Creating the workflow...")
        api_context["workflow"] = create_workflow(faiss_index)
        logger.info("Workflow created successfully.")
        yield
    except Exception:
        logger.exception("Failed to load FAISS index and create the workflow.")
        raise
    finally:
        api_context.pop("workflow", None)
        logger.info("Workflow cleaned up.")


app = FastAPI(title="Rag Graph API", version="0.1.0", lifespan=lifespan)


@app.get("/")
def read_root():
    """Root endpoint returning service status and documentation links."""
    return JSONResponse(
        content={
            "name": "Customer Support Agentic RAG API",
            "status": "online",
            "docs_url": "/docs",
            "health_url": "/health",
        }
    )


@app.post("/answer")
async def answer(question: Question):
    """
    Answer the question using the compiled LangGraph workflow with conversation memory.

    Uses async invocation (ainvoke) with thread_id configuration so conversation context
    is tracked across turns, while keeping the FastAPI event loop non-blocking.

    Args:
        question (Question): The user's question and optional thread_id.

    Returns:
        JSONResponse: The full graph state including llm_output, thread_id, chat_history, etc.
    """
    try:
        graph = api_context.get("workflow")
        if graph is None:
            raise HTTPException(
                status_code=503,
                detail="Service not ready. The workflow has not been initialized.",
            )
        thread_id = question.thread_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}
        state = await graph.ainvoke({"question": question.question}, config=config)
        state["thread_id"] = thread_id
        logger.info(f"Query answered successfully for thread {thread_id}: {question.question!r}")
        return JSONResponse(content=state)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"Failed to answer question: {question.question!r}")
        raise HTTPException(
            status_code=500,
            detail=f"An internal error occurred while processing your request: {exc}",
        )


@app.get("/health")
def health():
    return JSONResponse(content={"status": "ok"})
