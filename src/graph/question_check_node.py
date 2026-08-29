"""
LLM Input Safety Check Module

Pre-instantiates ONNX scanners once at module load to avoid reloading
neural models on every request (which causes multi-second latency spikes).
"""

from typing import Any

import torch
import torch._inductor.config
from llm_guard import scan_prompt
from llm_guard.input_scanners import PromptInjection, TokenLimit, Toxicity

from src.graph.state import AgentState

torch.set_float32_matmul_precision("high")
torch._inductor.config.fx_graph_cache = True

# Pre-instantiate scanners at module level — ONNX models are loaded once
prompt_injection_scanner = PromptInjection(use_onnx=True)
toxicity_scanner = Toxicity(use_onnx=True)
token_limit_scanner = TokenLimit(limit=200)


def scan_prompt_injection(state: AgentState) -> dict[str, Any]:
    """
    Scan the input question for prompt injection.
    Returns status 0 (safe) or 1 (violation).
    """
    question = state["question"]
    _, results_valid, _ = scan_prompt([prompt_injection_scanner], question)
    is_safe = results_valid.get("PromptInjection", True)
    return {"question_status": [0 if is_safe else 1]}


def scan_toxicity(state: AgentState) -> dict[str, Any]:
    """
    Scan the input question for toxicity.
    Returns status 0 (safe) or 1 (violation).
    """
    question = state["question"]
    _, results_valid, _ = scan_prompt([toxicity_scanner], question)
    is_safe = results_valid.get("Toxicity", True)
    return {"question_status": [0 if is_safe else 1]}


def scan_token_limit(state: AgentState) -> dict[str, Any]:
    """
    Scan the token limit of the input question.
    Returns status 0 (within limit) or 1 (exceeded).
    """
    question = state["question"]
    _, results_valid, _ = scan_prompt([token_limit_scanner], question)
    is_within_limit = results_valid.get("TokenLimit", True)
    return {"question_status": [0 if is_within_limit else 1]}


def question_check_node(state: AgentState) -> dict[str, Any]:
    """
    Aggregate scanner results from the last 3 status entries.
    All must be 0 (safe) for the question to be valid.
    """
    question_status = state["question_status"]
    all_checks_passed = all(status == 0 for status in question_status[-3:])
    if all_checks_passed:
        return {"question": state["question"], "question_valid": True}
    return {
        "llm_output": "Your question could not be processed due to safety checks. Please rephrase and try again.",
        "question_valid": False,
    }


if __name__ == "__main__":
    state = {"question": "What is the capital of France?"}

    print(scan_prompt_injection(state))
    print(scan_toxicity(state))
    print(scan_token_limit(state))
