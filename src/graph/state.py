from operator import add
from typing import Annotated, Literal, TypedDict


class AgentState(TypedDict):
    """
    Graph state.

    Attributes:
    -----------
    question: str
    question_status: list  — Annotated with `add` so parallel nodes append entries
    question_valid: bool
    on_topic: Literal["Yes", "No", ""]  — string to match graph conditional routing
    prompt: str
    llm_output: str
    documents: List[str]
    answer_status: list  — Annotated with `add` so parallel nodes append entries
    answer_valid: bool

    """

    question: str
    question_status: Annotated[list, add]
    question_valid: bool
    on_topic: Literal["Yes", "No", ""]
    prompt: str
    llm_output: str
    documents: list[str]
    answer_status: Annotated[list, add]
    answer_valid: bool
    chat_history: Annotated[list, add]
