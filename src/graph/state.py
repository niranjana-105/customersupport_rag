from operator import add
from typing import Annotated, List, TypedDict


class AgentState(TypedDict):
    """
    Graph state.

    Attributes:
    -----------
    question: str
    question_status: list
    question_valid: bool
    on_topic: bool
    prompt: str
    llm_output: str
    documents: List[str]
    answer_status: list
    answer_valid: bool

    """

    question: str
    question_status: Annotated[list, add]
    question_valid: bool
    on_topic: bool
    prompt: str
    llm_output: str
    documents: List[str]
    answer_status: Annotated[list, add]
    answer_valid: bool
