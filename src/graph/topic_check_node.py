from functools import lru_cache

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from src.config import settings
from src.graph.state import AgentState


@lru_cache(maxsize=100)
def classify_topic(question: str, local_llm: bool = False) -> str:
    """Return 'Yes' if the question is customer-support-related, else 'No'."""
    system = """You are a grader assessing whether a user's question is related to customer support
    for a product or a purchase.
    Customer support topics include:
    - Questions about purchasing products (e.g., "How do I place an order?")
    - Questions about order cancellations (e.g., "Can I cancel my order?")
    - Questions about refunds or returns (e.g., "How do I request a refund?")
    - Questions about product issues (e.g., "My product is not working.")
    - Questions about account issues (e.g., "I can't log in to my account.")

    Respond with ONLY the single word "Yes" or "No". Do not add any explanation.
    """

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: {question}"),
        ]
    )

    if local_llm:
        llm = ChatOllama(
            model=settings.OLLAMA_MODEL_NAME,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
    else:
        llm = ChatGroq(
            model=settings.GROQ_MODEL_NAME,
            api_key=settings.GROQ_API_KEY,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )

    chain = grade_prompt | llm | StrOutputParser()
    raw = chain.invoke({"question": question}).strip()
    # Robust parsing: take the first word and normalize
    first_word = raw.split()[0].strip(".,!?\"'").capitalize() if raw else "No"
    return first_word if first_word in ("Yes", "No") else "No"


def topic_classifier(state: AgentState):
    """Classify the topic of the question."""
    question = state["question"]
    score = classify_topic(question, local_llm=settings.USE_LOCAL_LLM)

    if score == "Yes":
        return {"on_topic": "Yes"}
    else:
        bot_reply = "Please ask a question about customer support so I can help you better."
        new_history = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": bot_reply},
        ]
        return {
            "on_topic": "No",
            "llm_output": bot_reply,
            "chat_history": new_history,
        }

