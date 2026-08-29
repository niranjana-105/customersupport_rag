from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from src.config import settings
from src.graph.state import AgentState


def _format_context(documents: list) -> str:
    """
    Format retrieved documents into clean, structured reference blocks.
    Handles both plain string documents and dictionary metadata objects.

    Example output:
        [Reference 1] Question: How do I cancel my order?
                      Answer: You can cancel your order within 24 hours...

        [Reference 2] Question: ...
    """
    formatted = []
    for i, doc in enumerate(documents, 1):
        if isinstance(doc, dict):
            question = doc.get("question", "")
            answer = doc.get("answer", "")
            formatted.append(
                f"[Reference {i}] Question: {question}\n             Answer: {answer}"
            )
        else:
            formatted.append(f"[Reference {i}] {doc}")
    return "\n\n".join(formatted)


def _format_chat_history(chat_history: list | None) -> str:
    """
    Format previous conversation turns into readable dialog context.
    Retains recent messages for context-aware follow-up resolution.
    """
    if not chat_history:
        return "No prior conversation."
    turns = []
    # Include up to the last 6 messages (3 full turns)
    for msg in chat_history[-6:]:
        if isinstance(msg, dict):
            role = "Customer" if msg.get("role") == "user" else "Assistant"
            content = msg.get("content", "")
            turns.append(f"{role}: {content}")
        else:
            turns.append(str(msg))
    return "\n".join(turns) if turns else "No prior conversation."


def generate_answer(
    question: str,
    context: list,
    chat_history: list | None = None,
    local_llm: bool = False,
):
    """
    Generate answer to the question based on context and conversation history.
    """
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

    template = """You are a helpful customer support assistant. Answer the customer's question based on the retrieved context and conversation history below.
If the customer is asking a follow-up question (e.g. using 'it', 'that', or referring to an earlier request), use the conversation history to maintain context.
If the context does not contain enough information, say so honestly.

IMPORTANT FORMATTING RULES:
- If the context contains placeholder text like {{Phone Number}}, {{Website URL}}, {{Order Number}}, or similar template variables, replace them with a generic description (e.g. "the phone number listed on our website", "our website", "your order number") or omit them if not needed.
- Write in clear, natural language. Use numbered steps only when the answer is a process.
- Do not invent specific contact details (phone numbers, emails, URLs) that are not in the context.

Conversation History:
{chat_history}

Retrieved Context:
{context}

Customer Question: {question}

Answer:"""

    prompt = ChatPromptTemplate.from_template(template=template)
    formatted_context = _format_context(context)
    formatted_history = _format_chat_history(chat_history)
    formatted_prompt = prompt.format(
        question=question,
        context=formatted_context,
        chat_history=formatted_history,
    )

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke(
        {
            "question": question,
            "context": formatted_context,
            "chat_history": formatted_history,
        }
    )
    return result, formatted_prompt


def answer_node(state: AgentState):
    """Generate answer node and record exchange in chat_history."""
    question = state["question"]
    context = state["documents"]
    chat_history = state.get("chat_history", [])
    answer, prompt = generate_answer(
        question,
        context,
        chat_history=chat_history,
        local_llm=settings.USE_LOCAL_LLM,
    )
    new_history = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]
    return {"llm_output": answer, "prompt": prompt, "chat_history": new_history}