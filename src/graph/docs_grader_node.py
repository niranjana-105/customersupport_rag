from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from src.config import settings
from src.graph.state import AgentState


def _format_doc(doc) -> str:
    """
    Format a document for LLM consumption.
    Handles both plain string documents and dictionary metadata objects
    (as returned by FAISS retriever_node).
    """
    if isinstance(doc, dict):
        question = doc.get("question", "")
        answer = doc.get("answer", "")
        return f"Question: {question}\nAnswer: {answer}"
    return str(doc)


def retrieval_grader(doc, question: str, local_llm: bool = False) -> str:
    """Return 'yes' or 'no' based on document relevance to the question."""
    system = """You are a grader assessing relevance of a retrieved document to a user question.
        If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
        Respond with ONLY the single word "yes" or "no". Do not add any explanation."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Retrieved document: \n\n {document} \n\n User question: {question}",
            ),
        ]
    )

    formatted_doc = _format_doc(doc)

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
    result = chain.invoke({"question": question, "document": formatted_doc})
    return result.strip().lower()



def grade_documents_node(state: AgentState):
    """Grade retrieved documents and filter to relevant ones only."""
    docs = state["documents"]
    question = state["question"]
    filtered_docs = []
    for doc in docs:
        grade = retrieval_grader(doc, question, local_llm=settings.USE_LOCAL_LLM)
        if "yes" in grade.lower():
            filtered_docs.append(doc)
    if not filtered_docs:
        filtered_docs = docs  # fallback: use all docs if none pass grading
    return {"documents": filtered_docs}