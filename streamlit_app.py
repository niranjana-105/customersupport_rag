"""
Streamlit Chat Application for Customer Support Agentic RAG.

Features:
- Multi-turn conversation with memory (supports follow-up questions)
- Live backend connectivity check
- Safety guardrail inspector
- Retrieved FAISS context document viewer
"""

import uuid
import httpx
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Customer Support AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean UI styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .badge-safe {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-fail {
        background-color: #FFEBEE;
        color: #C62828;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-info {
        background-color: #E3F2FD;
        color: #1565C0;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


def check_api_health() -> bool:
    """Check if the FastAPI backend is reachable."""
    try:
        r = httpx.get(f"{API_URL}/health", timeout=3.0)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=70)
    st.title("Settings & Status")

    is_healthy = check_api_health()
    if is_healthy:
        st.success("🟢 Backend Connected (Port 8000)")
    else:
        st.error("🔴 Backend Offline\nStart with: `uvicorn src.api.main:app --reload`")

    st.divider()

    st.subheader("Session Memory")
    st.caption(f"**Thread ID:** `{st.session_state.thread_id[:8]}...`")
    if st.button("➕ Start New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.divider()

    st.subheader("Inspector Controls")
    show_guardrails = st.checkbox("🛡️ Show Safety Guardrails", value=True)
    show_docs = st.checkbox("📚 Show Retrieved Context", value=True)

    st.divider()
    st.caption("🤖 **Customer Support Agentic RAG**")
    st.caption("Powered by LangGraph, FAISS, and LLM Guard.")


# --- Main Chat Area ---
st.markdown('<div class="main-header">🤖 Customer Support Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Ask questions about orders, returns, refunds, cancellations, and account issues. Follow-up questions are remembered!</div>',
    unsafe_allow_html=True,
)

# Display sample starter prompts if no messages yet
if not st.session_state.messages:
    st.info("💡 **Try asking:**\n- *How do I return a damaged product?*\n- *Can I cancel my order after placing it?*\n- *How can I track my shipment status?*")

# Render conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Display guardrail & document inspectors for assistant messages
        if msg["role"] == "assistant" and "metadata" in msg:
            meta = msg["metadata"]

            # Guardrails badges
            if show_guardrails:
                badges_html = []
                if meta.get("question_valid") is not None:
                    q_badge = '<span class="badge-safe">Input Safe: ✅</span>' if meta.get("question_valid") else '<span class="badge-fail">Input Blocked: ❌</span>'
                    badges_html.append(q_badge)
                if meta.get("on_topic"):
                    t_badge = f'<span class="badge-info">Topic: {meta.get("on_topic")}</span>'
                    badges_html.append(t_badge)
                if meta.get("answer_valid") is not None:
                    a_badge = '<span class="badge-safe">Output Verified: ✅</span>' if meta.get("answer_valid") else '<span class="badge-fail">Output Filtered: ⚠️</span>'
                    badges_html.append(a_badge)

                if badges_html:
                    st.markdown(" ".join(badges_html), unsafe_allow_html=True)

            # Retrieved documents expander
            if show_docs and meta.get("documents"):
                docs = meta["documents"]
                with st.expander(f"📚 Retrieved Context ({len(docs)} Reference Documents)", expanded=False):
                    for i, doc in enumerate(docs, 1):
                        if isinstance(doc, dict):
                            q = doc.get("question", "N/A")
                            a = doc.get("answer", "N/A")
                            st.markdown(f"**[Reference {i}]**")
                            st.markdown(f"**Q:** {q}")
                            st.markdown(f"**A:** {a}")
                        else:
                            st.markdown(f"**[Reference {i}]** {doc}")
                        if i < len(docs):
                            st.divider()


# Handle user chat input
if prompt := st.chat_input("Type your question here..."):
    # Render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Call FastAPI backend
    with st.chat_message("assistant"):
        with st.spinner("Analyzing query, retrieving documents, and verifying safety..."):
            try:
                response = httpx.post(
                    f"{API_URL}/answer",
                    json={"question": prompt, "thread_id": st.session_state.thread_id},
                    timeout=45.0,
                )

                if response.status_code == 200:
                    data = response.json()
                    bot_output = data.get("llm_output", "No answer generated.")
                    metadata = {
                        "question_valid": data.get("question_valid"),
                        "on_topic": data.get("on_topic"),
                        "answer_valid": data.get("answer_valid"),
                        "documents": data.get("documents", []),
                    }
                elif response.status_code == 503:
                    bot_output = "⚠️ **Service is starting up.** Vector index is loading, please try again in a moment."
                    metadata = {}
                else:
                    bot_output = f"⚠️ **Server error ({response.status_code}):** {response.text}"
                    metadata = {}

            except httpx.ConnectError:
                bot_output = "❌ **Could not connect to FastAPI backend.** Please make sure the server is running on `http://localhost:8000`."
                metadata = {}
            except Exception as e:
                bot_output = f"⚠️ **Unexpected error:** {e}"
                metadata = {}

            st.markdown(bot_output)

            # Show inspect badges
            if show_guardrails and metadata:
                badges_html = []
                if metadata.get("question_valid") is not None:
                    q_badge = '<span class="badge-safe">Input Safe: ✅</span>' if metadata.get("question_valid") else '<span class="badge-fail">Input Blocked: ❌</span>'
                    badges_html.append(q_badge)
                if metadata.get("on_topic"):
                    t_badge = f'<span class="badge-info">Topic: {metadata.get("on_topic")}</span>'
                    badges_html.append(t_badge)
                if metadata.get("answer_valid") is not None:
                    a_badge = '<span class="badge-safe">Output Verified: ✅</span>' if metadata.get("answer_valid") else '<span class="badge-fail">Output Filtered: ⚠️</span>'
                    badges_html.append(a_badge)

                if badges_html:
                    st.markdown(" ".join(badges_html), unsafe_allow_html=True)

            # Show retrieved docs
            if show_docs and metadata.get("documents"):
                docs = metadata["documents"]
                with st.expander(f"📚 Retrieved Context ({len(docs)} Reference Documents)", expanded=False):
                    for i, doc in enumerate(docs, 1):
                        if isinstance(doc, dict):
                            st.markdown(f"**[Reference {i}]**\n**Q:** {doc.get('question')}\n**A:** {doc.get('answer')}")
                        else:
                            st.markdown(f"**[Reference {i}]** {doc}")
                        if i < len(docs):
                            st.divider()

            st.session_state.messages.append({
                "role": "assistant",
                "content": bot_output,
                "metadata": metadata,
            })
