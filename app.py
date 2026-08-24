import sys
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from data_processing import load_knowledge_base, create_chunks
from embeddings import EmbeddingModel
from retrieval import Retriever
from generator import AnswerGenerator


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="BNPL AI Assistant",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ============================================================
# LOAD RAG SYSTEM
# ============================================================

@st.cache_resource
def load_rag_system():

    knowledge_base_path = (
        PROJECT_ROOT
        / "data"
        / "knowledge_base"
        / "bnpl_faq.txt"
    )

    text = load_knowledge_base(knowledge_base_path)
    chunks = create_chunks(text)

    embedding_model = EmbeddingModel()

    retriever = Retriever(embedding_model)
    retriever.build_index(chunks)

    generator = AnswerGenerator()

    return retriever, generator


with st.spinner("Loading BNPL AI Assistant..."):

    try:
        retriever, generator = load_rag_system()

    except Exception as e:
        st.error("Unable to load the AI assistant.")
        st.exception(e)
        st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("💳 BNPL AI Assistant")

st.caption(
    "Your smart assistant for payments, installments, "
    "refunds, and eligibility."
)


# ============================================================
# TOPIC OVERVIEW
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("💰", "Payments")

with col2:
    st.metric("📅", "Installments")

with col3:
    st.metric("↩️", "Refunds")

with col4:
    st.metric("✅", "Eligibility")


st.divider()


# ============================================================
# SESSION STATE
# ============================================================

if "question" not in st.session_state:
    st.session_state.question = ""

if "answer" not in st.session_state:
    st.session_state.answer = None


# ============================================================
# SUGGESTED QUESTIONS
# ============================================================

st.subheader("💡 Try an example")

col1, col2 = st.columns(2)

with col1:

    if st.button(
        "Missed payment",
        use_container_width=True,
    ):
        st.session_state.question = (
            "What happens if I miss a payment?"
        )
        st.session_state.answer = None
        st.rerun()

with col2:

    if st.button(
        "Payment methods",
        use_container_width=True,
    ):
        st.session_state.question = (
            "How can I pay my installments?"
        )
        st.session_state.answer = None
        st.rerun()


col3, col4 = st.columns(2)

with col3:

    if st.button(
        "Installment schedule",
        use_container_width=True,
    ):
        st.session_state.question = (
            "How many installments can I make?"
        )
        st.session_state.answer = None
        st.rerun()

with col4:

    if st.button(
        "Eligibility",
        use_container_width=True,
    ):
        st.session_state.question = (
            "What factors affect installment eligibility?"
        )
        st.session_state.answer = None
        st.rerun()


col5, col6 = st.columns(2)

with col5:

    if st.button(
        "Refunds",
        use_container_width=True,
    ):
        st.session_state.question = (
            "How does a refund affect my installment plan?"
        )
        st.session_state.answer = None
        st.rerun()

with col6:

    if st.button(
        "Late payment",
        use_container_width=True,
    ):
        st.session_state.question = (
            "What happens if a scheduled payment is missed?"
        )
        st.session_state.answer = None
        st.rerun()


st.divider()


# ============================================================
# QUESTION INPUT
# ============================================================

st.subheader("Ask your question")

question = st.text_input(
    "Enter your question",
    value=st.session_state.question,
    placeholder="e.g. What happens if I miss a payment?",
)

st.session_state.question = question


# ============================================================
# ASK BUTTON
# ============================================================

if st.button(
    "🤖 Ask BNPL Assistant",
    use_container_width=True,
):

    if not question.strip():

        st.warning("Please enter a question first.")

    else:

        with st.spinner("Finding the best answer..."):

            results = retriever.search(
                question,
                top_k=3,
                threshold=1.0,
            )

            # Guardrail
            if not results:

                answer = (
                    "I don't have enough information in my "
                    "knowledge base to answer this question."
                )

            else:

                retrieved_text = " ".join(
                    result["chunk"]
                    for result in results
                )

                answer = generator.generate(
                    question=question,
                    context=retrieved_text,
                )

            st.session_state.answer = answer


# ============================================================
# ANSWER
# ============================================================

if st.session_state.answer:

    st.subheader("🤖 Answer")

    with st.container(border=True):
        st.write(st.session_state.answer)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Powered by Retrieval-Augmented Generation (RAG) "
    "· BNPL AI Assistant"
)