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
# SIMPLE CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f8fafc;
    }

    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 16px;
        margin-bottom: 30px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">💳 BNPL AI Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Your smart assistant for payments, installments, refunds, and eligibility.'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# FEATURES
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("💰", "Payments")

with col2:
    st.metric("📅", "Installments")

with col3:
    st.metric("↩️", "Refunds")

with col4:
    st.metric("✓", "Eligibility")


st.divider()


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


# ============================================================
# INITIALIZE SYSTEM
# ============================================================

with st.spinner("Loading BNPL AI Assistant..."):

    try:

        retriever, generator = load_rag_system()

    except Exception as e:

        st.error("Unable to load the AI assistant.")

        st.exception(e)

        st.stop()


# ============================================================
# QUESTION
# ============================================================

st.subheader("Ask your question")

question = st.text_input(
    "Enter your question",
    placeholder="e.g. What happens if I miss a payment?",
)


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

            # ------------------------------------------------
            # Retrieve relevant context
            # ------------------------------------------------

            results = retriever.search(
                question,
                top_k=3,
                threshold=1.0,
            )

            # ------------------------------------------------
            # Guardrail
            # ------------------------------------------------

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


        # ====================================================
        # ANSWER
        # ====================================================

        with st.container(border=True):

            st.markdown("### 🤖 Answer")

            st.write(answer)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Powered by Retrieval-Augmented Generation (RAG) · BNPL AI Assistant"
)