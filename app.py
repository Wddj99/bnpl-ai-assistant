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
)


# ============================================================
# SIMPLE STYLING
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background-color: #f8fafc;
        }

        .block-container {
            max-width: 850px;
            padding-top: 3rem;
        }

        h1 {
            color: #111827;
            font-weight: 800;
        }

        .subtitle {
            color: #64748b;
            font-size: 1.05rem;
            margin-bottom: 2rem;
        }

        .answer-box {
            background-color: white;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 1.5rem;
            margin-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
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


# ============================================================
# HEADER
# ============================================================

st.title("💳 BNPL AI Assistant")

st.markdown(
    """
    <div class="subtitle">
    Your smart assistant for payments, installments,
    refunds, and eligibility.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# FEATURE CARDS
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
# LOAD MODEL
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
    "Your question",
    placeholder="e.g. What happens if I miss a payment?",
    label_visibility="collapsed",
)


# ============================================================
# ASK
# ============================================================

if st.button(
    "🤖 Ask BNPL Assistant",
    use_container_width=True
):

    if not question.strip():

        st.warning("Please enter a question first.")

    else:

        with st.spinner("Finding the best answer..."):

            results = retriever.search(
                question,
                top_k=3
            )

            retrieved_text = " ".join(
                result["chunk"]
                for result in results
            )

            answer = generator.generate(
                question=question,
                context=retrieved_text
            )

        # ====================================================
        # ANSWER
        # ====================================================

        st.subheader("🤖 Answer")

        st.info(answer)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.divider()

st.subheader("💡 Try asking")

examples = [
    "What happens if I miss a payment?",
    "How can I pay my installments?",
    "How many installments can I make?",
    "What happens when I receive a refund?",
]

for example in examples:
    st.write(f"• {example}")


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "Powered by Retrieval-Augmented Generation (RAG) · BNPL AI Assistant"
)