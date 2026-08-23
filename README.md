# BNPL AI Assistant

An AI-powered customer support assistant for Buy Now, Pay Later (BNPL) services using Retrieval-Augmented Generation (RAG).

## Overview

BNPL AI Assistant retrieves relevant information from a dedicated knowledge base and uses an AI language model to generate concise customer-friendly answers.

The project demonstrates a complete RAG pipeline:

User Question → Embeddings → FAISS Retrieval → Relevant Context → AI Generation → Answer

## Features

- Retrieval-Augmented Generation (RAG)
- Semantic search using Sentence Transformers
- FAISS vector search
- AI answer generation using Google FLAN-T5
- BNPL FAQ knowledge base
- Automated retrieval and answer evaluation
- Streamlit web interface
- Simple and user-friendly UI

## Supported Topics

The current knowledge base covers:

- Payment methods
- Installment schedules
- Late payments
- Eligibility
- Refunds

## Project Structure

```text
bnpl-ai-assistant/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── knowledge_base/
│       └── bnpl_faq.txt
│
├── evaluation/
│   ├── evaluate.py
│   └── test_questions.csv
│
└── src/
    ├── data_processing.py
    ├── embeddings.py
    ├── retrieval.py
    ├── generator.py
    └── test_rag.py