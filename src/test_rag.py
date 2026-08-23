from pathlib import Path

from data_processing import load_knowledge_base, create_chunks
from embeddings import EmbeddingModel
from retrieval import Retriever
from generator import AnswerGenerator


# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Knowledge base path
file_path = BASE_DIR / "data" / "knowledge_base" / "bnpl_faq.txt"


# 1. Load the knowledge base
text = load_knowledge_base(file_path)

# 2. Create text chunks
chunks = create_chunks(text)

print(f"Number of chunks: {len(chunks)}")


# 3. Create embedding model
embedding_model = EmbeddingModel()


# 4. Build the retrieval system
retriever = Retriever(embedding_model)
retriever.build_index(chunks)


# 5. Create the language model
generator = AnswerGenerator()


# 6. Ask a question
question = "What happens if I miss a payment?"


# 7. Retrieve relevant information
results = retriever.search(question, top_k=3)

context = "\n\n".join(
    result["chunk"] for result in results
)


# 8. Generate an answer using the retrieved context
answer = generator.generate(
    question=question,
    context=context
)


# 9. Display the result
print("\nUser Question:")
print(question)

print("\nRetrieved Context:")
print("=" * 50)
print(context)

print("\nAI Answer:")
print("=" * 50)
print(answer)
