import csv
import sys
from pathlib import Path

# Allow Python to import modules from the src folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from data_processing import load_knowledge_base, create_chunks
from embeddings import EmbeddingModel
from retrieval import Retriever


# 1. Load the knowledge base
knowledge_base_path = (
    PROJECT_ROOT
    / "data"
    / "knowledge_base"
    / "bnpl_faq.txt"
)

text = load_knowledge_base(knowledge_base_path)

# 2. Create chunks
chunks = create_chunks(text)

# 3. Create embeddings
embedding_model = EmbeddingModel()

# 4. Build retrieval index
retriever = Retriever(embedding_model)
retriever.build_index(chunks)


# 5. Load evaluation questions
questions_path = PROJECT_ROOT / "evaluation" / "test_questions.csv"

with open(questions_path, newline="", encoding="utf-8") as file:
    questions = list(csv.DictReader(file))


# 6. Evaluate retrieval
correct = 0

for item in questions:
    question = item["question"]
    expected_topic = item["expected_topic"]

    results = retriever.search(question, top_k=3)

    retrieved_text = " ".join(
        result["chunk"] for result in results
    ).upper()

    if expected_topic in retrieved_text:
        correct += 1

    print(f"\nQuestion: {question}")
    print(f"Expected topic: {expected_topic}")
    print(f"Retrieved: {'YES' if expected_topic in retrieved_text else 'NO'}")


# 7. Calculate accuracy
accuracy = correct / len(questions) * 100

print("\n" + "=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
print(f"Correct: {correct}/{len(questions)}")
print(f"Retrieval Accuracy: {accuracy:.1f}%")
