import csv
import sys
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.append(str(SRC_DIR))

from data_processing import load_knowledge_base, create_chunks
from embeddings import EmbeddingModel
from retrieval import Retriever
from generator import AnswerGenerator


# 1. Load knowledge base
knowledge_base_path = (
    PROJECT_ROOT
    / "data"
    / "knowledge_base"
    / "bnpl_faq.txt"
)

text = load_knowledge_base(knowledge_base_path)

# 2. Create chunks
chunks = create_chunks(text)

# 3. Create embedding model
embedding_model = EmbeddingModel()

# 4. Build retrieval system
retriever = Retriever(embedding_model)
retriever.build_index(chunks)

# 5. Create answer generator
generator = AnswerGenerator()


# 6. Load evaluation questions
questions_path = PROJECT_ROOT / "evaluation" / "test_questions.csv"

with open(questions_path, newline="", encoding="utf-8") as file:
    questions = list(csv.DictReader(file))


retrieval_correct = 0
answer_correct = 0


# 7. Evaluate each question
for item in questions:

    question = item["question"]
    expected_topic = item["expected_topic"]
    expected_answer = item["expected_answer"]

    # Retrieve relevant chunks
    results = retriever.search(question, top_k=3)

    retrieved_text = " ".join(
        result["chunk"] for result in results
    )

    # Check retrieval
    retrieval_success = expected_topic.lower() in retrieved_text.lower()

    if retrieval_success:
        retrieval_correct += 1

    # Generate AI answer
    answer = generator.generate(
        question=question,
        context=retrieved_text
    )

    # Simple answer evaluation
    expected_keywords = [
        word.lower()
        for word in expected_answer.split()
        if len(word) > 4
    ]

    matched_keywords = [
        word
        for word in expected_keywords
        if word in answer.lower()
    ]

    answer_score = (
        len(matched_keywords) / len(expected_keywords)
        if expected_keywords
        else 0
    )

    answer_success = answer_score >= 0.30

    if answer_success:
        answer_correct += 1

    print("\n" + "-" * 60)
    print(f"Question: {question}")
    print(f"Expected topic: {expected_topic}")
    print(f"Retrieved correctly: {'YES' if retrieval_success else 'NO'}")
    print(f"AI Answer: {answer}")
    print(f"Answer score: {answer_score:.2f}")


# 8. Calculate metrics
retrieval_accuracy = (
    retrieval_correct / len(questions) * 100
)

answer_accuracy = (
    answer_correct / len(questions) * 100
)


# 9. Display results
print("\n" + "=" * 60)
print("FINAL EVALUATION RESULTS")
print("=" * 60)

print(
    f"Retrieval Accuracy: "
    f"{retrieval_accuracy:.1f}% "
    f"({retrieval_correct}/{len(questions)})"
)

print(
    f"Answer Quality Pass Rate: "
    f"{answer_accuracy:.1f}% "
    f"({answer_correct}/{len(questions)})"
)

print("\n" + "=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
print(f"Correct: {correct}/{len(questions)}")
print(f"Retrieval Accuracy: {accuracy:.1f}%")
