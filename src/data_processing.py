from pathlib import Path


def load_knowledge_base(file_path):
    """
    Load the BNPL knowledge base from a text file.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Knowledge base not found: {file_path}")

    return path.read_text(encoding="utf-8")


def create_chunks(text, chunk_size=100):
    """
    Split the knowledge base into smaller text chunks.
    """
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    return chunks
