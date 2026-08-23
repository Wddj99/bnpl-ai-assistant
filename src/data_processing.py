from pathlib import Path


def load_knowledge_base(file_path):
    """
    Load the BNPL knowledge base from a text file.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Knowledge base not found: {file_path}")

    return path.read_text(encoding="utf-8")


def create_chunks(text):
    """
    Split the knowledge base into sections using topic headings.
    """
    sections = text.split("\n\n")
    
    chunks = []

    for section in sections:
        section = section.strip()

        if section:
            chunks.append(section)

    return chunks