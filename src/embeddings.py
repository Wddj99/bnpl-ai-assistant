from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Generate semantic embeddings for text using Sentence Transformers.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        """
        Convert text into numerical embeddings.
        """
        return self.model.encode(texts)
