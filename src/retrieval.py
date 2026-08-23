import faiss
import numpy as np

from embeddings import EmbeddingModel


class Retriever:
    """
    Retrieve the most relevant knowledge base chunks
    using vector similarity search.
    """

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.index = None
        self.chunks = []

    def build_index(self, chunks):
        """
        Create a FAISS vector index from document chunks.
        """
        self.chunks = chunks

        embeddings = self.embedding_model.encode(chunks)
        embeddings = np.array(embeddings).astype("float32")

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query, top_k=3, threshold=1.0):
        """
        Retrieve the most relevant knowledge base chunks.

        If the closest result is above the threshold,
        the query is considered outside the knowledge base.
        """

        if self.index is None:
            raise ValueError("The index has not been built yet.")

        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")

        distances, indices = self.index.search(
            query_embedding,
            top_k
        )

        # Check whether the best result is relevant enough
        if distances[0][0] > threshold:
            return []

        results = []

        for distance, index in zip(distances[0], indices[0]):
            results.append({
                "chunk": self.chunks[index],
                "distance": float(distance)
            })

        return results