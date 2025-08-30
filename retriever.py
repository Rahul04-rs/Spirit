# backend/retriever.py

class Retriever:
    def __init__(self):
        self.chunks = []

    def add_chunks(self, chunks):
        """Add chunks to the retriever's storage."""
        self.chunks.extend(chunks)

    def retrieve(self, query, top_k=5):
        """
        Retrieve up to top_k chunks containing the query as a substring (case-insensitive).
        Simple keyword-based retrieval.
        """
        matches = [chunk for chunk in self.chunks if query.lower() in chunk["text"].lower()]
        return matches[:top_k]
