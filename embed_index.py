from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np

class FaissRetriever:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatIP(self.dim)
        self.docs = []

    def add_chunks(self, chunks):
        """Add text chunks into the FAISS index."""
        if not chunks:  # safety check
            print("⚠️ No chunks provided to add_chunks()")
            return
        
        # Extract texts
        texts = [c["text"] for c in chunks if c.get("text")]
        if not texts:  # safety: chunks without text
            print("⚠️ No valid text found in chunks")
            return

        # Encode with SentenceTransformer
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )

        # Ensure it's 2D and float32
        embeddings = np.array(embeddings, dtype="float32")
        if embeddings.ndim == 1:  # case: single chunk only
            embeddings = embeddings.reshape(1, -1)

        # Add to FAISS index
        self.index.add(embeddings)

        # Save original chunks for later retrieval
        self.docs.extend(chunks)

        print(f"✅ Added {len(chunks)} chunks to FAISS index")

    def retrieve(self, query, top_k=5):
        q_emb = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        
        D, I = self.index.search(q_emb, top_k)
        results = []
        for idx in I[0]:
            if idx < len(self.docs):
                results.append(self.docs[idx])
        return results

    def save(self, faiss_path="outputs/faiss.index", docs_path="outputs/docs.pkl"):
        faiss.write_index(self.index, faiss_path)
        with open(docs_path, "wb") as f:
            pickle.dump(self.docs, f)

    def load(self, faiss_path="outputs/faiss.index", docs_path="outputs/docs.pkl"):
        self.index = faiss.read_index(faiss_path)
        with open(docs_path, "rb") as f:
            self.docs = pickle.load(f)
