import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os

class EmbeddingFAISS:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print("Loading embedding model...")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.embeddings = None

    def generate_embeddings(self, texts):
        print("Generating embeddings...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        self.embeddings = np.array(embeddings)
        return self.embeddings

    def build_faiss_index(self):
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)
        print("FAISS index built successfully.")

    def add_documents(self, documents):
        self.documents = documents
        embeddings = self.generate_embeddings(documents)
        self.build_faiss_index()

    def search(self, query, top_k=5):
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding), top_k)

        results = []
        for i in indices[0]:
            results.append(self.documents[i])

        return results

    def save_index(self, path="vector_store"):
        if not os.path.exists(path):
            os.makedirs(path)

        faiss.write_index(self.index, f"{path}/faiss.index")
        with open(f"{path}/docs.pkl", "wb") as f:
            pickle.dump(self.documents, f)

    def load_index(self, path="vector_store"):
        self.index = faiss.read_index(f"{path}/faiss.index")
        with open(f"{path}/docs.pkl", "rb") as f:
            self.documents = pickle.load(f)


if __name__ == "__main__":
    docs = ["AI is artificial intelligence", "ML is subset of AI"]
    store = EmbeddingFAISS()
    store.add_documents(docs)
    print(store.search("What is AI?"))