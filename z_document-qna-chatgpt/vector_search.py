import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class VectorSearch:
    """
    Handles document embedding storage and retrieval using ChromaDB.
    """

    def __init__(self):
        # Use persistent storage instead of in-memory
        self.client = chromadb.PersistentClient(path="chromadb_storage")
        self.collection = self.client.get_or_create_collection(name="document_chunks")

        # Load SentenceTransformer once
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Dictionary to track stored chunk IDs for each document
        self.doc_chunk_map: Dict[str, List[str]] = {}

    def add_documents(self, doc_id: str, chunks: List[str]):
        """
        Add document chunks as embeddings into ChromaDB.
        """
        if not chunks:
            return

        embeddings = self.model.encode(chunks)
        ids = [f"{doc_id}_{i}" for i in range(len(chunks))]

        # Store chunk IDs for later removal
        self.doc_chunk_map[doc_id] = ids

        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=chunks
        )

    def search(self, query: str, top_k: int = 10) -> List[str]:
        """
        Search for the most relevant chunks based on a query.
        """
        query_embedding = self.model.encode([query])
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        # print("result of search",results) # Debug logs
        # Extract and return retrieved documents
        return results['documents'][0] if results['documents'] else []

    def remove_document(self, doc_id: str):
        """
        Remove all stored embeddings related to a specific document ID.
        """
        if doc_id in self.doc_chunk_map:
            self.collection.delete(ids=self.doc_chunk_map[doc_id])
            del self.doc_chunk_map[doc_id]  # Remove reference
