import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set
import tiktoken
from text_extraction import extract_text_from_file
from vector_search import VectorSearch

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class Document:
    """
    Represents an uploaded document, storing chunks and metadata.
    """
    def __init__(self, filename: str, chunks: List[str]):
        self.filename = filename
        self.chunks = chunks
        self.last_updated = datetime.now()
        self.document_id = str(hash(f"{filename}{datetime.now().isoformat()}"))
        self.token_count = sum(len(chunk) for chunk in chunks)

class QAHistory:
    """
    Stores question-answer pairs for a session.
    """
    def __init__(self):
        self.qa_pairs: List[Dict] = []

    def add_qa(self, question: str, answer: str, document_ids: Set[str]):
        """
        Store a QA pair along with associated document IDs.
        """
        self.qa_pairs.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now(),
            "document_ids": document_ids
        })
        logger.info(f"Added QA entry: {question}")

class RfpSession:
    """
    Represents an active RFP session with its own document storage and vector search.
    """
    MAX_ALLOWED_TOKENS = 120000  # Keep a buffer below OpenAI's 128K limit

    def __init__(self, session_id: str = None):
        self.session_id = session_id
        self.documents: Dict[str, Document] = {}
        self.qa_history = QAHistory()
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.total_tokens = 0
        self.vector_search = VectorSearch()  # Initialize vector search for this session

    # def __init__(self):
    #     # self.session_id = session_id
    #     self.documents: Dict[str, Document] = {}
    #     self.qa_history = QAHistory()
    #     self.encoder = tiktoken.get_encoding("cl100k_base")
    #     self.total_tokens = 0
    #     self.vector_search = VectorSearch()  # Initialize vector search for this session

    def process_and_chunk_document(self, filename: str, content: bytes) -> List[str]:
        """
        Extract text from a document and chunk it into manageable sizes.
        """
        logger.info(f"Processing document: {filename}")

        temp_file_path = f"/tmp/{filename}"
        with open(temp_file_path, "wb") as f:
            f.write(content)

        try:
            text = extract_text_from_file(temp_file_path)
        finally:
            Path(temp_file_path).unlink(missing_ok=True)

        tokens = self.encoder.encode(text)
        chunk_size = 1000
        chunks = [self.encoder.decode(tokens[i:i + chunk_size]) for i in range(0, len(tokens), chunk_size)]
        return chunks

    def add_or_update_document(self, filename: str, content: bytes) -> bool:
        """
        Add or update a document and store its vector embeddings.
        """
        chunks = self.process_and_chunk_document(filename, content)
        is_update = filename in self.documents

        if is_update:
            old_doc_id = self.documents[filename].document_id
            self.vector_search.remove_document(old_doc_id)
            self.total_tokens -= self.documents[filename].token_count

        doc = Document(filename, chunks)
        self.documents[filename] = doc
        self.total_tokens += doc.token_count

        self.vector_search.add_documents(doc.document_id, chunks)
        logger.info(f"{'Updated' if is_update else 'Added'} document {filename}. Total tokens: {self.total_tokens}")
        return is_update

    


    def get_relevant_context(self, query: str, top_k: int = 25) -> str:
        """
        Retrieve relevant document chunks using vector search while ensuring we don't exceed OpenAI's token limit.
        
        """
        
        total_tokens = sum(doc.token_count for doc in self.documents.values())

        if total_tokens == 0:
            relevant_chunks = self.vector_search.search(query, top_k=top_k)
            while sum(len(self.encoder.encode(chunk)) for chunk in relevant_chunks) > self.MAX_ALLOWED_TOKENS and relevant_chunks:
                relevant_chunks.pop()
            return "\n\n".join(relevant_chunks)
        
        elif total_tokens < 8000:
            combined_text = "\n\n".join(["\n".join(doc.chunks) for doc in self.documents.values()])
            return combined_text[:self.MAX_ALLOWED_TOKENS]

        elif total_tokens < 20000:
            relevant_chunks = self.vector_search.search(query, top_k=top_k)
            while sum(len(self.encoder.encode(chunk)) for chunk in relevant_chunks) > self.MAX_ALLOWED_TOKENS and relevant_chunks:
                relevant_chunks.pop()
            return "\n\n".join(relevant_chunks)
        
        elif total_tokens > 20000:
            relevant_chunks = self.vector_search.search(query, top_k=top_k)
            while sum(len(self.encoder.encode(chunk)) for chunk in relevant_chunks) > self.MAX_ALLOWED_TOKENS and relevant_chunks:
                relevant_chunks.pop()
            return "\n\n".join(relevant_chunks)

        
        
        else:
            print("non")
            all_chunks = []
            token_count = 0
            for doc in self.documents.values():
                tokens = self.encoder.encode("\n".join(doc.chunks))
                for i in range(0, len(tokens), 10000):
                    chunk = self.encoder.decode(tokens[i:i + 10000])
                    chunk_tokens = len(self.encoder.encode(chunk))
                    if token_count + chunk_tokens > self.MAX_ALLOWED_TOKENS:
                        break
                    all_chunks.append(chunk)
                    token_count += chunk_tokens
            return "\n\n".join(all_chunks)

    def list_documents(self) -> Dict[str, Dict]:
        """
        List all documents stored in this session.
        """
        return {
            "documents": [
                {
                    "filename": doc.filename,
                    "last_updated": doc.last_updated.isoformat(),
                    "token_count": doc.token_count,
                    "document_id": doc.document_id
                }
                for doc in self.documents.values()
            ],
            "total_documents": len(self.documents),
            "total_tokens": self.total_tokens
        }

    def get_qa_history(self) -> Dict:
        """
        Retrieve QA history for this session.
        """
        current_doc_ids = {doc.document_id for doc in self.documents.values()}
        return {
            "history": [
                {
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "timestamp": pair["timestamp"].isoformat(),
                    "using_current_docs": pair["document_ids"].issubset(current_doc_ids),
                    "documents_used": list(pair["document_ids"])
                }
                for pair in self.qa_history.qa_pairs
            ]
        }
