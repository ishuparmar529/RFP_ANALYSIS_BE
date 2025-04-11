import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
from fastapi import HTTPException
from rfp_session import RfpSession

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DocumentManager:
    """
    Manages multiple RFP Sessions, each containing documents.
    """

    def __init__(self):
        self.sessions: Dict[str, RfpSession] = {}
        print("sessions:", self.sessions)
        self.current_session_id: Optional[str] = None
        print("current sessions id: ",self.current_session_id)

    def create_new_session(self, session_id: str):
        """Create a new RFP session with a unique session ID."""
        if session_id in self.sessions:
            raise HTTPException(status_code=400, detail=f"Session with ID {session_id} already exists.")
        self.sessions[session_id] = RfpSession(session_id)
        
        # Only auto-switch if no active session exists
        if not self.current_session_id:
            self.current_session_id = session_id

        logger.info(f"Created new RFP session: {session_id}")

    def switch_to_session(self, session_id: str):
        """Switch to an existing RFP session."""
        if session_id not in self.sessions:
            logger.warning(f"Attempted to switch to non-existent session: {session_id}")
            raise HTTPException(status_code=400, detail=f"Session with ID {session_id} does not exist.")
        
        self.current_session_id = session_id
        logger.info(f"Switched to active RFP session: {session_id}")

    def delete_session(self, session_id: str):
        """Delete an RFP session and all associated documents."""
        if session_id not in self.sessions:
            raise HTTPException(status_code=400, detail=f"Session with ID {session_id} does not exist.")
        
        logger.info(f"Deleting session: {session_id}")
        del self.sessions[session_id]

        if self.current_session_id == session_id:
            self.current_session_id = None
            logger.info("Active session deleted. No active session now.")

    def get_current_session(self) -> RfpSession:
        """Retrieve the currently active RFP session."""
        if not self.current_session_id or self.current_session_id not in self.sessions:
            raise HTTPException(status_code=404, detail="No active RFP session. Please create or switch to a session.")
        return self.sessions[self.current_session_id]


    def add_or_update_document(self, filename: str, content: bytes) -> str:
        """
        Add or update a document in the current RFP session.
        Processing and chunking now happens inside `RfpSession`.
        """
        try:
            current_session = self.get_current_session() 
        except HTTPException as e:
            rfp_session = RfpSession(session_id=None)
            doc_id = rfp_session.add_or_update_document(filename, content)
            logger.info(f"Added/Updated document {filename} without session.")
            return doc_id
        
        doc_id = current_session.add_or_update_document(filename, content)

        logger.info(f"Added/Updated document {filename} in session {current_session.session_id}.")
        return doc_id
    

    def remove_document(self, filename: str) -> bool:
        """Remove a document from the current RFP session."""
        current_session = self.get_current_session()
        return current_session.remove_document(filename)

    def get_relevant_context(self, query: str, top_k: int = 30) -> str:
        """Retrieve relevant document chunks using vector search."""
        try:
            current_session = self.get_current_session()
        except HTTPException as e:
            rfp_session = RfpSession(session_id = None)
            # print("rfp_session",rfp_session)
            return rfp_session.get_relevant_context(query, top_k=top_k)
        
        return current_session.get_relevant_context(query, top_k=top_k)
    
    def get_relevant_context_for_rag_chat(self, query: str, top_k: int = 30) -> str:
        """Retrieve relevant document chunks using vector search."""
        try:
            current_session = self.get_current_session()
        except HTTPException as e:
            rfp_session = RfpSession(session_id = None)
            # print("rfp_session",rfp_session)
            return rfp_session.get_relevant_context_for_rag_chat(query, top_k=top_k)
       
        return current_session.get_relevant_context_for_rag_chat(query, top_k=top_k)
    
    def get_relevant_context_without_session(self, query: str, top_k: int = 30) -> str:
        """Retrieve relevant document chunks using vector search."""
        rfp_session = RfpSession()
        return rfp_session.get_relevant_context(query, top_k=top_k)

    def list_documents(self):
        """List all documents in the current RFP session."""
        return self.get_current_session().list_documents()

    def get_qa_history(self):
        """Retrieve QA history for the current RFP session."""
        return self.get_current_session().get_qa_history()
