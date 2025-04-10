import os
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Form, Query
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from datetime import datetime, timedelta, date
from typing import Optional, List
from sqlalchemy.orm import sessionmaker
from database import engine, get_db
import shutil
from models import Project, Document, ChatHistory
from document_manager import DocumentManager
from qa_utils import query_gpt
from prompts import RFP_SYNOPSIS_PROMPT, DEPENDENCIES_PROMPT, RESPONSE_STRUCTURE_PROMPT, STORYBOARDING_PROMPT, RESPONSE_TO_STORYBOARDING_PROMPT, FINAL_CONSOLIDATION_PROMPT
from config import FOLDER_PATH, ALLOWED_EXTENSIONS

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Document Q&A API")

# Initialize document manager
doc_manager = DocumentManager()

class QueryRequest(BaseModel):
    user_query: str
    project_id: int
    # use_rag: bool


    

class DeleteFileRequest(BaseModel):
    project_id: int
    file_id: int

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Ensure upload directory exists
if not FOLDER_PATH.exists():
    FOLDER_PATH.mkdir(parents=True, exist_ok=True)
elif not FOLDER_PATH.is_dir():
    raise ValueError(f"Upload path exists but is not a directory: {FOLDER_PATH}")




# --- SESSION MANAGEMENT ENDPOINTS ---
@app.post("/new-session/")
async def create_new_rfp_session(session_id: str):
    """Create a new RFP session."""
    try:
        doc_manager.create_new_session(session_id)
        return {"message": f"New RFP session created: {session_id}"}
    except HTTPException as e:
        raise e

@app.post("/switch-session/")
async def switch_rfp_session(session_id: str):
    """Switch to an existing RFP session."""
    try:
        doc_manager.switch_to_session(session_id)
        return {"message": f"Switched to RFP session: {session_id}"}
    except HTTPException as e:
        raise e

@app.delete("/delete-session/")
async def delete_rfp_session(session_id: str):
    """Delete an RFP session."""
    try:
        doc_manager.delete_session(session_id)
        return {"message": f"Deleted RFP session: {session_id}"}
    except HTTPException as e:
        raise e

# --- DOCUMENT UPLOAD TO KNOWLEDGEBASE ENDPOINT ---
@app.post("/upload-knowledge/")
async def upload_file(files: list[UploadFile] = File(...)):
    """
    Upload multiple files and process them for document extraction.
    """
    responses = []
    # current_session = doc_manager.get_current_session()  # Ensure a session is active

    for file in files:
        file_path = FOLDER_PATH / file.filename
        print("file_path", file_path)
        try:
            if not file.filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
                responses.append({
                    "file_name": file.filename,
                    "status": "failed",
                    "message": f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
                })
                continue

            # Save file temporarily
            content = await file.read()
            doc_id = doc_manager.add_or_update_document(file.filename, content)
            print("doc_id:", doc_id)
            responses.append({
                "file_name": file.filename,
                "status": "success",
                "message": "File uploaded successfully",
                "document_id": doc_id
            })
        except Exception as e:
            responses.append({
                "file_name": file.filename,
                "status": "failed",
                "message": f"Error processing file: {str(e)}"
            })
        finally:
            if file_path.exists():
                file_path.unlink()  # Cleanup temporary files

    return {
        "responses": responses,
        "total_processed": len(responses),
        "successful": sum(1 for r in responses if r["status"] == "success"),
        "failed": sum(1 for r in responses if r["status"] == "failed")
    }

# --- PROJECT CREATION ENDPOINT ---
@app.post("/projects/")
async def create_project(name: str, db = Depends(get_db)):
    try:
        project = Project(name=name)
        db.add(project)
        db.commit()
        db.refresh(project)
        return JSONResponse(content={"id": project.id, "name": project.name}, status_code=200)
    except Exception as err:
        return HTTPException(status_code=500, detail=str(err))


# --- DOCUMENT UPLOAD ENDPOINT ---
@app.post("/upload-files/")
async def upload_file( files: list[UploadFile] = File(...), project_id: int = Form(...), db = Depends(get_db)):
    """
    Upload multiple files and process them for document extraction.
    """
    not_support_files = []
    project = db.query(Project).filter(Project.id == project_id).first()
    # print("project", project.id)
    if not project:
        raise HTTPException(status_code=404, detail="Folder not found")

    uploaded_files = []
    existing_files = []
    # current_session = doc_manager.get_current_session()  # Ensure a session is active
    try:
        for file in files:
            # file_path = FOLDER_PATH / file.filename
            # print("file_path", file_path)
            if not file.filename.lower().endswith(tuple(ALLOWED_EXTENSIONS)):
                not_support_files.append({
                    "file_name": file.filename,
                    "status": "failed",
                    "message": f"Invalid file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
                })
                continue
            
            db_result = db.query(Document).filter(Document.filename==file.filename , Document.project_id == project_id).first()
            if db_result:
                existing_files.append(file.filename)
            else:
                file_location = os.path.join(UPLOAD_DIR, file.filename)
                with open(file_location, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                document = Document(filename=file.filename, project_id=project_id)
                db.add(document)
                uploaded_files.append({
                    # "id": document.id, 
                    "filename": document.filename, 
                    "project_id": document.project_id
                })

        db.commit()
        # db.refresh(document)
        return JSONResponse(content={"uploaded_files": uploaded_files, "existing_files": existing_files, "failed_files": not_support_files}, status_code=200)
    except Exception as err:
        print(err)
        return HTTPException(status_code=500, detail=str(err))


# --- ANALYSIS ENDPOINTS ---
@app.post("/rfp-synopsis/")
async def extract_rfp_synopsis():
    """Extract RFP synopsis from uploaded documents."""
    current_session = doc_manager.get_current_session()
    
    if not current_session.documents:
        raise HTTPException(status_code=404, detail="No documents uploaded in the current session.")


    context = current_session.get_relevant_context("RFP synopsis", top_k=30)
    # print("🔍 Retrieved Context for RFP Synopsis:", context)  # Debugging log

    if not context.strip():
        raise HTTPException(status_code=400, detail="No relevant context retrieved for RFP synopsis.")

    result = query_gpt("Extract RFP Synopsis", RFP_SYNOPSIS_PROMPT.format(context=context))
    return {"result": result}

@app.post("/critical-dependencies/")
async def extract_critical_dependencies():
    """Extract critical dependencies explicitly stated in the RFP."""
    current_session = doc_manager.get_current_session()
    if not current_session.documents:
        raise HTTPException(status_code=404, detail="No documents uploaded in the current session.")
    
    context = current_session.get_relevant_context("Critical dependencies", top_k=30)  # Fix: Ensure top_k=30
    print("🔍 Retrieved Context for Critical Dependencies:", context)  # Debugging log
    
    if not context.strip():
        raise HTTPException(status_code=400, detail="No relevant context retrieved for critical dependencies.")
    
    result = query_gpt("Extract Critical Dependencies", DEPENDENCIES_PROMPT.format(context=context))
    return {"result": result}

@app.post("/response-structure/")
async def extract_response_structure():
    """Extract the response structure required in the RFP."""
    current_session = doc_manager.get_current_session()
    
    if not current_session.documents:
        raise HTTPException(status_code=404, detail="No documents uploaded in the current session.")

    context = current_session.get_relevant_context("Response structure", top_k=30)
    print("🔍 Retrieved Context for Response Structure:", context)  # Debugging log

    if not context.strip():
        raise HTTPException(status_code=400, detail="No relevant context retrieved for response structure.")

    result = query_gpt("Extract Response Structure", RESPONSE_STRUCTURE_PROMPT.format(context=context))
    return {"result": result}


@app.post("/generate-response/")
async def generate_response(request: QueryRequest, db = Depends(get_db)):
    """
    Generate structured storyboarding response using full content from multiple PDF/DOCX/xlsx/txt files.
    - Checks each document in local 'uploads/' directory and database
    - Combines all file contents and generates a storyboarding response.
    - Combined the storboarding response using rag or without rag as the per the instructions it generate a refined response.
    - Combined the storyboarding response and refined response it generate the final response which is the output of this api.
    """
    try:
        extracted_docs = []  # List to store documents data

        document = db.query(Document).filter(Document.project_id == request.project_id).all()
        project = db.query(Project).filter(Project.id == request.project_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="No documents uploaded in the current project.")
        
        for file in document:
            file_path = f"uploads/{file.filename}"
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail=f"File '{file.filename}' not found in uploads directory.")

            extracted_docs.append(file_path) 

        extracted_data = "\n".join(extracted_docs)  # Join all chunks into a single string efficiently
        
        custom_prompt = STORYBOARDING_PROMPT.format(extracted_data, request.user_query)
        storyboarding_response = query_gpt("Generate StoryBuilding Steps.", custom_prompt)

        if storyboarding_response:

            vector_response = doc_manager.get_relevant_context(request.user_query, top_k=3)
            # print("vector response", vector_response)
            if not vector_response:
                return HTTPException(status_code=404, detail="Data from rag is not found")
            
            custom_prompt = RESPONSE_TO_STORYBOARDING_PROMPT.format(extracted_data, vector_response, request.user_query, storyboarding_response)
           
            rag_refined_response = query_gpt("Response of story building with rag response.", custom_prompt)
            # print("rag refined response",rag_refined_response)
            
            if rag_refined_response:
                custom_prompt = FINAL_CONSOLIDATION_PROMPT.format(extracted_data, request.user_query, storyboarding_response, rag_refined_response)
                final_response = query_gpt("Generate final response.", custom_prompt)

                try:
                    chat_history = ChatHistory(message=request.user_query, response=final_response, project_id=request.project_id, project_name=project.name)
                    db.add(chat_history)
                    db.commit()
                    db.refresh(chat_history)
                    return JSONResponse(content={"final_response": final_response}, status_code=200)
                except Exception as err:
                    return HTTPException(status_code=404, detail=str(err))

            else:
                return JSONResponse(content={"message": "There is no response of storyboarding using rag system from ai."}, status_code=200)

        else:
            return JSONResponse(content={"message": "There is no storyboarding provided from ai."}, status_code=200)

    except Exception as e:
        raise HTTPException(status_code=500, detail=e)
    

# --- GETTING LATEST CREATED PROJECT ---
@app.get("/get-projects/")
async def get_project(db = Depends(get_db)):
    try:
        projects = db.query(Project).order_by(Project.created_at.desc()).all()
        if not projects:
            return HTTPException(status_code=404, detail="No project found. Please create a project.")
        
        projects_data = jsonable_encoder(projects)
        return JSONResponse(content={"projects": projects_data}, status_code=200)
    except Exception as err:
        return HTTPException(status_code=500, detail=str(err))
    

# --- LIST OF FILES WHICH IS UPLOAD IN A PROJECT ---
@app.get("/list-files/")
async def list_files(project_id: int, db= Depends(get_db)):
    try:
        files_list = db.query(Document).filter(Document.project_id == project_id).all()
        if not files_list:
            return HTTPException(status_code=404, detail="No file found in the project. Please upload a file.")
        
        files_list_data = jsonable_encoder(files_list)
        return JSONResponse(content={"files_list": files_list_data}, status_code=200)
    
    except Exception as err:
        return HTTPException(status_code=500, detail=str(err))
    
@app.get("/chat-history/")
async def chat_history(filter: Optional[str] = Query(None, description="Filter type: today, yesterday, last_7_days, this_month, custom"),
start_date: Optional[date] = Query(None, description="Start date for custom filter"),
end_date: Optional[date] = Query(None, description="End date for custom filter"),
db=Depends(get_db)):
    try:
        query = db.query(ChatHistory)

        now = datetime.utcnow().date()

        if filter == "today":
            query = query.filter(ChatHistory.created_at >= datetime.combine(now, datetime.min.time()),
                                ChatHistory.created_at <= datetime.combine(now, datetime.max.time()))

        elif filter == "yesterday":
            yesterday = now - timedelta(days=1)
            query = query.filter(ChatHistory.created_at >= datetime.combine(yesterday, datetime.min.time()),
                                ChatHistory.created_at <= datetime.combine(yesterday, datetime.max.time()))

        elif filter == "last_7_days":
            last_7 = now - timedelta(days=7)
            query = query.filter(ChatHistory.created_at >= last_7)

        elif filter == "this_month":
            first_day = now.replace(day=1)
            query = query.filter(ChatHistory.created_at >= first_day)

        elif filter == "custom" and start_date and end_date:
            query = query.filter(ChatHistory.created_at >= datetime.combine(start_date, datetime.min.time()),
                                ChatHistory.created_at <= datetime.combine(end_date, datetime.max.time()))

        chat_history_result = query.order_by(ChatHistory.created_at.desc()).all()
        chat_history = jsonable_encoder(chat_history_result)
        return JSONResponse(content={"result": chat_history}, status_code = 200)
    
    except Exception as err:
        return HTTPException(status_code=500, detail=str(err))

    
# --- PROJECT DELETION ENDPOINT ---
@app.delete("/project-delete/")
async def delete_project(project_id: int, db = Depends(get_db)):
    try:
        project = db.query(Project).filter(Project.id == project_id).first()
        if not project:
            return HTTPException(status_code=404, detail="No project found.")
        
        db.query(Document).filter(Document.project_id == project_id).delete()
        db.delete(project)
        db.commit()
        # db.refresh(project)
        return JSONResponse(content={"{} is deleted successfully.".format(project.name)}, status_code=200)
    except Exception as err:
        return HTTPException(status_code=500, detail=str(err))

    
# --- DOCUMENT MANAGEMENT ENDPOINTS ---
@app.delete("/file-delete/")
async def delete_file(request: DeleteFileRequest, db = Depends(get_db)):
    try:
        document = db.query(Document).filter(Document.id == request.file_id, Document.project_id == request.project_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="File not found")
        
        db.delete(document)
        db.commit()
        # db.refresh(document)
        return JSONResponse(content={"message": "{} is deleted successfully".format(document.filename)}, status_code=200)
    
    except Exception as err:    
        return HTTPException(status_code=500, detail=str(err))
    
# --- DOCUMENT MANAGEMENT ENDPOINTS ---
@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete a document from the current RFP session."""
    if doc_manager.remove_document(filename):
        return {
            "message": f"Document '{filename}' removed successfully",
            "remaining_documents": list(doc_manager.get_current_session().documents.keys()),
            "new_total_tokens": doc_manager.get_current_session().total_tokens
        }
    raise HTTPException(status_code=404, detail="Document not found")

@app.get("/documents/")
def list_documents():
    """List all documents in the current RFP session."""
    return doc_manager.list_documents()

# --- QA HISTORY ENDPOINT ---
@app.get("/history/")
def get_qa_history():
    """Retrieve QA history for the current RFP session."""
    return doc_manager.get_qa_history()

# --- SERVING FRONTEND ---
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

# app.mount("/static", StaticFiles(directory="frontend/build/static"), name="static")

# @app.get("/", response_class=HTMLResponse)
# async def read_index():
#     """Serve the frontend React app."""
#     with open("frontend/build/index.html", "r") as f:
#         return HTMLResponse(content=f.read())

# --- RUN SERVER ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
