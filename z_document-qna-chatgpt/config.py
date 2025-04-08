import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
# Set the folder path where uploaded files will be temporarily stored
FOLDER_PATH = Path(os.getenv("UPLOAD_FOLDER", "/Users/home/PycharmProjects/z_document-qna-chatgpt/uploads"))

# Allowed file extensions for document uploads
ALLOWED_EXTENSIONS = {".pdf", ".doc", ".docx", ".xlsx", ".xls", ".txt"}

# Ensure the folder exists
if not FOLDER_PATH.exists():
    FOLDER_PATH.mkdir(parents=True, exist_ok=True)
