import logging
from pathlib import Path
import PyPDF2
import pandas as pd
import docx
from io import StringIO

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def extract_text_from_file(file_path: str) -> str:
    """
    Extract text from various file types with proper encoding and error handling.
    Supported file types: PDF, DOC/DOCX, XLS/XLSX, TXT.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: Extracted text from the file.
    """
    file_extension = Path(file_path).suffix.lower()

    try:
        if file_extension == '.pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                return "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

        elif file_extension in {'.doc', '.docx'}:
            doc = docx.Document(file_path)
            return '\n'.join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())

        elif file_extension in {'.xlsx', '.xls'}:
            df = pd.read_excel(file_path)
            buffer = StringIO()
            df.to_string(buffer)
            return buffer.getvalue()

        elif file_extension == '.txt':
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
            raise ValueError("Could not decode text file with any supported encoding.")

        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}")
        return ""
