import io
import pdfplumber
import docx

def extract_text_from_pdf(file_stream) -> str:
    """
    Extracts text from a PDF file stream using pdfplumber.
    Raises ValueError if it's likely a scanned/image PDF with no text.
    """
    text = ""
    try:
        with pdfplumber.open(file_stream) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        raise ValueError(f"Corrupt or unreadable PDF: {str(e)}")
        
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("No text could be extracted. This might be a scanned image-based PDF which is unsupported.")
        
    return cleaned_text

def extract_text_from_docx(file_stream) -> str:
    """
    Extracts text from a DOCX file stream using python-docx.
    """
    text = ""
    try:
        doc = docx.Document(file_stream)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        raise ValueError(f"Corrupt or unreadable DOCX: {str(e)}")
        
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("No text could be extracted. The document appears to be empty.")
        
    return cleaned_text

def extract_text(file_stream, filename: str) -> str:
    """
    Determines the file type from the filename and calls the appropriate extractor.
    """
    filename_lower = filename.lower()
    if filename_lower.endswith('.pdf'):
        return extract_text_from_pdf(file_stream)
    elif filename_lower.endswith('.docx'):
        return extract_text_from_docx(file_stream)
    else:
        raise ValueError(f"Unsupported file format for '{filename}'. Only .pdf and .docx are supported.")
