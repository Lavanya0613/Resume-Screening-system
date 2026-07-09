import pytest
from unittest.mock import MagicMock
from src.resume_parser import extract_text_from_pdf, extract_text_from_docx, extract_text

def test_extract_text_pdf(mocker):
    mock_pdfplumber = mocker.patch('src.resume_parser.pdfplumber.open')
    
    mock_pdf = MagicMock()
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "PDF Mock Text"
    mock_pdf.pages = [mock_page]
    
    mock_pdfplumber.return_value.__enter__.return_value = mock_pdf
    
    text = extract_text_from_pdf("dummy.pdf")
    assert "PDF Mock Text" in text

def test_extract_text_docx(mocker):
    mock_docx = mocker.patch('src.resume_parser.docx.Document')
    
    mock_doc = MagicMock()
    mock_paragraph = MagicMock()
    mock_paragraph.text = "DOCX Mock Text"
    mock_doc.paragraphs = [mock_paragraph]
    
    mock_docx.return_value = mock_doc
    
    text = extract_text_from_docx("dummy.docx")
    assert "DOCX Mock Text" in text

def test_extract_text_corrupt_pdf(mocker):
    mock_pdfplumber = mocker.patch('src.resume_parser.pdfplumber.open')
    mock_pdfplumber.side_effect = Exception("File is corrupt")
    
    with pytest.raises(ValueError) as excinfo:
        extract_text_from_pdf("corrupt.pdf")
    assert "Corrupt or unreadable PDF" in str(excinfo.value)

def test_extract_text_routing(mocker):
    mock_pdf = mocker.patch('src.resume_parser.extract_text_from_pdf', return_value="pdf")
    mock_docx = mocker.patch('src.resume_parser.extract_text_from_docx', return_value="docx")
    
    assert extract_text("stream", "test.pdf") == "pdf"
    assert extract_text("stream", "test.docx") == "docx"
    
    with pytest.raises(ValueError):
        extract_text("stream", "test.txt")
