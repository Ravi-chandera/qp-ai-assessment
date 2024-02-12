import PyPDF2
import docx2txt

def extract_text_from_files(file_paths):
    """Extracts text from a list of files, handling PDFs and Word documents.

    Args:
        file_paths (list): A list of file paths.

    Returns:
        str: The combined text extracted from all files.

    Raises:
        ValueError: If an unsupported file type is encountered.
    """

    extracted_text = ""
    for file_path in file_paths:
        try:
            if file_path.endswith(".pdf"):
                extracted_text += extract_text_from_pdf(file_path)
            elif file_path.endswith(".docx"):
                extracted_text += extract_text_from_docx(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_path}")
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    return extracted_text

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.

    Returns:
        str: The text extracted from the PDF file.
    """

    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text


def extract_text_from_docx(docx_path):
    """Extracts text from a Word document.

    Args:
        docx_path (str): The path to the Word document.

    Returns:
        str: The text extracted from the Word document.
    """
    text = docx2txt.process(docx_path)
    return text


# Usage example:
# file_paths = ["report.pdf", "presentation.docx", "unsupported_file.txt"]
# extracted_text = extract_text_from_files(file_paths)
# print(extracted_text)
