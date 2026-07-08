from pypdf import PdfReader
import pandas as pd

# PDF Extraction --------
def extract_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    text = ""
  
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text: 
            text += page_text + "\n"

    return text


# # Excel Extraction --------
# def extract_excel(file_path: str) -> str:
#     df = pd.read_excel(file_path)

#     text = ""
#     for col in df.columns:
#         values = " ".join(df[col].astype(str))
#         text += f"{col}: {values}\n"

#     return text
 
from docx import Document

def extract_word(file_path: str) -> str:
    doc = Document(file_path)
    text = ""

    for para in doc.paragraphs:
        text += para.text + "\n"

    return text