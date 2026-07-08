from fastapi import FastAPI, UploadFile, File
import os
import numpy as np

from data_loader import extract_pdf, extract_word
from chunking import recursive_chunk
from embedding import embed_chunks, model
from vector_store import create_index, search

app = FastAPI()

# Base directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
os.makedirs(DATA_DIR, exist_ok=True)


@app.get("/")
def home():
    return {"message": "RAG API Running"}


# ✅ Unified Upload API (PDF + DOCX)
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    file_path = os.path.join(DATA_DIR, file.filename)

    # Save file
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # -------- File Type Handling --------
    if file.filename.endswith(".pdf"):
        text = extract_pdf(file_path)

    elif file.filename.endswith(".docx"):
        text = extract_word(file_path)

    else: 
        return {"error": "Only PDF and DOCX files are supported"}

    # -------- Chunking --------
    chunks = recursive_chunk(text)

    # -------- Embedding --------
    embeddings = embed_chunks(chunks)

    # -------- Store --------
    create_index(embeddings, chunks)

    return {
        "message": "File processed successfully",
        "filename": file.filename,
        "text_length": len(text),
        "total_chunks": len(chunks),
        "Chunks": chunks[:3],
        "Embeddings":embeddings[:3]
    }


# Query API
@app.post("/ask")
async def ask_question(query: str):

    query_embedding = model.encode([query])
    results = search(np.array(query_embedding))

    return {
        "query": query,
        "top_chunks": results
    }