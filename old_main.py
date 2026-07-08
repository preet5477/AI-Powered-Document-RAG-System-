from fastapi import FastAPI, UploadFile, File
import os
import numpy as np

from data_loader import extract_pdf
from chunking import recursive_chunk
from embedding import embed_chunks, model
from vector_store import create_index, search

app = FastAPI()

UPLOAD_PATH = "Data/mlalgo.pdf"

@app.get("/")
def home():
    return {"message": "RAG API Running"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    
    with open(UPLOAD_PATH, "wb") as f:
        f.write(await file.read())

    # Extract
    text = extract_pdf(UPLOAD_PATH)

    # Chunk
    chunks = recursive_chunk(text)

    # Embed
    embeddings = embed_chunks(chunks)

    # Store
    create_index(embeddings, chunks)

    return {
        "message": "PDF processed successfully",
        "total_chunks": len(chunks),
        "chunks": chunks[:250],
        "embeddings": embeddings[:250]
    }


@app.post("/ask")
async def ask_question(query: str): 
    
    query_embedding = model.encode([query])
    results = search(np.array(query_embedding))

    return {
        "query": query,
        "top_chunks": results
    }