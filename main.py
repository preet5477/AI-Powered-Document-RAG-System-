import numpy as np
import os

from fastapi import FastAPI, UploadFile, File, Depends
from sqlalchemy.orm import Session
from db import SessionLocal
from data_loader import extract_pdf, extract_word
from chunking import recursive_chunk
from embedding import embed_chunks, model
from vector_store import store_embeddings, search_similar
 
app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "Data")
os.makedirs(DATA_DIR, exist_ok=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
    
# DB dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/")
def home():
    return {"message": "RAG with PGVector Running"}


# Upload API
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), db: Session = Depends(get_db)):

    file_path = os.path.join(DATA_DIR, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Extract
    if file.filename.endswith(".pdf"):
        text = extract_pdf(file_path)

    elif file.filename.endswith(".docx"):
        text = extract_word(file_path)

    else:
        return {"error": "Only PDF and DOCX supported"}

    # Chunk
    chunks = recursive_chunk(text)

    # Embed
    embeddings = embed_chunks(chunks)

    # Store in DB
    store_embeddings(db, embeddings, chunks)

    return {
        "message": "Stored in PGVector DB",
        "chunks": len(chunks)
    }


# Query API
@app.post("/ask")
async def ask(query: str, db: Session = Depends(get_db)):

    query_embedding = model.encode([query])

    results = search_similar(db, np.array(query_embedding))

    return {
        "query": query,
        "results": results
    }