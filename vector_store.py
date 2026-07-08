# import faiss
# import numpy as np

# index = None
# stored_chunks = []

# def create_index(embeddings, chunks):
#     global index, stored_chunks 

#     dim = len(embeddings[0])
#     index = faiss.IndexFlatL2(dim)#Flat Index (Exact search)

#     index.add(np.array(embeddings))
#     stored_chunks = chunks


# def search(query_embedding, k=3):
#     global index, stored_chunks

#     D, I = index.search(query_embedding, k)
#     results = [stored_chunks[i] for i in I[0]]
#     return results

#-------------------Using PGVector------------------#

import numpy as np
from sqlalchemy.orm import Session
from models import Document

def store_embeddings(db: Session, embeddings, chunks):
    for emb, chunk in zip(embeddings, chunks):
        doc = Document(
            content=chunk,
            embedding=emb.tolist()
        )
        db.add(doc)

    db.commit()

def search_similar(db: Session, query_embedding, k=3):
    query_vector = query_embedding.tolist()[0]

    results = db.query(Document).order_by(
        Document.embedding.l2_distance(query_vector)# Euclidean distance
    ).limit(k).all()

    return [r.content for r in results]