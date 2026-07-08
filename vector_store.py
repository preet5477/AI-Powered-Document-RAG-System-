import numpy as np

_stored_embeddings = []
_stored_chunks = []


def create_index(embeddings, chunks):
    global _stored_embeddings, _stored_chunks
    _stored_embeddings = [np.array(emb, dtype=float) for emb in embeddings]
    _stored_chunks = list(chunks)


def search(query_embedding, k=3):
    global _stored_embeddings, _stored_chunks

    if not _stored_embeddings:
        return []

    query_vector = np.array(query_embedding, dtype=float).reshape(-1)

    def normalize(vec):
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    query_vector = normalize(query_vector)
    stored_vectors = np.array([normalize(vec) for vec in _stored_embeddings])
    scores = stored_vectors @ query_vector
    top_indices = np.argsort(scores)[::-1][:k]

    return [_stored_chunks[i] for i in top_indices]


def store_embeddings(db, embeddings, chunks):
    create_index(embeddings, chunks)


def search_similar(db, query_embedding, k=3):
    return search(query_embedding, k=k)