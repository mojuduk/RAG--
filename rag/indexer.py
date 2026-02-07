import json

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def load_chunks(chunks_path):
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_tfidf_index(chunks, vector_path, vectorizer_path):
    texts = [chunk["content"] for chunk in chunks]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
    matrix = vectorizer.fit_transform(texts)
    vector_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, vectorizer_path)
    from scipy import sparse

    sparse.save_npz(vector_path, matrix)
    return vectorizer, matrix


def load_tfidf_index(vector_path, vectorizer_path):
    vectorizer = joblib.load(vectorizer_path)
    from scipy import sparse

    matrix = sparse.load_npz(vector_path)
    return vectorizer, matrix


def search(query, chunks, vectorizer, matrix, top_k=5):
    query_vec = vectorizer.transform([query])
    scores = matrix @ query_vec.T
    scores = scores.toarray().ravel()
    if top_k >= len(scores):
        top_indices = np.argsort(-scores)
    else:
        top_indices = np.argpartition(-scores, top_k)[:top_k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
    results = []
    for idx in top_indices:
        results.append(
            {
                "score": float(scores[idx]),
                "chunk": chunks[idx],
            }
        )
    return results
