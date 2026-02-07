import json

from .indexer import load_chunks, load_tfidf_index, search


def run_query(chunks_path, vector_path, vectorizer_path, query, top_k):
    chunks = load_chunks(chunks_path)
    vectorizer, matrix = load_tfidf_index(vector_path, vectorizer_path)
    results = search(query, chunks, vectorizer, matrix, top_k=top_k)
    return results


def format_results(results):
    lines = []
    for idx, item in enumerate(results, 1):
        chunk = item["chunk"]
        lines.append(
            json.dumps(
                {
                    "rank": idx,
                    "score": round(item["score"], 6),
                    "content": chunk["content"],
                    "metadata": chunk.get("metadata", {}),
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(lines)
