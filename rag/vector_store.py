import json


def load_chunks(chunks_path):
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_embedder(model_name):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "sentence-transformers not installed; cannot build embedding index."
        ) from exc
    return SentenceTransformer(model_name)


def _get_collection(persist_dir, collection_name, model_name):
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("chromadb not installed; cannot build vector store.") from exc

    client = chromadb.PersistentClient(path=str(persist_dir))
    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedder,
        metadata={"model": model_name},
    )
    return collection


def _sanitize_metadata_value(value):
    # Chroma metadata only allows scalar values (str/int/float/bool/None).
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple, set)):
        if all(isinstance(v, (str, int, float, bool)) or v is None for v in value):
            return " | ".join("" if v is None else str(v) for v in value)
        return json.dumps(list(value), ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _sanitize_metadata(metadata):
    out = {}
    for k, v in (metadata or {}).items():
        out[str(k)] = _sanitize_metadata_value(v)
    return out


def build_chroma_index(chunks, persist_dir, store_path, model_name, collection_name="rag_chunks"):
    texts = [chunk["content"] for chunk in chunks]
    ids = [f"chunk_{i:06d}" for i in range(len(chunks))]
    metadatas = []
    for chunk in chunks:
        metadata = _sanitize_metadata(dict(chunk.get("metadata", {})))
        metadata["source"] = "knowledge_base"
        metadatas.append(metadata)

    collection = _get_collection(persist_dir, collection_name, model_name)
    if collection.count() > 0:
        collection.delete(where={"source": "knowledge_base"})

    collection.add(ids=ids, documents=texts, metadatas=metadatas)

    store = {
        "model": model_name,
        "collection": collection_name,
        "count": len(chunks),
        "items": chunks,
    }
    store_path.parent.mkdir(parents=True, exist_ok=True)
    with open(store_path, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def query_chroma(persist_dir, store_path, query, top_k=5):
    with open(store_path, "r", encoding="utf-8") as f:
        store = json.load(f)
    collection = _get_collection(persist_dir, store["collection"], store["model"])
    results = collection.query(query_texts=[query], n_results=top_k)

    out = []
    for score, doc, meta in zip(
        results["distances"][0],
        results["documents"][0],
        results["metadatas"][0],
    ):
        out.append({"score": float(score), "chunk": {"content": doc, "metadata": meta}})
    return out
