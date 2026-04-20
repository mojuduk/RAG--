import argparse
import json
from pathlib import Path

from rag.config import (
    DEFAULT_DOCS,
    DEFAULT_PDFS,
    CLEANED_DATA_PATH,
    CLEANED_DOCX_PATH,
    CLEANED_PDF_PATH,
    KNOWLEDGE_BASE_PATH,
    KNOWLEDGE_BASE_DOCX_PATH,
    KNOWLEDGE_BASE_PDF_PATH,
    CHROMA_DIR,
    VECTOR_STORE_PATH,
    EMBEDDING_MODEL_NAME,
    VECTOR_PATH,
    VECTORIZER_PATH,
    PDF_MMD_DIR,
    TABLE_PARENTS_PATH,
    NOUGAT_CMD,
    PDF_IMAGE_DIR,
    PDF_IMAGE_DPI,
    DOCX_IMAGE_DIR,
    PDF_IMAGE_EXTRACT_DIR,
    PDF_OCR_IMAGE_DIR,
    PDF_OCR_DPI,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_NAME,
)
from rag.ingest_docx import ingest_docs
from rag.pdf_ingest import ingest_pdfs
from rag.indexer import build_tfidf_index, load_chunks
from rag.query import run_query, format_results
from rag.vector_store import build_chroma_index, query_chroma
from rag.ask import run_ask, format_ask_output
from rag.eval import run_eval
from rag.qa_builder import build_qa_candidates


def _load_chunks_safe(path: Path):
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _chunk_dedup_key(chunk):
    content = (chunk.get("content") or "").strip()
    metadata = chunk.get("metadata") or {}
    try:
        meta_key = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
    except Exception:
        meta_key = str(metadata)
    return f"{content}||{meta_key}"


def _dedup_chunks(chunks):
    out = []
    seen = set()
    for c in chunks:
        key = _chunk_dedup_key(c)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _write_chunks(path: Path, chunks):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


def _write_cleaned(path: Path, chunks):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            txt = (c.get("content") or "").strip()
            if txt:
                f.write(txt + "\n")


def _merge_knowledge_bases():
    docx_chunks = _load_chunks_safe(KNOWLEDGE_BASE_DOCX_PATH)
    pdf_chunks = _load_chunks_safe(KNOWLEDGE_BASE_PDF_PATH)
    merged = _dedup_chunks(docx_chunks + pdf_chunks)
    _write_chunks(KNOWLEDGE_BASE_PATH, merged)
    _write_cleaned(CLEANED_DATA_PATH, merged)
    return merged, len(docx_chunks), len(pdf_chunks)


def handle_ingest(args):
    docs = [Path(p) for p in args.docs] if args.docs else DEFAULT_DOCS
    docx_chunks = ingest_docs(docs, KNOWLEDGE_BASE_DOCX_PATH, CLEANED_DOCX_PATH, image_dir=DOCX_IMAGE_DIR)
    merged, docx_count, pdf_count = _merge_knowledge_bases()
    print(f"Ingested {len(docs)} docx -> {KNOWLEDGE_BASE_DOCX_PATH}")
    print(f"Merged KB (docx={docx_count}, pdf={pdf_count}, total={len(merged)}) -> {KNOWLEDGE_BASE_PATH}")
    print(f"Cleaned text -> {CLEANED_DATA_PATH}")
    table_examples = [c for c in docx_chunks if c.get("metadata")][:2]
    text_examples = [c for c in docx_chunks if not c.get("metadata")][:2]
    for chunk in table_examples + text_examples:
        print(chunk)


def handle_pdf_ingest(args):
    pdfs = [Path(p) for p in args.pdfs] if args.pdfs else DEFAULT_PDFS
    pdf_chunks, _parents = ingest_pdfs(
        pdfs,
        PDF_MMD_DIR,
        TABLE_PARENTS_PATH,
        nougat_cmd=args.nougat_cmd or NOUGAT_CMD,
        image_root=PDF_IMAGE_DIR if args.pdf_to_images else None,
        image_dpi=args.pdf_image_dpi,
        table_tool=args.table_tool,
        tabula_guess=not args.tabula_no_guess,
        tabula_lattice=not args.tabula_no_lattice,
        tabula_stream=not args.tabula_no_stream,
        tabula_jvm_opts=args.tabula_jvm_opts,
        image_extract_dir=PDF_IMAGE_EXTRACT_DIR,
        ocr_on_garbled=args.ocr_on_garbled,
        ocr_image_dir=PDF_OCR_IMAGE_DIR,
        ocr_dpi=args.ocr_dpi,
    )
    _write_chunks(KNOWLEDGE_BASE_PDF_PATH, pdf_chunks)
    _write_cleaned(CLEANED_PDF_PATH, pdf_chunks)
    merged, docx_count, pdf_count = _merge_knowledge_bases()
    print(f"Ingested {len(pdfs)} pdf -> {KNOWLEDGE_BASE_PDF_PATH}")
    print(f"Merged KB (docx={docx_count}, pdf={pdf_count}, total={len(merged)}) -> {KNOWLEDGE_BASE_PATH}")
    print(f"Cleaned text -> {CLEANED_DATA_PATH}")
    table_examples = [c for c in pdf_chunks if c.get("metadata")][:2]
    text_examples = [c for c in pdf_chunks if not c.get("metadata")][:2]
    for chunk in table_examples + text_examples:
        print(chunk)


def handle_index(args):
    chunks = load_chunks(KNOWLEDGE_BASE_PATH)
    build_tfidf_index(chunks, VECTOR_PATH, VECTORIZER_PATH)
    print(f"Indexed {len(chunks)} chunks -> {VECTOR_PATH}")


def handle_query(args):
    results = run_query(
        KNOWLEDGE_BASE_PATH, VECTOR_PATH, VECTORIZER_PATH, args.query, args.top_k
    )
    print(format_results(results))


def handle_vindex(args):
    chunks = load_chunks(KNOWLEDGE_BASE_PATH)
    build_chroma_index(chunks, CHROMA_DIR, VECTOR_STORE_PATH, args.model)
    print(f"Vector store -> {CHROMA_DIR}")
    print(f"Vector store -> {VECTOR_STORE_PATH}")


def handle_vquery(args):
    results = query_chroma(CHROMA_DIR, VECTOR_STORE_PATH, args.query, top_k=args.top_k)
    if args.hybrid:
        tfidf_results = run_query(
            KNOWLEDGE_BASE_PATH,
            VECTOR_PATH,
            VECTORIZER_PATH,
            args.query,
            args.top_k,
        )
        seen = set()
        merged = []
        for item in results + tfidf_results:
            content = item["chunk"]["content"]
            if content in seen:
                continue
            seen.add(content)
            merged.append(item)
        results = merged[: args.top_k]

    for idx, item in enumerate(results, 1):
        print(
            {
                "rank": idx,
                "score": round(item["score"], 6),
                "content": item["chunk"]["content"],
                "metadata": item["chunk"].get("metadata", {}),
            }
        )


def handle_ask(args):
    use_hybrid = bool(args.hybrid) and not bool(args.no_hybrid)
    result = run_ask(
        question=args.question,
        chunks_path=KNOWLEDGE_BASE_PATH,
        vector_path=VECTOR_PATH,
        vectorizer_path=VECTORIZER_PATH,
        chroma_dir=CHROMA_DIR,
        store_path=VECTOR_STORE_PATH,
        top_k=args.top_k,
        use_hybrid=use_hybrid,
        model_name=args.model,
        llm_provider=args.llm_provider,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
        openai_model=args.openai_model,
        openai_base_url=args.openai_base_url,
        openai_api_key_env=args.openai_api_key_env,
        max_new_tokens=args.max_new_tokens,
    )
    print(format_ask_output(result))


def handle_serve(args):
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("uvicorn is required for serve command.") from exc
    uvicorn.run("rag.api:app", host=args.host, port=args.port, reload=args.reload)


def handle_eval(args):
    summaries = run_eval(
        qa_path=Path(args.qa_file),
        top_k=args.top_k,
        chunks_path=KNOWLEDGE_BASE_PATH,
        vector_path=VECTOR_PATH,
        vectorizer_path=VECTORIZER_PATH,
        chroma_dir=CHROMA_DIR,
        store_path=VECTOR_STORE_PATH,
        out_summary_csv=Path(args.out_summary),
        out_details_csv=Path(args.out_details),
        llm_provider=args.llm_provider,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
        openai_model=args.openai_model,
        openai_base_url=args.openai_base_url,
        openai_api_key_env=args.openai_api_key_env,
        max_new_tokens=args.max_new_tokens,
    )
    for row in summaries:
        print(row)
    print(f"Summary CSV -> {Path(args.out_summary)}")
    print(f"Details CSV -> {Path(args.out_details)}")


def handle_gen_qa(args):
    candidates = build_qa_candidates(
        chunks_path=KNOWLEDGE_BASE_PATH,
        out_path=Path(args.out),
        max_items=args.max_items,
        include_table=not args.no_table,
        include_text=not args.no_text,
    )
    print(f"Generated {len(candidates)} QA candidates -> {Path(args.out)}")


def build_parser():
    parser = argparse.ArgumentParser(description="RAG pipeline for docx datasets.")
    sub = parser.add_subparsers(dest="command")

    ingest_parser = sub.add_parser("ingest", help="Extract chunks from DOCX files.")
    ingest_parser.add_argument("--docs", nargs="*", help="DOCX paths")
    ingest_parser.set_defaults(func=handle_ingest)

    pdf_parser = sub.add_parser("pdf-ingest", help="Extract chunks from PDF files.")
    pdf_parser.add_argument("--pdfs", nargs="*", help="PDF paths")
    pdf_parser.add_argument(
        "--nougat-cmd",
        nargs=argparse.REMAINDER,
        help="Nougat CLI command, e.g. python -m predict --markdown",
    )
    pdf_parser.add_argument(
        "--pdf-to-images",
        action="store_true",
        help="Render PDF pages to images before running Nougat.",
    )
    pdf_parser.add_argument(
        "--pdf-image-dpi",
        type=int,
        default=PDF_IMAGE_DPI,
        help="DPI used when rendering PDF pages to images.",
    )
    pdf_parser.add_argument(
        "--table-tool",
        choices=["nougat", "pdfplumber", "tabula"],
        default="pdfplumber",
        help="Table extraction backend for PDF.",
    )
    pdf_parser.add_argument(
        "--ocr-on-garbled",
        action="store_true",
        help="Run PaddleOCR when PDF text is garbled.",
    )
    pdf_parser.add_argument(
        "--ocr-dpi",
        type=int,
        default=PDF_OCR_DPI,
        help="DPI used when rendering pages for OCR.",
    )
    pdf_parser.add_argument(
        "--tabula-no-guess",
        action="store_true",
        help="Disable tabula guess mode.",
    )
    pdf_parser.add_argument(
        "--tabula-no-lattice",
        action="store_true",
        help="Disable tabula lattice mode.",
    )
    pdf_parser.add_argument(
        "--tabula-no-stream",
        action="store_true",
        help="Disable tabula stream mode.",
    )
    pdf_parser.add_argument(
        "--tabula-jvm-opts",
        type=str,
        help="Extra JVM options for tabula, separated by ';' e.g. -Dfile.encoding=UTF-8;-Djava.awt.headless=true",
    )
    pdf_parser.set_defaults(func=handle_pdf_ingest)

    index_parser = sub.add_parser("index", help="Build TF-IDF index.")
    index_parser.set_defaults(func=handle_index)

    query_parser = sub.add_parser("query", help="Query the index.")
    query_parser.add_argument("query", help="Query text")
    query_parser.add_argument("--top-k", type=int, default=5)
    query_parser.set_defaults(func=handle_query)

    vindex_parser = sub.add_parser("vindex", help="Build embedding vector index.")
    vindex_parser.add_argument("--model", default=EMBEDDING_MODEL_NAME)
    vindex_parser.set_defaults(func=handle_vindex)

    vquery_parser = sub.add_parser("vquery", help="Query embedding vector index.")
    vquery_parser.add_argument("query", help="Query text")
    vquery_parser.add_argument("--top-k", type=int, default=5)
    vquery_parser.add_argument(
        "--hybrid", action="store_true", help="Merge TF-IDF and vector results."
    )
    vquery_parser.set_defaults(func=handle_vquery)

    ask_parser = sub.add_parser("ask", help="Retrieve + generate answer with evidence.")
    ask_parser.add_argument("question", help="User question")
    ask_parser.add_argument("--top-k", type=int, default=5, help="Number of evidence chunks")
    ask_parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Use hybrid retrieval (vector + keyword).",
    )
    ask_parser.add_argument(
        "--no-hybrid",
        action="store_true",
        help="Disable hybrid retrieval and use a single channel fallback.",
    )
    ask_parser.add_argument(
        "--model",
        default=None,
        help="Optional local model name/path for generation. If empty, fallback extractive answer is used.",
    )
    ask_parser.add_argument(
        "--llm-provider",
        choices=["ollama", "openai"],
        default="ollama",
        help="Generation provider for ask: ollama or openai-compatible endpoint.",
    )
    ask_parser.add_argument(
        "--ollama-model",
        default=OLLAMA_MODEL_NAME,
        help="Ollama model name. Default: qwen3:8b",
    )
    ask_parser.add_argument(
        "--ollama-url",
        default=OLLAMA_BASE_URL,
        help="Ollama base URL, e.g. http://127.0.0.1:11434",
    )
    ask_parser.add_argument(
        "--openai-model",
        default=None,
        help="OpenAI-compatible model name (used when --llm-provider openai).",
    )
    ask_parser.add_argument(
        "--openai-base-url",
        default=None,
        help="OpenAI-compatible base URL (e.g. DashScope compatible endpoint).",
    )
    ask_parser.add_argument(
        "--openai-api-key-env",
        default="OPENAI_API_KEY",
        help="Env var name containing API key for openai-compatible provider.",
    )
    ask_parser.add_argument("--max-new-tokens", type=int, default=160)
    ask_parser.set_defaults(func=handle_ask, hybrid=False)

    serve_parser = sub.add_parser("serve", help="Run FastAPI backend service.")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    serve_parser.set_defaults(func=handle_serve)

    eval_parser = sub.add_parser("eval", help="Run Day2 evaluation experiments.")
    eval_parser.add_argument("--qa-file", required=True, help="QA set path (.csv or .json)")
    eval_parser.add_argument("--top-k", type=int, default=5, help="Top-K for retrieval hit")
    eval_parser.add_argument(
        "--out-summary",
        default=str(Path("data") / "eval_summary.csv"),
        help="Output CSV for mode summary",
    )
    eval_parser.add_argument(
        "--out-details",
        default=str(Path("data") / "eval_details.csv"),
        help="Output CSV for per-sample details",
    )
    eval_parser.add_argument(
        "--llm-provider",
        choices=["ollama", "openai"],
        default="ollama",
        help="Generation provider in generation modes.",
    )
    eval_parser.add_argument(
        "--ollama-model",
        default=OLLAMA_MODEL_NAME,
        help="Ollama model name for hybrid+generation experiment",
    )
    eval_parser.add_argument(
        "--ollama-url",
        default=OLLAMA_BASE_URL,
        help="Ollama base URL",
    )
    eval_parser.add_argument(
        "--openai-model",
        default=None,
        help="OpenAI-compatible model name (used when --llm-provider openai).",
    )
    eval_parser.add_argument(
        "--openai-base-url",
        default=None,
        help="OpenAI-compatible base URL (e.g. DashScope compatible endpoint).",
    )
    eval_parser.add_argument(
        "--openai-api-key-env",
        default="OPENAI_API_KEY",
        help="Env var name containing API key for openai-compatible provider.",
    )
    eval_parser.add_argument("--max-new-tokens", type=int, default=160)
    eval_parser.set_defaults(func=handle_eval)

    genqa_parser = sub.add_parser("gen-qa", help="Generate candidate QA set from knowledge base.")
    genqa_parser.add_argument(
        "--out",
        default=str(Path("data") / "qa_candidates.json"),
        help="Output candidate QA JSON path",
    )
    genqa_parser.add_argument("--max-items", type=int, default=80)
    genqa_parser.add_argument("--no-table", action="store_true", help="Do not generate table-based QA")
    genqa_parser.add_argument("--no-text", action="store_true", help="Do not generate text-based QA")
    genqa_parser.set_defaults(func=handle_gen_qa)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
