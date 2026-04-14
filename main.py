import argparse
import json
from pathlib import Path

from rag.config import (
    DEFAULT_DOCS,
    DEFAULT_PDFS,
    CLEANED_DATA_PATH,
    KNOWLEDGE_BASE_PATH,
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
)
from rag.ingest_docx import ingest_docs
from rag.pdf_ingest import ingest_pdfs
from rag.indexer import build_tfidf_index, load_chunks
from rag.query import run_query, format_results
from rag.vector_store import build_chroma_index, query_chroma


def handle_ingest(args):
    docs = [Path(p) for p in args.docs] if args.docs else DEFAULT_DOCS
    chunks = ingest_docs(docs, KNOWLEDGE_BASE_PATH, CLEANED_DATA_PATH, image_dir=DOCX_IMAGE_DIR)
    print(f"Ingested {len(docs)} docx -> {KNOWLEDGE_BASE_PATH}")
    print(f"Cleaned text -> {CLEANED_DATA_PATH}")
    table_examples = [c for c in chunks if c.get("metadata")][:2]
    text_examples = [c for c in chunks if not c.get("metadata")][:2]
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
    KNOWLEDGE_BASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with KNOWLEDGE_BASE_PATH.open("w", encoding="utf-8") as f:
        json.dump(pdf_chunks, f, ensure_ascii=False, indent=2)
    CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CLEANED_DATA_PATH.open("w", encoding="utf-8") as f:
        for chunk in pdf_chunks:
            f.write(chunk["content"] + "\n")

    print(f"Ingested {len(pdfs)} pdf -> {KNOWLEDGE_BASE_PATH}")
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
