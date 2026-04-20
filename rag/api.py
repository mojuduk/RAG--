import json
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag.ask import retrieve_evidence, run_ask
from rag.config import (
    CHROMA_DIR,
    CLEANED_DATA_PATH,
    DEFAULT_DOCS,
    DEFAULT_PDFS,
    DOCX_IMAGE_DIR,
    EMBEDDING_MODEL_NAME,
    KNOWLEDGE_BASE_PATH,
    PDF_IMAGE_DIR,
    PDF_IMAGE_DPI,
    PDF_IMAGE_EXTRACT_DIR,
    PDF_MMD_DIR,
    PDF_OCR_DPI,
    PDF_OCR_IMAGE_DIR,
    TABLE_PARENTS_PATH,
    VECTOR_PATH,
    VECTOR_STORE_PATH,
    VECTORIZER_PATH,
    NOUGAT_CMD,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL_NAME,
)
from rag.ingest_docx import ingest_docs
from rag.indexer import build_tfidf_index, load_chunks
from rag.pdf_ingest import ingest_pdfs
from rag.vector_store import build_chroma_index


app = FastAPI(title="Pharma RAG Service", version="0.1.0")
WEB_DIR = Path(__file__).resolve().parents[1] / "web"
INDEX_HTML = WEB_DIR / "index.html"


class DocxIngestRequest(BaseModel):
    docs: Optional[List[str]] = None


class PdfIngestRequest(BaseModel):
    pdfs: Optional[List[str]] = None
    table_tool: str = Field(default="tabula", pattern="^(nougat|pdfplumber|tabula)$")
    ocr_on_garbled: bool = True
    tabula_no_guess: bool = False
    tabula_no_lattice: bool = False
    tabula_no_stream: bool = True
    tabula_jvm_opts: Optional[str] = None
    pdf_to_images: bool = False
    pdf_image_dpi: int = PDF_IMAGE_DPI


class RetrieveRequest(BaseModel):
    question: str
    top_k: int = 5
    hybrid: bool = True


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    hybrid: bool = True
    model: Optional[str] = None
    ollama_model: Optional[str] = OLLAMA_MODEL_NAME
    ollama_url: str = OLLAMA_BASE_URL
    max_new_tokens: int = 160


def _write_chunks(chunks):
    KNOWLEDGE_BASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with KNOWLEDGE_BASE_PATH.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    CLEANED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CLEANED_DATA_PATH.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk["content"] + "\n")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def home():
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(str(INDEX_HTML))


@app.post("/ingest/docx")
def ingest_docx_api(req: DocxIngestRequest):
    docs = [Path(p) for p in req.docs] if req.docs else DEFAULT_DOCS
    chunks = ingest_docs(docs, KNOWLEDGE_BASE_PATH, CLEANED_DATA_PATH, image_dir=DOCX_IMAGE_DIR)
    return {"ingested_docx": len(docs), "chunks": len(chunks), "path": str(KNOWLEDGE_BASE_PATH)}


@app.post("/ingest/pdf")
def ingest_pdf_api(req: PdfIngestRequest):
    pdfs = [Path(p) for p in req.pdfs] if req.pdfs else DEFAULT_PDFS
    chunks, parents = ingest_pdfs(
        pdfs,
        PDF_MMD_DIR,
        TABLE_PARENTS_PATH,
        nougat_cmd=NOUGAT_CMD,
        image_root=PDF_IMAGE_DIR if req.pdf_to_images else None,
        image_dpi=req.pdf_image_dpi,
        table_tool=req.table_tool,
        tabula_guess=not req.tabula_no_guess,
        tabula_lattice=not req.tabula_no_lattice,
        tabula_stream=not req.tabula_no_stream,
        tabula_jvm_opts=req.tabula_jvm_opts,
        image_extract_dir=PDF_IMAGE_EXTRACT_DIR,
        ocr_on_garbled=req.ocr_on_garbled,
        ocr_image_dir=PDF_OCR_IMAGE_DIR,
        ocr_dpi=PDF_OCR_DPI,
    )
    _write_chunks(chunks)
    return {
        "ingested_pdf": len(pdfs),
        "chunks": len(chunks),
        "table_parents": len(parents),
        "path": str(KNOWLEDGE_BASE_PATH),
    }


@app.post("/retrieve")
def retrieve_api(req: RetrieveRequest):
    if not KNOWLEDGE_BASE_PATH.exists():
        raise HTTPException(status_code=400, detail="knowledge_base.json not found. Run ingest first.")
    t0 = time.perf_counter()
    evidence = retrieve_evidence(
        question=req.question,
        chunks_path=KNOWLEDGE_BASE_PATH,
        vector_path=VECTOR_PATH,
        vectorizer_path=VECTORIZER_PATH,
        chroma_dir=CHROMA_DIR,
        store_path=VECTOR_STORE_PATH,
        top_k=req.top_k,
        use_hybrid=req.hybrid,
    )
    t1 = time.perf_counter()
    return {
        "question": req.question,
        "retrieval_mode": "hybrid" if req.hybrid else "keyword",
        "latency_ms": {"retrieve": round((t1 - t0) * 1000, 2)},
        "evidence": evidence,
    }


@app.post("/ask")
def ask_api(req: AskRequest):
    if not KNOWLEDGE_BASE_PATH.exists():
        raise HTTPException(status_code=400, detail="knowledge_base.json not found. Run ingest first.")
    result = run_ask(
        question=req.question,
        chunks_path=KNOWLEDGE_BASE_PATH,
        vector_path=VECTOR_PATH,
        vectorizer_path=VECTORIZER_PATH,
        chroma_dir=CHROMA_DIR,
        store_path=VECTOR_STORE_PATH,
        top_k=req.top_k,
        use_hybrid=req.hybrid,
        model_name=req.model,
        ollama_model=req.ollama_model,
        ollama_url=req.ollama_url,
        max_new_tokens=req.max_new_tokens,
    )
    return result


@app.post("/index")
def index_api():
    if not KNOWLEDGE_BASE_PATH.exists():
        raise HTTPException(status_code=400, detail="knowledge_base.json not found. Run ingest first.")
    chunks = load_chunks(KNOWLEDGE_BASE_PATH)
    build_tfidf_index(chunks, VECTOR_PATH, VECTORIZER_PATH)
    try:
        build_chroma_index(chunks, CHROMA_DIR, VECTOR_STORE_PATH, EMBEDDING_MODEL_NAME)
    except Exception:
        # Keep service usable even when vector deps/models are unavailable.
        pass
    return {"indexed_chunks": len(chunks)}


@app.get("/metrics")
def metrics():
    if not KNOWLEDGE_BASE_PATH.exists():
        return {
            "knowledge_base_exists": False,
            "chunks": 0,
            "by_source": {},
            "by_chunk_type": {},
        }
    chunks = load_chunks(KNOWLEDGE_BASE_PATH)
    by_source = {}
    by_chunk_type = {}
    for c in chunks:
        m = c.get("metadata", {})
        src = m.get("source", "unknown")
        ctype = m.get("chunk_type", "text")
        by_source[src] = by_source.get(src, 0) + 1
        by_chunk_type[ctype] = by_chunk_type.get(ctype, 0) + 1

    return {
        "knowledge_base_exists": True,
        "chunks": len(chunks),
        "by_source": by_source,
        "by_chunk_type": by_chunk_type,
        "tfidf_ready": VECTOR_PATH.exists() and VECTORIZER_PATH.exists(),
        "vector_ready": VECTOR_STORE_PATH.exists(),
    }
