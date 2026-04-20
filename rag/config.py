from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
INDEX_DIR = DATA_DIR / "index"

CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
VECTOR_PATH = INDEX_DIR / "tfidf.npz"
VECTORIZER_PATH = INDEX_DIR / "vectorizer.joblib"
KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.json"
KNOWLEDGE_BASE_DOCX_PATH = DATA_DIR / "knowledge_base_docx.json"
KNOWLEDGE_BASE_PDF_PATH = DATA_DIR / "knowledge_base_pdf.json"
CLEANED_DATA_PATH = DATA_DIR / "cleaned_data.txt"
CLEANED_DOCX_PATH = DATA_DIR / "cleaned_docx.txt"
CLEANED_PDF_PATH = DATA_DIR / "cleaned_pdf.txt"

CHROMA_DIR = INDEX_DIR / "chroma"
VECTOR_STORE_PATH = INDEX_DIR / "vector_store.json"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL_NAME = "qwen3:8b"

DEFAULT_DOCS = [
    BASE_DIR / "精益六西格玛补充方案.docx",
]

DEFAULT_PDFS = [
    BASE_DIR / "精益六西格玛补充方案.pdf",
]

PDF_MMD_DIR = DATA_DIR / "mmd"
TABLE_PARENTS_PATH = DATA_DIR / "table_parents.json"
NOUGAT_CMD = ["python", "-m", "predict"]
# Images rendered for Nougat input
PDF_IMAGE_DIR = DATA_DIR / "pdf_images"
DOCX_IMAGE_DIR = IMAGE_DIR / "docx"
PDF_IMAGE_EXTRACT_DIR = IMAGE_DIR / "pdf"
PDF_OCR_IMAGE_DIR = IMAGE_DIR / "pdf_ocr"
PDF_OCR_DPI = 200
PDF_IMAGE_DPI = 200

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
