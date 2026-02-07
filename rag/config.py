from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "images"
INDEX_DIR = DATA_DIR / "index"

CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
VECTOR_PATH = INDEX_DIR / "tfidf.npz"
VECTORIZER_PATH = INDEX_DIR / "vectorizer.joblib"
KNOWLEDGE_BASE_PATH = DATA_DIR / "knowledge_base.json"
CLEANED_DATA_PATH = DATA_DIR / "cleaned_data.txt"

CHROMA_DIR = INDEX_DIR / "chroma"
VECTOR_STORE_PATH = INDEX_DIR / "vector_store.json"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

DEFAULT_DOCS = [
    BASE_DIR / "精益六西格玛补充方案.docx",
]

DEFAULT_PDFS = [
    BASE_DIR / "精益六西格玛补充方案.pdf",
]

PDF_MMD_DIR = DATA_DIR / "mmd"
TABLE_PARENTS_PATH = DATA_DIR / "table_parents.json"
NOUGAT_CMD = ["python", "-m", "predict"]
PDF_IMAGE_DIR = DATA_DIR / "pdf_images"
PDF_IMAGE_DPI = 200

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
