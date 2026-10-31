import os
import time
import random
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.rate_limiters import InMemoryRateLimiter

# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError(f"GOOGLE_API_KEY not found in {BASE_DIR / '.env'}")

PDF_PATH = BASE_DIR / "Introduction to Machine Learning with Python ( PDFDrive.com )-min.pdf"
CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "pdf_rag"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 10
MAX_RETRIES = 6
INITIAL_RETRY_DELAY = 20

# Keep False to preserve the existing database.
# Set True only if you intentionally want to delete and rebuild it.
RESET_VECTOR_STORE = False

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=GOOGLE_API_KEY,
    rate_limiter=rate_limiter,
)


def load_pdf(pdf_path: Path):
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    print(f"Loading PDF: {pdf_path.name}")
    documents = PyPDFLoader(str(pdf_path)).load()
    print(f"PDF pages loaded: {len(documents)}")
    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    print(f"Document chunks created: {len(chunks)}")
    return chunks


def create_vector_store():
    if RESET_VECTOR_STORE and CHROMA_PATH.exists():
        import shutil
        print("RESET_VECTOR_STORE=True: deleting existing ChromaDB.")
        shutil.rmtree(CHROMA_PATH)

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH),
    )


def is_retryable_error(exc):
    message = str(exc).lower()
    return any(marker in message for marker in (
        "429", "resource_exhausted", "rate limit", "too many requests",
        "500", "502", "503", "504", "temporarily unavailable",
    ))


def index_documents(vector_store, chunks):
    collection = vector_store._collection
    existing_ids = set()
    count = collection.count()