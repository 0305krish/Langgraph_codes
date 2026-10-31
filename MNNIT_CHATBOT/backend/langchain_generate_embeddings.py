import hashlib
import os
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.rate_limiters import InMemoryRateLimiter


# --------------------------------------------------
# 1. CONFIGURATION: CHANGE THESE VALUES ONLY
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(r"C:\Users\krish\Desktop\langgraph\MNNIT_CHATBOT\.env", override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        f"GOOGLE_API_KEY not found in {BASE_DIR / '.env'}"
    )

# Available collections
VALID_COLLECTIONS = ("placements", "clubs", "fest")

# Select collection through a terminal argument
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--collection",
    choices=VALID_COLLECTIONS,
    default="placements"
)
args = parser.parse_args()

COLLECTION_NAME = args.collection

# Each collection has its own PDF folder
PDF_DIR = BASE_DIR / "pdfs" / COLLECTION_NAME

# Optional year metadata
YEAR = "2026"

# Set True only when you want to delete and rebuild
# the selected collection from scratch
RESET_COLLECTION = False

# ChromaDB storage directory
CHROMA_PATH = BASE_DIR / "chroma_db"


# --------------------------------------------------
# 2. EMBEDDING CONFIGURATION
# --------------------------------------------------

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

BATCH_SIZE = 10
MAX_RETRIES = 6
INITIAL_RETRY_DELAY = 20


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


# --------------------------------------------------
# 3. LOAD PDF FILES
# --------------------------------------------------

def load_pdfs(directory: Path):
    if not directory.is_dir():
        raise NotADirectoryError(
            f"PDF directory does not exist: {directory}"
        )

    # Read PDFs directly inside the specified directory.
    # Use directory.rglob("*.pdf") if you also want
    # to search all subdirectories.
    pdf_files = sorted(directory.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found inside: {directory}"
        )

    all_documents = []

    print(f"\nPDF directory: {directory}")
    print(f"PDF files found: {len(pdf_files)}")

    for pdf_path in pdf_files:
        print(f"\nLoading: {pdf_path.name}")

        documents = PyPDFLoader(str(pdf_path)).load()

        # Preserve the original PDF filename in metadata.
        for document in documents:
            document.metadata["source"] = pdf_path.name

        print(f"Pages loaded: {len(documents)}")

        all_documents.extend(documents)

    print(f"\nTotal PDF pages loaded: {len(all_documents)}")

    return all_documents


# --------------------------------------------------
# 4. SPLIT DOCUMENTS INTO CHUNKS
# --------------------------------------------------

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    return chunks


# --------------------------------------------------
# 5. ADD METADATA
# --------------------------------------------------

def add_metadata(chunks, category, year):
    for chunk in chunks:
        chunk.metadata["category"] = category

        if year:
            chunk.metadata["year"] = str(year)

    return chunks


# --------------------------------------------------
# 6. CREATE OR OPEN VECTOR STORE
# --------------------------------------------------

def create_vector_store(collection_name, reset=False):
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)

    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH),
    )

    if reset:
        print(
            f"\nDeleting collection '{collection_name}' "
            "before rebuilding..."
        )

        store.delete_collection()

        store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_PATH),
        )

    return store


# --------------------------------------------------
# 7. GENERATE STABLE CHUNK IDS
# --------------------------------------------------

def make_chunk_id(chunk):
    """
    Generate a deterministic ID for each chunk.

    Identical source, page, position and content
    produce the same ID on subsequent runs.
    """

    raw = (
        f"{chunk.metadata.get('source', '')}|"
        f"{chunk.metadata.get('page', '')}|"
        f"{chunk.metadata.get('start_index', '')}|"
        f"{chunk.page_content}"
    )

    return hashlib.sha1(
        raw.encode("utf-8")
    ).hexdigest()


# --------------------------------------------------
# 8. HANDLE RETRYABLE API ERRORS
# --------------------------------------------------

def is_retryable_error(exc):
    message = str(exc).lower()

    return any(
        marker in message
        for marker in (
            "429",
            "resource_exhausted",
            "rate limit",
            "too many requests",
            "500",
            "502",
            "503",
            "504",
            "temporarily unavailable",
        )
    )


def add_batch_with_retry(vector_store, batch, batch_ids):
    delay = INITIAL_RETRY_DELAY

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            vector_store.add_documents(
                documents=batch,
                ids=batch_ids,
            )

            return

        except Exception as exc:
            if (
                attempt == MAX_RETRIES
                or not is_retryable_error(exc)
            ):
                raise

            wait = delay + random.uniform(0, 5)

            print(
                f"Retryable error: {exc.__class__.__name__}. "
                f"Attempt {attempt}/{MAX_RETRIES}. "
                f"Waiting {wait:.0f} seconds..."
            )

            time.sleep(wait)

            delay *= 2


# --------------------------------------------------
# 9. INCREMENTAL INDEXING
# --------------------------------------------------

def index_documents(vector_store, chunks):
    collection = vector_store._collection

    # Retrieve existing chunk IDs.
    existing_count = collection.count()

    existing_ids = set()

    if existing_count > 0:
        existing_ids = set(
            collection.get(include=[])["ids"]
        )

    print(
        f"\nChunks already stored: {existing_count}"
    )

    # Generate IDs for all new chunks.
    chunk_ids = [
        make_chunk_id(chunk)
        for chunk in chunks
    ]

    # Remove previously indexed chunks and duplicates.
    pending_chunks = []
    pending_ids = []
    seen_ids = set()

    for chunk, chunk_id in zip(chunks, chunk_ids):

        if (
            chunk_id in existing_ids
            or chunk_id in seen_ids
        ):
            continue

        seen_ids.add(chunk_id)

        pending_chunks.append(chunk)
        pending_ids.append(chunk_id)

    skipped = len(chunks) - len(pending_chunks)

    print(f"Chunks to embed: {len(pending_chunks)}")
    print(f"Chunks skipped: {skipped}")

    if not pending_chunks:
        print(
            "\nNothing to do. "
            "All chunks are already indexed."
        )
        return

    total_batches = (
        len(pending_chunks) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    # Process the remaining chunks in batches.
    for i in range(0, len(pending_chunks), BATCH_SIZE):

        batch = pending_chunks[
            i:i + BATCH_SIZE
        ]

        batch_ids = pending_ids[
            i:i + BATCH_SIZE
        ]

        batch_number = i // BATCH_SIZE + 1

        print(
            f"\nProcessing batch "
            f"{batch_number}/{total_batches}..."
        )

        add_batch_with_retry(
            vector_store,
            batch,
            batch_ids,
        )

    print(
        f"\nIndexing complete. "
        f"Total chunks stored: {collection.count()}"
    )


# --------------------------------------------------
# 10. MAIN FUNCTION
# --------------------------------------------------

def main():

    print("=" * 50)
    print("PDF EMBEDDING GENERATOR")
    print("=" * 50)

    print(f"Input directory: {PDF_DIR}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"ChromaDB directory: {CHROMA_PATH}")

    # Step 1: Load all PDFs.
    documents = load_pdfs(PDF_DIR)

    # Step 2: Split PDFs into chunks.
    chunks = split_documents(documents)

    # Step 3: Attach metadata.
    chunks = add_metadata(
        chunks,
        COLLECTION_NAME,
        YEAR,
    )

    # Step 4: Open or create the collection.
    vector_store = create_vector_store(
        COLLECTION_NAME,
        reset=RESET_COLLECTION,
    )

    # Step 5: Generate embeddings only for new chunks.
    index_documents(
        vector_store,
        chunks,
    )

    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    main()