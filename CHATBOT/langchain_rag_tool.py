import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.rate_limiters import InMemoryRateLimiter

# --------------------------------------------------
# Configuration: connect to existing ChromaDB only
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError(f"GOOGLE_API_KEY not found in {BASE_DIR / '.env'}")

CHROMA_PATH = BASE_DIR / "chroma_db"
COLLECTION_NAME = "pdf_rag"
TOP_K = 4

if not CHROMA_PATH.exists():
    raise FileNotFoundError(
        f"ChromaDB directory not found: {CHROMA_PATH}. "
        "Run generate_embeddings.py first to create the index."
    )

# This limiter paces query-embedding requests. It does not create document embeddings.
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

# Connecting to the persisted collection does not re-index the PDF.
vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(CHROMA_PATH),
)

stored_count = vector_store._collection.count()
print(f"Connected to existing ChromaDB. Stored chunks: {stored_count}")

if stored_count == 0:
    raise RuntimeError(
        "The ChromaDB collection is empty. Run generate_embeddings.py "
        "when your embedding quota is available."
    )

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": TOP_K},
)


@tool
def rag_tool(query: str) -> dict:
    """Search the already-indexed PDF chunks for information relevant to a query."""

    try:
        documents = retriever.invoke(query)

        return {
            "query": query,
            "documents": [
                {
                    "document": doc.page_content,
                    "metadata": doc.metadata,
                }
                for doc in documents
            ],
        }

    except Exception as exc:
        return {
            "query": query,
            "error": str(exc),
            "message": (
                "RAG retrieval failed. Query embedding still uses Gemini, "
                "so check whether the Gemini embedding quota is available."
            ),
        }


if __name__ == "__main__":
    question = input("Ask a question about the indexed PDF: ").strip()

    if question:
        result = rag_tool.invoke({"query": question})
        print("\nRAG RESULT")
        print("=" * 60)
        print(result)
    else:
        print("Please enter a query.")
