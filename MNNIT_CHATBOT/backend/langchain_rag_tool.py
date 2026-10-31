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
load_dotenv(r"C:\Users\krish\Desktop\langgraph\MNNIT_CHATBOT\.env", override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError(f"GOOGLE_API_KEY not found in {BASE_DIR / '.env'}")

CHROMA_PATH = BASE_DIR / "chroma_db"

if not CHROMA_PATH.exists():
    raise FileNotFoundError(
        f"ChromaDB directory not found: {CHROMA_PATH}. "
        "Run langchain_generate_embeddings.py first to create the index."
    )

# This limiter paces query-embedding requests. It does not create document embeddings.
rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.5,
    check_every_n_seconds=0.1,
    max_bucket_size=1,
)

# One shared embedding client for all three tools
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2",
    google_api_key=GOOGLE_API_KEY,
    rate_limiter=rate_limiter,
)


def build_rag_tool(tool_name: str, collection_name: str, description: str, top_k: int = 4):
    """Create one retrieval tool bound to one Chroma collection."""
    store = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH),
    )

    stored_count = store._collection.count()
    print(f"[{tool_name}] collection '{collection_name}': {stored_count} chunks")
    if stored_count == 0:
        # Warn instead of crash, so you can index topics one at a time
        print(f"[{tool_name}] WARNING: collection is empty. Index it with "
              f"langchain_generate_embeddings.py --collection {collection_name}")

    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    @tool(tool_name, description=description)
    def _rag(query: str) -> dict:
        try:
            documents = retriever.invoke(query)
            return {
                "query": query,
                "documents": [
                    {"document": doc.page_content, "metadata": doc.metadata}
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

    return _rag


placement_tool = build_rag_tool(
    "placement_info",
    "placements",
    "Search the college placement data: companies that visited, packages, "
    "branch-wise stats, eligibility criteria and the placement process.",
)

clubs_tool = build_rag_tool(
    "clubs_info",
    "clubs",
    "Search information about college clubs and committees: what each one does, "
    "its members, events, and how to join.",
    top_k=8,  # list-style questions ("which clubs exist?") need more chunks
)

fest_tool = build_rag_tool(
    "fest_info",
    "fest",
    "Search information about the college fest: events, schedule, registration, "
    "competitions, sponsors and contacts.",
)

rag_tools = [placement_tool, clubs_tool, fest_tool]


if __name__ == "__main__":
    tools_by_name = {t.name: t for t in rag_tools}
    print("Tools:", ", ".join(tools_by_name))

    name = input("Which tool? ").strip()
    question = input("Ask a question: ").strip()

    if name in tools_by_name and question:
        result = tools_by_name[name].invoke({"query": question})
        print("\nRAG RESULT")
        print("=" * 60)
        print(result)
    else:
        print("Enter a valid tool name and a query.")