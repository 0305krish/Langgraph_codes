from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from langgraph.prebuilt import ToolNode, tools_condition
from ddgs import DDGS
from langchain_core.tools import tool

from backend.langchain_rag_tool import (
    placement_tool,
    clubs_tool,
    fest_tool
)


load_dotenv(r"C:\Users\krish\Desktop\langgraph\MNNIT_CHATBOT\.env", override=True)

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

SYSTEM_PROMPT = """You are a helpful assistant for MNNIT Allahabad students.
You answer questions about college placements, clubs and committees, and the college fest.

Tool usage:
- placement_info: placements, companies, packages, eligibility, process.
- clubs_info: clubs and committees, their activities, members, how to join.
- fest_info: fest events, schedule, registration, competitions.
- web_search: only for live or very recent information the other tools do not have.

Rules:
- Always try the matching college tool first, before answering from your own knowledge.
- Answer only from the retrieved context. If the tools return nothing relevant, say
  "I don't have that information" instead of guessing.
- Never invent numbers, dates, names or packages.
- When the retrieved metadata has a year or source, mention it (for example "as per 2025 data").
- If a question covers more than one topic, call more than one tool.
"""


@tool
def web_search(query: str) -> str:
    """Search the web for current information using DuckDuckGo."""

    try:
        results = DDGS().text(
            query,
            region="us-en",
            max_results=5,
            backend="duckduckgo"
        )

        if not results:
            return "No search results found."

        return "\n\n".join(
            f"Title: {result.get('title', '')}\n"
            f"URL: {result.get('href', '')}\n"
            f"Snippet: {result.get('body', '')}"
            for result in results
        )

    except Exception as e:
        return f"Search failed: {type(e).__name__}: {e}"


tools = [web_search, placement_tool, clubs_tool, fest_tool]
model_with_tools = model.bind_tools(tools)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    # Prepend the system prompt on every call without saving it in the checkpoint
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]

    response = model_with_tools.invoke(messages)

    return {"messages": [response]}


tool_node = ToolNode(tools)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")


# SQLite database for LangGraph checkpoints
conn = sqlite3.connect(
    database="chatbot.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(conn=conn)

# Initialize checkpoint tables
checkpointer.setup()

# Compile graph with persistent checkpointing
chatbot = graph.compile(checkpointer=checkpointer)