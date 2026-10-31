from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

from langgraph.prebuilt import ToolNode,tools_condition
from ddgs import DDGS
from langchain_core.tools import tool
import requests
import random

from langchain_rag_tool import rag_tool


load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

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


tools = [web_search, rag_tool]
model_with_tools = model.bind_tools(tools)

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state["messages"]

    response = model_with_tools.invoke(messages)

    return {"messages": [response]}

tool_node = ToolNode(tools)

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node('tools', tool_node)

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