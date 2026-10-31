from langgraph.graph import StateGraph, START, END
from typing import TypedDict  
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Annotated
import operator
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
model = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash-lite"
)

class chatState(TypedDict):
    messages : Annotated[list[BaseMessage],add_messages]

def chat_node(state : chatState):
    messages = state['messages']

    response = model.invoke(messages)

    return {'messages' : [response]}


graph = StateGraph(chatState)
checkpointer = MemorySaver()

graph.add_node('chat_node', chat_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

