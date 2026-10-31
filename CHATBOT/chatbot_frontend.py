import streamlit as st
from langchain_core.messages import AIMessageChunk, HumanMessage
from chatbot_backend import chatbot
import time
import uuid


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Chatbot",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Chatbot")


# ==========================================
# Initialize Session State
# ==========================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}

if "thread_ids" not in st.session_state:
    first_thread_id = str(uuid.uuid4())

    st.session_state.thread_ids = [first_thread_id]

    st.session_state.current_thread_id = first_thread_id

if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = (
        st.session_state.thread_ids[0]
    )

# Initialize history for the current conversation
current_thread_id = st.session_state.current_thread_id

if current_thread_id not in st.session_state.chat_history:
    st.session_state.chat_history[current_thread_id] = []


# ==========================================
# Sidebar
# ==========================================

with st.sidebar:

    st.title("💬 Chat History")

    # New Chat Button
    if st.button("➕ New Chat", use_container_width=True):

        new_thread_id = str(uuid.uuid4())

        st.session_state.thread_ids.insert(0, new_thread_id)

        st.session_state.chat_history[new_thread_id] = []

        st.session_state.current_thread_id = new_thread_id

        st.rerun()

    st.divider()

    st.subheader("Previous Conversations")

    # Display conversation IDs
    for thread_id in st.session_state.thread_ids:

        # Shorten UUID for display
        display_id = thread_id[:8]

        button_label = f"💬 {display_id}"

        # Highlight the active conversation
        if thread_id == st.session_state.current_thread_id:
            button_label = f"🟢 {display_id}"

        if st.button(
            button_label,
            key=f"thread_{thread_id}",
            use_container_width=True
        ):
            st.session_state.current_thread_id = thread_id
            st.rerun()


# ==========================================
# Display Current Conversation
# ==========================================

current_thread_id = st.session_state.current_thread_id

st.caption(f"Thread ID: `{current_thread_id}`")

current_history = st.session_state.chat_history[current_thread_id]

# Display previous messages in the selected conversation
for message in current_history:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==========================================
# Chat Input
# ==========================================

user_message = st.chat_input("Type your message here...")


if user_message:

    # Store user message in the current conversation
    st.session_state.chat_history[current_thread_id].append({
        "role": "user",
        "content": user_message
    })

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_message)

    # LangGraph configuration for this conversation
    config = {
        "configurable": {
            "thread_id": current_thread_id
        }
    }

    # Generate assistant response
    with st.chat_message("assistant"):

        def generate_response():

            for chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(content=user_message)
                    ]
                },
                config=config,
                stream_mode="messages"
            ):

                if not isinstance(chunk, AIMessageChunk):
                    continue

                content = chunk.content

                if isinstance(content, str):

                    if content:
                        time.sleep(0.05)
                        yield content

                elif isinstance(content, list):

                    for block in content:

                        if (
                            isinstance(block, dict)
                            and block.get("type") == "text"
                        ):
                            time.sleep(0.05)
                            yield block["text"]

        ai_response = st.write_stream(generate_response())

    # Store the complete assistant response
    st.session_state.chat_history[current_thread_id].append({
        "role": "assistant",
        "content": ai_response
    })
