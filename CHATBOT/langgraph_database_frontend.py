import streamlit as st
import sqlite3
import uuid
import time

from datetime import datetime, timezone
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph_database_backend import chatbot


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Chatbot",
    page_icon="💬",
    layout="centered"
)

st.title("💬 Chatbot")


# ==========================================
# SQLITE DATABASE
# ==========================================

DB_PATH = "chatbot.db"


def get_db_connection():
    conn = sqlite3.connect(
        DB_PATH,
        timeout=30
    )

    conn.row_factory = sqlite3.Row

    return conn


def get_timestamp():
    return datetime.now(timezone.utc).isoformat()


def initialize_database():

    with get_db_connection() as conn:

        # Store conversation metadata
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                thread_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Store individual messages
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (thread_id)
                    REFERENCES conversations(thread_id)
                    ON DELETE CASCADE
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_messages_thread_id
            ON messages(thread_id, id)
        """)


initialize_database()


# ==========================================
# DATABASE FUNCTIONS
# ==========================================

def get_all_conversations():

    with get_db_connection() as conn:

        rows = conn.execute("""
            SELECT thread_id, title, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC
        """).fetchall()

        return [dict(row) for row in rows]


def create_conversation():

    thread_id = str(uuid.uuid4())
    timestamp = get_timestamp()

    with get_db_connection() as conn:

        conn.execute("""
            INSERT INTO conversations
                (thread_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
        """, (
            thread_id,
            "New Chat",
            timestamp,
            timestamp
        ))

    return thread_id


def save_message(thread_id, role, content):

    timestamp = get_timestamp()

    with get_db_connection() as conn:

        # Save the message
        conn.execute("""
            INSERT INTO messages
                (thread_id, role, content, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            thread_id,
            role,
            content,
            timestamp
        ))

        # Update conversation timestamp
        conn.execute("""
            UPDATE conversations
            SET updated_at = ?
            WHERE thread_id = ?
        """, (
            timestamp,
            thread_id
        ))

        # Set the title using the first user message
        if role == "user":

            row = conn.execute("""
                SELECT title
                FROM conversations
                WHERE thread_id = ?
            """, (thread_id,)).fetchone()

            if row and row["title"] == "New Chat":

                title = " ".join(content.split())

                if len(title) > 40:
                    title = title[:37] + "..."

                conn.execute("""
                    UPDATE conversations
                    SET title = ?
                    WHERE thread_id = ?
                """, (
                    title or "New Chat",
                    thread_id
                ))


def get_messages(thread_id):

    with get_db_connection() as conn:

        rows = conn.execute("""
            SELECT role, content
            FROM messages
            WHERE thread_id = ?
            ORDER BY id ASC
        """, (thread_id,)).fetchall()

        return [dict(row) for row in rows]


# ==========================================
# SESSION STATE
# ==========================================

# Load conversation IDs from SQLite
if "thread_ids" not in st.session_state:

    conversations = get_all_conversations()

    st.session_state.thread_ids = [
        conversation["thread_id"]
        for conversation in conversations
    ]


# Create the first conversation if the database is empty
if not st.session_state.thread_ids:

    first_thread_id = create_conversation()

    st.session_state.thread_ids = [first_thread_id]


# Select the current conversation
if "current_thread_id" not in st.session_state:

    st.session_state.current_thread_id = (
        st.session_state.thread_ids[0]
    )


# Validate that the selected thread exists
if (
    st.session_state.current_thread_id
    not in st.session_state.thread_ids
):

    st.session_state.current_thread_id = (
        st.session_state.thread_ids[0]
    )


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("💬 Chat History")

    if st.button(
        "➕ New Chat",
        use_container_width=True
    ):

        new_thread_id = create_conversation()

        st.session_state.thread_ids.insert(
            0,
            new_thread_id
        )

        st.session_state.current_thread_id = new_thread_id

        st.rerun()

    st.divider()

    st.subheader("Previous Conversations")

    # Always fetch the latest conversation titles from SQLite
    conversations = get_all_conversations()

    for conversation in conversations:

        thread_id = conversation["thread_id"]
        title = conversation["title"]

        # Identify the selected conversation
        if thread_id == st.session_state.current_thread_id:
            label = f"🟢 {title}"
        else:
            label = f"💬 {title}"

        if st.button(
            label,
            key=f"thread_{thread_id}",
            use_container_width=True
        ):

            st.session_state.current_thread_id = thread_id

            st.rerun()


# ==========================================
# DISPLAY CURRENT CONVERSATION
# ==========================================

current_thread_id = st.session_state.current_thread_id

st.caption(f"Thread ID: `{current_thread_id}`")

# Read messages from SQLite every time the conversation is displayed
current_history = get_messages(current_thread_id)

for message in current_history:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ==========================================
# CHAT INPUT
# ==========================================

user_message = st.chat_input("Type your message here...")


if user_message:

    # LangGraph configuration
    config = {
        "configurable": {
            "thread_id": current_thread_id
        }
    }

    # Save user message to SQLite first
    save_message(
        current_thread_id,
        "user",
        user_message
    )

    # Display user message immediately
    with st.chat_message("user"):
        st.markdown(user_message)

    # Generate the assistant response
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

                            text_content = block.get("text", "")

                            if text_content:
                                time.sleep(0.05)
                                yield text_content

        try:

            ai_response = st.write_stream(
                generate_response()
            )

            # Persist the completed assistant response
            save_message(
                current_thread_id,
                "assistant",
                ai_response
            )

        except Exception as e:

            st.error(f"Error generating response: {e}")
