import streamlit as st
import pandas as pd
import requests
import json
import time
import re
from datetime import datetime
import os
from dotenv import load_dotenv
import sys
# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import Realtime functionality
from chat_realtime import initialize_realtime_chat, cleanup_realtime as cleanup_chat_realtime
from database.vector_store_realtime import initialize_vector_store_realtime, cleanup_realtime_subscriptions
import streamlit_antd_components as sac
from authentication.auth_forms import auth_forms, handle_auth_callback, init_auth_session# Import directly from flameaudio.py - no fallback needed
from pages.flameaudio import load_agents, AUTH_ENABLED

# Initialize authentication session and handle callback
init_auth_session()
handle_auth_callback()
# Load environment variables
load_dotenv()

# Set up backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Check if authentication is enabled
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() == "true"

# Page configuration
st.set_page_config(
    page_title="Flame Audio AI: Chat",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.title("Flame Audio AI: Chat")

# Navigation menu (always visible)
with st.sidebar:
    sac.menu([
        sac.MenuItem('Home', icon='house-fill', href='/flamehome'),
	    sac.MenuItem('Playground', icon='mic-fill', href='/flameaudio'),
        sac.MenuItem('Agents', icon='person-fill', href='/agents'),
        sac.MenuItem('Documents', icon='file-text-fill', href='/documents'),
        sac.MenuItem('Chat', icon='chat-fill'),
        sac.MenuItem('MCP', icon='gear-fill', href='/flame_mcp'),
        sac.MenuItem('MCP Chat', icon='chat-dots-fill', href='/mcp_chat'),
    ], open_all=True)

# Show authentication forms if not authenticated
if AUTH_ENABLED and not st.session_state.get("authenticated", False):
    with st.sidebar:
        auth_forms()

# Refactored to use st.chat_message
st.title("Flame Audio AI: Chat")

# Initialize session state for chat agent selection if not present
if "chat_agent" not in st.session_state:
    st.session_state.chat_agent = ""

# Chat sessions container in the sidebar
st.sidebar.subheader("Chat Sessions")

# Add a note about session filtering
if st.session_state.get('chat_document'):
    doc_name = ""
    if "chat_documents" in st.session_state:
        for doc in st.session_state.chat_documents:
            if doc.get("id") == st.session_state.get('chat_document'):
                doc_name = doc.get("document_name", "selected document")
                break

    if doc_name:
        st.sidebar.caption(f"Showing sessions for document: **{doc_name}**")
    st.sidebar.caption("Sessions are specific to both the selected agent and document.")

# Only show the New Chat button and sessions if an agent is selected
if st.session_state.get("chat_agent"):
    # Create a New Chat button at the top level in the sidebar
    if st.sidebar.button("New Chat", key="new_chat_btn", use_container_width=True):
        # Clean up any existing Realtime subscription
        cleanup_chat_realtime()

        # Clear chat history and session ID
        st.session_state.chat_history = []
        st.session_state.chat_session_id = None

        # Reset document selection to match agent's documents
        if "chat_documents" in st.session_state:
            del st.session_state.chat_documents

        # Rerun to update UI
        st.rerun()

    # Display session reload button
    if st.sidebar.button("🔄 Reload Sessions", key="reload_sessions_btn", use_container_width=True):
        st.sidebar.success("Reloading chat sessions...")
        st.session_state.sessions_reload_requested = True

    # Check if sessions reload was requested
    if st.session_state.get("sessions_reload_requested", False):
        # Clear the flag first
        st.session_state.sessions_reload_requested = False
        # Use experimental_rerun for safer page refresh
        st.experimental_rerun()

    # Create a scrollable container for all chat sessions
    sessions_container = st.sidebar.container(height=400, border=True)

    # Function to load chat sessions for the current agent and document
    def load_chat_sessions():
        try:
            # Build the URL with query parameters for the agent_id
            url = f"{BACKEND_URL}/chat/sessions?agent_id={st.session_state.get('chat_agent', '')}"

            response = requests.get(
                url,
                headers={
                    "Authorization": f"Bearer {st.session_state.get('_auth_token_', '')}"
                },
                timeout=10
            )

            if response.status_code == 200:
                sessions = response.json()

                # If a document is selected, filter the sessions in the frontend
                if st.session_state.get('chat_document'):
                    document_id = st.session_state.get('chat_document')
                    # Filter sessions to only include those with the matching document_id
                    sessions = [session for session in sessions if session.get('document_id') == document_id]

                return sessions
            else:
                st.sidebar.error(f"Failed to fetch chat sessions: {response.text}")
                return []
        except Exception as e:
            st.sidebar.error(f"Error fetching chat sessions: {str(e)}")
            return []

    # Function to load a specific chat session
    def load_chat_session(session_id):
        try:
            # Clean up any existing Realtime subscription
            cleanup_chat_realtime()

            response = requests.get(
                f"{BACKEND_URL}/chat/sessions/{session_id}",
                headers={
                    "Authorization": f"Bearer {st.session_state.get('_auth_token_', '')}"
                },
                timeout=10
            )

            if response.status_code == 200:
                session_data = response.json()
                st.session_state.chat_history = session_data.get("messages", [])

                # Initialize Realtime subscription for this session
                if initialize_realtime_chat(agent_id=st.session_state.chat_agent, session_id=session_id):
                    st.toast("Realtime chat updates enabled", icon="")

                return session_data
            else:
                st.sidebar.error(f"Failed to fetch chat session: {response.text}")
                return None
        except Exception as e:
            st.sidebar.error(f"Error fetching chat session: {str(e)}")
            return None

    # Load and display existing chat sessions
    with sessions_container:
        sessions = load_chat_sessions()

        if not sessions:
            if st.session_state.get('chat_document'):
                # Get document name for better user feedback
                doc_name = ""
                if "chat_documents" in st.session_state:
                    for doc in st.session_state.chat_documents:
                        if doc.get("id") == st.session_state.get('chat_document'):
                            doc_name = doc.get("document_name", "selected document")
                            break

                if doc_name:
                    st.info(f"No chat sessions found for agent with document: {doc_name}")
                    st.caption("Sessions are specific to both the selected agent and document.")
                else:
                    st.info("No chat sessions found for this agent with the selected document.")
                    st.caption("Sessions are specific to both the selected agent and document.")
            else:
                st.info("No chat sessions found for this agent.")

        # Display all sessions inside the scrollable container
        for session in sessions:
            session_title = session.get("title", "Untitled Chat")
            session_id = session.get("id")
            doc_name = session.get("document_name", "")
            created_at = datetime.fromisoformat(session.get("created_at", datetime.now().isoformat()).replace("Z", "+00:00"))
            formatted_date = created_at.strftime("%Y-%m-%d %H:%M")

            # Create a container for each session
            session_container = st.container(border=True)
            with session_container:
                # Session title and metadata
                st.write(f"**{session_title}**")
                st.caption(f"Created: {formatted_date}")
                if doc_name:
                    st.caption(f"Document: {doc_name}")

                # Actions for this session
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Load", key=f"load_{session_id}"):
                        load_chat_session(session_id)
                        st.session_state.chat_session_id = session_id
                        st.rerun()
                with col2:
                    if st.button("Delete", key=f"delete_{session_id}"):
                        try:
                            response = requests.delete(
                                f"{BACKEND_URL}/chat/sessions/{session_id}",
                                headers={
                                    "Authorization": f"Bearer {st.session_state.get('_auth_token_', '')}"
                                },
                                timeout=10
                            )

                            if response.status_code == 200:
                                if st.session_state.chat_session_id == session_id:
                                    st.session_state.chat_history = []
                                    st.session_state.chat_session_id = None
                                st.rerun()
                            else:
                                st.sidebar.error(f"Failed to delete session: {response.text}")
                        except Exception as e:
                            st.sidebar.error(f"Error deleting session: {str(e)}")
else:
    # If no agent is selected, show a message
    st.sidebar.info("Select an agent in the settings below to see your chat sessions.")

# Sidebar for agent and model selection


    # Initialize the tab selection in session state if not present
with st.sidebar:
    if "chat_config_tab" not in st.session_state:
        st.session_state.chat_config_tab = "Agent config"

    # Create a container for the chat settings
    st.markdown("## ⚙️ Chat Settings")
    chat_settings_container = st.container(border=True)

    with chat_settings_container:
        # Create radio buttons with horizontal layout
        config_tab = st.radio(
            "Configuration",
            ["Agent config", "Model config", "Vector store config"],
            index=0 if st.session_state.chat_config_tab == "Agent config" else 1 if st.session_state.chat_config_tab == "Model config" else 2,
            key="chat_sidebar_tabs",
            horizontal=True
        )

        # Update the session state
        st.session_state.chat_config_tab = config_tab

        # Create a scrollable container for all settings
        settings_container = st.container(height=500, border=False)

        with settings_container:
            # Agent Config Section
            if config_tab == "Agent config":
                # Agent Selection
                st.subheader("Agent Selection")
                if not AUTH_ENABLED or st.session_state.get("authenticated", False):
                    # Load agents when page loads
                    def load_agents():
                        """Load agents from the backend"""
                        try:
                            response = requests.get(
                                f"{BACKEND_URL}/db/agents",
                                headers={
                                    "Authorization": f"Bearer {st.session_state.get('_auth_token_', '')}"
                                },
                                timeout=10
                            )

                            if response.status_code == 200:
                                return response.json()
                            else:
                                st.error(f"Failed to load agents: {response.text}")
                                return []
                        except Exception as e:
                            st.error(f"Error loading agents: {str(e)}")
                            return []

                    # Fetch agents for the current user
                    if "agents" not in st.session_state or st.button("🔄 Refresh Agents", key="refresh_chat_agents"):
                        with st.spinner("Loading agents..."):
                            st.session_state.agents = load_agents()

                    # Display agent selection dropdown
                    agent_options = [{"label": "-- Select an Agent --", "value": ""}]
                    if "agents" in st.session_state and st.session_state.agents:
                        for agent in st.session_state.agents:
                            agent_options.append({
                                "label": agent.get("name", "Unnamed Agent"),
                                "value": agent.get("id", ""),
                                "system_prompt": agent.get("system_message", "")
                            })

                    # Set default chat agent if not already set
                    if "chat_agent" not in st.session_state:
                        st.session_state.chat_agent = ""

                    # Agent selection
                    selected_agent = st.selectbox(
                        "Select Agent",
                        options=[a["value"] for a in agent_options],
                        format_func=lambda x: next((a["label"] for a in agent_options if a["value"] == x), x),
                        index=next((i for i, a in enumerate(agent_options) if a["value"] == st.session_state.chat_agent), 0),
                        key="chat_agent_selector"
                    )

                    # Update selected agent
                    if selected_agent != st.session_state.chat_agent:
                        st.session_state.chat_agent = selected_agent
                        # Reset document selection
                        if "chat_document" in st.session_state:
                            st.session_state.chat_document = ""
                        # Force refresh of documents
                        if "chat_documents" in st.session_state:
                            del st.session_state.chat_documents
                else:
                    st.info("Please sign in to access agent settings")

                # Model Selection
                st.subheader("Model Selection")
                if not AUTH_ENABLED or st.session_state.get("authenticated", False):
                    # Fetch models from the backend
                    if "chat_models" not in st.session_state or st.button("🔄 Refresh Models", key="refresh_chat_models"):
                        with st.spinner("Loading chat models..."):
                            try:
                                response = requests.get(
                                    f"{BACKEND_URL}/chat/models",
                                    headers={
                                        "Authorization": f"Bearer {st.session_state.get('_auth_token_', '')}"
                                    },
                                    timeout=10
                                )

                                if response.status_code == 200:
                                    st.session_state.chat_models = response.json()
                                else:
                                    st.error(f"Failed to fetch chat models: {response.text}")
                                    st.session_state.chat_models = []
                            except Exception as e:
                                st.error(f"Error fetching chat models: {str(e)}")
                                st.session_state.chat_models = []

                    # Display model selection dropdown
                    model_options = ["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"]  # Default options
                    model_descriptions = {}

                    if "chat_models" in st.session_state and st.session_state.chat_models:
                        model_options = []
                        for model in st.session_state.chat_models:
                            model_id = model.get("id")
                            model_options.append(model_id)
                            model_descriptions[model_id] = model.get("description", "No description available")

                    # Set default model if not already set
                    if "chat_model" not in st.session_state and model_options:
                        st.session_state.chat_model = model_options[0] if model_options else ""

                    # Model selection
                    selected_model = st.selectbox(
                        "Select Model",
                        options=model_options,
                        index=model_options.index(st.session_state.chat_model) if st.session_state.chat_model in model_options else 0,
                        key="chat_model_selector"
                    )

                    # Update selected model
                    if selected_model != st.session_state.chat_model:
                        st.session_state.chat_model = selected_model

                    # Display model description if available
                    if selected_model in model_descriptions:
                        st.info(model_descriptions[selected_model])
                else:
                    st.info("Please sign in to access model settings")

                # Document Selection
                st.subheader("Document Selection")
                if not AUTH_ENABLED or st.session_state.get("authenticated", False):
                    # Only show document selection if an agent is selected
                    if st.session_state.chat_agent:
                        # Fetch documents for the current agent
                        if "chat_documents" not in st.session_state or st.button("🔄 Refresh Documents", key="refresh_chat_documents"):
                            with st.spinner("Loading documents..."):
                                try:
                                    response = requests.get(
                                        f"{BACKEND_URL}/db/documents?agent_id={st.session_state.chat_agent}",
                                        headers={
                                            "Authorization": f"Bearer {st.session_state.get('_auth_token_', '')}"
                                        },
                                        timeout=30  # Increased timeout from 10 to 30 seconds
                                    )

                                    if response.status_code == 200:
                                        st.session_state.chat_documents = response.json()
                                    else:
                                        st.error(f"Failed to fetch documents: {response.text}")
                                        st.session_state.chat_documents = []
                                except Exception as e:
                                    st.error(f"Error fetching documents: {str(e)}")
                                    st.session_state.chat_documents = []

                        # Display document selection dropdown
                        doc_options = []
                        if "chat_documents" in st.session_state and st.session_state.chat_documents:
                            for doc in st.session_state.chat_documents:
                                doc_options.append({
                                    "label": doc.get("document_name", "Unnamed Document"),
                                    "value": doc.get("id", "")
                                })

                        # Set default document if not already set
                        if "chat_document" not in st.session_state:
                            st.session_state.chat_document = ""

                        # Document selection
                        if doc_options:  # Only show dropdown if there are documents
                            selected_doc = st.selectbox(
                                "Select Document",
                                options=[d["value"] for d in doc_options],
                                format_func=lambda x: next((d["label"] for d in doc_options if d["value"] == x), x),
                                index=next((i for i, d in enumerate(doc_options) if d["value"] == st.session_state.chat_document), 0) if st.session_state.chat_document in [d["value"] for d in doc_options] else 0,
                                key="chat_document_selector"
                            )
                        else:
                            # No documents available
                            selected_doc = ""
                            st.session_state.chat_document = ""

                        # Update selected document
                        if selected_doc != st.session_state.chat_document:
                            st.session_state.chat_document = selected_doc

                        # Show info about document selection
                        if doc_options and selected_doc:
                            doc_name = next((d["label"] for d in doc_options if d["value"] == selected_doc), "Selected Document")
                            st.success(f"The agent will focus on document: {doc_name}")
                        elif not doc_options:
                            st.warning("No documents available for this agent. Please add documents first.")
                    else:
                        st.warning("Please select an agent first to view associated documents.")
                else:
                    st.info("Please sign in to access document settings")

            # Model Config Section
            elif config_tab == "Model config":
                # System Prompt Settings
                st.subheader("System Prompt")
                if not AUTH_ENABLED or st.session_state.get("authenticated", False):
                    # Get system prompt from selected agent
                    system_prompt = ""
                    if st.session_state.chat_agent and "agents" in st.session_state:
                        for agent in st.session_state.agents:
                            if agent.get("id") == st.session_state.chat_agent:
                                system_prompt = agent.get("system_message", "")
                                break

                    # Allow editing system prompt
                    edited_system_prompt = st.text_area(
                        "System Prompt",
                        value=system_prompt,
                        height=200,
                        key="chat_system_prompt"
                    )

                    # Show warning if system prompt was modified
                    if edited_system_prompt != system_prompt:
                        st.warning("System prompt has been modified from the agent's default. This change will only apply to this chat session.")
                else:
                    st.info("Please sign in to access system prompt settings")

                # Model Parameters
                st.subheader("Model Parameters")
                if not AUTH_ENABLED or st.session_state.get("authenticated", False):
                    # Initialize parameters with defaults if not set
                    if "chat_parameters" not in st.session_state:
                        st.session_state.chat_parameters = {
                            "temperature": 0.7,
                            "max_tokens": 1024,
                            "top_p": 0.9
                        }

                    # Temperature slider
                    temperature = st.slider(
                        "Temperature",
                        min_value=0.0,
                        max_value=1.0,
                        value=st.session_state.chat_parameters.get("temperature", 0.7),
                        step=0.1,
                        format="%.1f",
                        help="Higher values (closer to 1) make output more random, lower values make it more deterministic"
                    )

                    # Max tokens slider
                    max_tokens = st.slider(
                        "Max Output Tokens",
                        min_value=128,
                        max_value=4096,
                        value=st.session_state.chat_parameters.get("max_tokens", 1024),
                        step=128,
                        help="Maximum number of tokens to generate in the response"
                    )

                    # Top-p slider
                    top_p = st.slider(
                        "Top P",
                        min_value=0.1,
                        max_value=1.0,
                        value=st.session_state.chat_parameters.get("top_p", 0.9),
                        step=0.1,
                        format="%.1f",
                        help="Controls diversity of generated text. Lower values generate more focused text."
                    )

                    # Update parameters in session state
                    st.session_state.chat_parameters = {
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "top_p": top_p
                    }
                else:
                    st.info("Please sign in to adjust model parameters")

            # Vector Store Config Section
            elif config_tab == "Vector store config":
                st.subheader("Vector Store Configuration")
                if not AUTH_ENABLED or st.session_state.get("authenticated", False):
                    try:
                        # Fetch vector store settings from backend
                        if "vector_store_settings" not in st.session_state or st.button("🔄 Refresh Vector Store Settings", key="refresh_vector_settings"):
                            with st.spinner("Loading vector store settings..."):
                                response = requests.get(
                                    f"{BACKEND_URL}/vectorstore/settings",
                                    headers={
                                        "Authorization": f"Bearer {st.session_state.get('_auth_token_', '')}"
                                    },
                                    timeout=10
                                )

                                if response.status_code == 200:
                                    st.session_state.vector_store_settings = response.json()

                                    # Initialize real-time updates if authenticated
                                    if "user" in st.session_state and "id" in st.session_state.user:
                                        if initialize_vector_store_realtime(st.session_state.user["id"]):
                                            st.toast("Real-time vector store updates enabled", icon="")
                                else:
                                    st.error(f"Failed to load vector store settings: {response.text}")
                    except Exception as e:
                        st.error(f"Error loading vector store settings: {str(e)}")

                    # Get settings from session state or use defaults
                    settings = {}
                    available_models = []

                    if "vector_store_settings" in st.session_state:
                        settings = st.session_state.vector_store_settings.get("settings", {})
                        available_models = st.session_state.vector_store_settings.get("available_models", [])

                    # Create columns for better layout
                    vs_col1, vs_col2 = st.columns(2)

                    with vs_col1:
                        # Embedding model selection
                        st.subheader("Embedding Model")

                        model_options = [model.get("id") for model in available_models]
                        model_names = {model.get("id"): model.get("name") for model in available_models}
                        model_dimensions = {model.get("id"): model.get("dimensions") for model in available_models}

                        if model_options:
                            embedding_model = st.selectbox(
                                "Model",
                                options=model_options,
                                format_func=lambda x: f"{model_names.get(x)} ({model_dimensions.get(x)} dimensions)",
                                index=model_options.index(settings.get("embedding_model", "all-MiniLM-L6-v2")) if settings.get("embedding_model") in model_options else 0,
                                key="vector_embedding_model"
                            )
                        else:
                            embedding_model = st.selectbox(
                                "Model",
                                ["all-MiniLM-L6-v2"],
                                key="vector_embedding_model_fallback"
                            )
                            st.caption("Failed to load models from API. Using default model.")

                        # Add collection information - fetch from API
                        st.subheader("Storage Collection")

                        # Fetch collection info from backend
                        if "vector_collections" not in st.session_state or st.button("🔄 Refresh Vector Collections", key="refresh_vector_collections"):
                            try:
                                with st.spinner("Loading collection information..."):
                                    response = requests.get(
                                        f"{BACKEND_URL}/vectorstore/collections",
                                        headers={
                                            "Authorization": f"Bearer {st.session_state.get('_auth_token_', '')}"
                                        },
                                        timeout=10
                                    )

                                    if response.status_code == 200:
                                        st.session_state.vector_collections = response.json()
                                    else:
                                        st.error(f"Failed to load collection information: {response.text}")
                                        if "vector_collections" not in st.session_state:
                                            st.session_state.vector_collections = {
                                                "status": "error",
                                                "collections": [],
                                                "primary_collection": "transcription_embeddings"
                                            }
                            except Exception as e:
                                st.error(f"Error loading collection information: {str(e)}")
                                if "vector_collections" not in st.session_state:
                                    st.session_state.vector_collections = {
                                        "status": "error",
                                        "collections": [],
                                        "primary_collection": "transcription_embeddings"
                                    }

                        # Display collection information
                        collections_info = st.session_state.get("vector_collections", {})
                        primary_collection = collections_info.get("primary_collection", "transcription_embeddings")
                        collections = collections_info.get("collections", [])

                        # Find the primary collection in the list
                        primary_info = None
                        for collection in collections:
                            if collection.get("name") == primary_collection:
                                primary_info = collection
                                break

                        # Display basic information
                        st.info(f"Documents are stored in the Qdrant collection: **{primary_collection}**")

                        # Display collection stats if available
                        if primary_info:
                            status = primary_info.get("status", "")
                            points_count = primary_info.get("points_count", 0)
                            indexed_count = primary_info.get("indexed_vectors_count", 0)

                            if status == "active":
                                stats_color = "success" if indexed_count > 0 else "warning"
                                stats_message = f"Status: Active | Documents: {points_count} | Indexed: {indexed_count}"
                                if stats_color == "success":
                                    st.success(stats_message)
                                else:
                                    st.warning(stats_message)
                            else:
                                st.warning(f"Collection status: {status}")

                        # Add help text about collections
                        collection_help = """This is the collection used for document storage.
                            All vector embeddings and document metadata are stored here for efficient retrieval."""
                        st.caption(collection_help)

                    with vs_col2:
                        # Similarity search configuration
                        st.subheader("Search Configuration")

                        similarity_threshold = st.slider(
                            "Similarity Threshold",
                            min_value=0.0,
                            max_value=1.0,
                            value=float(settings.get("similarity_threshold", 0.2)),
                            step=0.05,
                            format="%.2f",
                            key="vector_similarity_threshold",
                            help="Lower values return more results but may be less relevant. Higher values are more strict."
                        )

                        match_count = st.slider(
                            "Result Limit",
                            min_value=1,
                            max_value=50,
                            value=int(settings.get("match_count", 10)),
                            step=1,
                            key="vector_match_count",
                            help="Maximum number of results to return from vector search"
                        )

                    # Text chunking configuration
                    st.subheader("Text Chunking")
                    chunk_size = st.slider(
                        "Chunk Size",
                        min_value=100,
                        max_value=2000,
                        value=int(settings.get("chunk_size", 1000)),
                        step=100,
                        key="vector_chunk_size",
                        help="Size of text chunks for embedding (larger chunks provide more context but less precision)"
                    )

                    chunk_overlap = st.slider(
                        "Chunk Overlap",
                        min_value=0,
                        max_value=500,
                        value=int(settings.get("chunk_overlap", 200)),
                        step=50,
                        key="vector_chunk_overlap",
                        help="Overlap between chunks to ensure context continuity"
                    )

                    # Save button
                    if st.button("Save Vector Store Settings", key="save_vector_settings", type="primary"):
                        try:
                            # Prepare settings payload
                            vector_settings = {
                                "embedding_model": embedding_model,
                                "dimension": model_dimensions.get(embedding_model, 384),
                                "similarity_threshold": similarity_threshold,
                                "chunk_size": chunk_size,
                                "chunk_overlap": chunk_overlap,
                                "match_count": match_count,
                                "enabled": True
                            }

                            # Save settings to backend
                            response = requests.post(
                                f"{BACKEND_URL}/vectorstore/settings",
                                json=vector_settings,
                                headers={
                                    "Authorization": f"Bearer {st.session_state.get('_auth_token_', '')}"
                                },
                                timeout=10
                            )

                            if response.status_code == 200:
                                st.success("Vector store settings saved successfully!")
                                # Update session state
                                st.session_state.vector_store_settings = {
                                    "settings": vector_settings,
                                    "available_models": available_models
                                }
                            else:
                                st.error(f"Failed to save vector store settings: {response.text}")
                        except Exception as e:
                            st.error(f"Error saving vector store settings: {str(e)}")

                    # Add information about the vector store
                    with st.expander("About Vector Embeddings"):
                        st.markdown("""
                        **Vector Embeddings** are numerical representations of text that capture semantic meaning.
                        When similar text is converted to vectors, they will be closer in the vector space.

                        **Key Settings:**
                        - **Model**: Different models create different quality embeddings with varying dimensions
                        - **Similarity Threshold**: How similar embeddings need to be (lower = more results)
                        - **Chunk Size**: How large each piece of text should be when embedded
                        - **Chunk Overlap**: How much chunks should overlap to maintain context
                        """)



    # User Profile Container
    st.sidebar.markdown("## User Profile")
    user_profile_container = st.sidebar.container(border=True)
    with user_profile_container:
        if AUTH_ENABLED and st.session_state.get("authenticated", False) and "user" in st.session_state:
            email = st.session_state['user'].get('email', '')
            st.markdown(f"**Signed in as:**")
            st.info(email)
            if st.button("Sign Out", key="sign_out_btn", use_container_width=True):
                # Use the proper logout function from auth_forms.py
                from authentication.auth_forms import logout
                logout()
                # No need for sign_out_requested flag as logout() handles everything

with st.sidebar:
    with st.container(border=True):
        st.subheader("Connect with us")
        sac.buttons([
            sac.ButtonsItem(label='About FlameheadLabs', icon='info-circle', href='http://flameheadlabs.tech/'),
            sac.ButtonsItem(label='Give 5 stars on Github', icon='github', href='https://github.com/Flamehead-Labs-Ug/flame-audio'),
            sac.ButtonsItem(label='Follow on X', icon='twitter', href='https://x.com/flameheadlabsug'),
            sac.ButtonsItem(label='Follow on Linkedin', icon='linkedin', href='https://www.linkedin.com/in/flamehead-labs-919910285'),
            sac.ButtonsItem(label='Email', icon='mail', href='mailto:Flameheadlabs256@gmail.com'),
        ],
        label='',
        align='center')
# Main content area
if not AUTH_ENABLED or st.session_state.get("authenticated", False):
    # Initialize session state for chat history if not already there
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = None

    # Ensure chat parameters are initialized
    if "chat_parameters" not in st.session_state:
        st.session_state.chat_parameters = {
            "temperature": 0.7,
            "max_tokens": 1024,
            "top_p": 0.9
        }

    # Function to send a chat message
    def send_message(message, session_id=None):
        # Prepare data for the request
        chat_data = {
            "message": message, # Send the original message
            "agent_id": st.session_state.chat_agent,
            "parameters": st.session_state.get("chat_parameters", {}),
            "metadata": {} # Initialize metadata
        }



        # Add session_id if provided
        if session_id:
            chat_data["session_id"] = session_id

        # Add document context directly to metadata if selected
        if st.session_state.chat_document:
            chat_data["document_id"] = st.session_state.chat_document # Still needed for session association
            doc_name = ""
            if "chat_documents" in st.session_state:
                 for doc in st.session_state.chat_documents:
                     if doc.get("id") == st.session_state.chat_document:
                         doc_name = doc.get("document_name", "")
                         break
            # Pass context info in metadata for backend processing
            chat_data["metadata"]["document_context"] = {
                "job_id": st.session_state.chat_document,
                "document_name": doc_name
            }
            # Add search hints from the original message if present
            # (These were previously appended to enhanced_message)
            import re
            quoted_phrases = re.findall(r'"([^"]*)"', message) # Check original message
            search_command_match_find = re.search(r'(!find|!search)\s+([^\[\]]+)', message.lower())
            search_command_match_exact = re.search(r'(!exact)\s+([^\[\]]+)', message.lower())

            search_hints = {}
            if search_command_match_find:
                 search_term = search_command_match_find.group(2).strip()
                 if search_term:
                     search_hints = {"direct_search": search_term, "bypass_vector": True, "raw_text_search": True}
            elif search_command_match_exact:
                 search_term = search_command_match_exact.group(2).strip()
                 if search_term:
                     search_hints = {"exact_match": search_term, "bypass_vector": True, "raw_text_search": True, "force_exact": True}
            elif quoted_phrases:
                 valid_phrases = [p for p in quoted_phrases if len(p) > 2]
                 if valid_phrases:
                     # Send list of phrases, backend needs to handle multiple if necessary
                     search_hints = {"exact_phrase": valid_phrases[0] if valid_phrases else "", "bypass_vector": True}
            elif any(term in message.lower() for term in ["where", "find", "search", "locate", "mention", "say", "said", "talk", "spoke"]):
                 search_terms = [word for word in message.split() if len(word) > 3 and word.lower() not in ["where", "find", "search", "locate", "mention", "about", "does", "said", "talk", "spoke"]]
                 if search_terms:
                     search_hints = {"text_search": ' '.join(search_terms)}

            if search_hints:
                 chat_data["metadata"]["search_hints"] = search_hints


        # Handle custom system prompt override (keep this logic)
        system_prompt = st.session_state.get("chat_system_prompt", "")
        original_prompt = ""
        if "agents" in st.session_state:
            for agent in st.session_state.agents:
                if agent.get("id") == st.session_state.chat_agent:
                    original_prompt = agent.get("system_message", "")
                    break
        # If system prompt is modified (and not empty), send it in metadata
        if system_prompt and system_prompt != original_prompt:
             chat_data["metadata"]["custom_system_prompt"] = system_prompt
        # Note: Backend will now be responsible for adding document context to the system prompt if needed
        # User message is now added to history *before* calling send_message
        # st.session_state.chat_history.append({"role": "user", "content": message}) # Removed this line

        # Send request to backend
        try:
            with st.spinner("Agent is thinking..."):
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json=chat_data,
                    headers={
                        "Authorization": f"Bearer {st.session_state.get('_auth_token_', '')}"
                    },
                    timeout=60  # Longer timeout for chat responses
                )

                if response.status_code == 200:
                    response_data = response.json()
                    # Update session ID if this was a new session
                    st.session_state.chat_session_id = response_data.get("session_id")

                    # Get the assistant's response
                    assistant_message = response_data.get("response")
                    assistant_metadata = response_data.get("metadata", {})

                    # Add assistant response to local chat history for immediate display
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": assistant_message,
                        "metadata": assistant_metadata
                    })

                    # Fetch complete chat history from backend to ensure consistency - REMOVED
                    # Calling this overwrites the local history with the backend version,
                    # which includes the enhanced user message. We rely on local updates now.
                    # load_chat_session(st.session_state.chat_session_id) # Removed this line
                else:
                    st.session_state.chat_history.append({"role": "system", "content": f"Error: {response.text}", "timestamp": datetime.now().isoformat()})

            # Ensure container scrolls to show latest messages
            st.experimental_rerun()
        except Exception as e:
            st.session_state.chat_history.append({"role": "system", "content": f"Error: {str(e)}", "timestamp": datetime.now().isoformat()})

    # Function to save an agent
    def save_agent(name, system_message, agent_id=None):
        try:
            # Prepare data
            agent_data = {
                "name": name,
                "system_message": system_message
            }

            # Add agent_id for updates
            if agent_id:
                agent_data["id"] = agent_id

            # API endpoint
            url = f"{BACKEND_URL}/db/agents"

            # Make request
            response = requests.post(
                url,
                json=agent_data,
                headers={
                    "Authorization": f"Bearer {st.session_state.get('_auth_token_', '')}"
                },
                timeout=10
            )

            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Failed to save agent: {response.text}")
                return None
        except Exception as e:
            st.error(f"Error saving agent: {str(e)}")
            return None

    # Function to load chat sessions
    def list_sessions():
        try:
            # Prepare parameters for the request - only include agent_id
            params = {
                "agent_id": st.session_state.get("chat_agent")
            }

            response = requests.get(
                f"{BACKEND_URL}/chat/sessions",
                headers={
                    "Authorization": f"Bearer {st.session_state.get('_auth_token_', '')}"
                },
                params=params,
                timeout=10
            )

            if response.status_code == 200:
                sessions = response.json()

                # If a document is selected, filter the sessions in the frontend
                if st.session_state.get('chat_document'):
                    document_id = st.session_state.get('chat_document')
                    # Filter sessions to only include those with the matching document_id
                    sessions = [session for session in sessions if session.get('document_id') == document_id]

                return sessions
            else:
                st.error(f"Failed to load chat sessions: {response.text}")
                return []
        except Exception as e:
            st.error(f"Error loading chat sessions: {str(e)}")
            return []

    # Function to create a new chat session
    def create_session(title, agent_id):
        try:
            # Prepare session data
            session_data = {
                "title": title,
                "agent_id": agent_id
            }

            # Add document_id if a specific document is selected
            if st.session_state.get('chat_document'):
                session_data["document_id"] = st.session_state.get('chat_document')

            response = requests.post(
                f"{BACKEND_URL}/chat/sessions",
                json=session_data,
                headers={
                    "Authorization": f"Bearer {st.session_state.get('_auth_token_', '')}"
                },
                timeout=10
            )

            if response.status_code == 200:
                return response.json().get("id")
            else:
                st.error(f"Failed to create chat session: {response.text}")
                return None
        except Exception as e:
            st.error(f"Error creating chat session: {str(e)}")
            return None

    # Function to delete a chat session
    def delete_session(session_id):
        try:
            response = requests.delete(
                f"{BACKEND_URL}/chat/sessions/{session_id}",
                headers={
                    "Authorization": f"Bearer {st.session_state.get('_auth_token_', '')}"
                },
                timeout=10
            )

            if response.status_code == 200:
                return True
            else:
                st.error(f"Failed to delete chat session: {response.text}")
                return False
        except Exception as e:
            st.error(f"Error deleting chat session: {str(e)}")
            return False

    # Function to post a chat message
    def post_chat_message(session_id, message, role="user"):
        try:
            response = requests.post(
                f"{BACKEND_URL}/chat/message",
                json={
                    "session_id": session_id,
                    "message": message,
                    "role": role
                },
                headers={
                    "Authorization": f"Bearer {st.session_state.get('_auth_token_', '')}"
                },
                timeout=30  # Longer timeout for message processing
            )

            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Failed to post message: {response.text}")
                return None
        except Exception as e:
            st.error(f"Error posting message: {str(e)}")
            return None

    # Function to get agent details
    def get_agent_details(agent_id):
        try:
            response = requests.get(
                f"{BACKEND_URL}/db/agents/{agent_id}",
                headers={
                    "Authorization": f"Bearer {st.session_state.get('_auth_token_', '')}"
                },
                timeout=10
            )

            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Failed to get agent details: {response.text}")
                return None
        except Exception as e:
            st.error(f"Error getting agent details: {str(e)}")
            return None

    # Check if an agent is selected
    if not st.session_state.chat_agent:
        # Show welcome message
        st.info("Welcome to Flame Audio Chat! Please select an agent from the sidebar to start chatting.")
    else:
        # Use a single column for the chat area (removed sessions column)
        chat_col = st.container()

        with chat_col:
            # Get agent name to display in title
            agent_name = "Selected Agent"
            if "agents" in st.session_state:
                for agent in st.session_state.agents:
                    if agent.get("id") == st.session_state.chat_agent:
                        agent_name = agent.get("name", "Selected Agent")
                        break

            # Show document name if selected
            doc_label = ""
            if st.session_state.chat_document and "chat_documents" in st.session_state:
                for doc in st.session_state.chat_documents:
                    if doc.get("id") == st.session_state.chat_document:
                        doc_label = f" - {doc.get('document_name', 'Selected Document')}"
                        break

            # Create a container for the chat
            st.subheader(f"Chat with {agent_name}")

            # Add a visual indicator for document context
            if st.session_state.chat_document and "chat_documents" in st.session_state:
                doc_name = ""
                for doc in st.session_state.chat_documents:
                    if doc.get("id") == st.session_state.chat_document:
                        doc_name = doc.get("document_name", "")
                        break

                if doc_name:
                    st.info(f"📄 This chat is focused on the document: **{doc_name}**. The AI will prioritize information from this document.")

            # Chat container with scrolling
            chat_container = st.container(height=500, border=False)

            with chat_container:
                # If we have a session ID but no history, load the session
                if st.session_state.chat_session_id and not st.session_state.chat_history:
                    load_chat_session(st.session_state.chat_session_id)

                # Display chat messages using Streamlit's native chat components
                if not st.session_state.chat_history:
                    # Show welcome message if no history
                    with st.chat_message("assistant"):
                        st.write("Hello! How can I help you with your documents today?")
                else:
                    # Display all messages in the conversation history
                    for message in st.session_state.chat_history:
                        role = message.get("role")
                        content = message.get("content")
                        metadata = message.get("metadata", {})

                        with st.chat_message(role):
                            st.write(content)

                            # Add document sources expander for assistant messages if we have source documents
                            if role == "assistant" and metadata and "source_documents" in metadata:
                                # Get source documents
                                source_documents = metadata.get("source_documents", [])

                                # New Vector Store Results expander
                                with st.expander("🔍 Vector Store Results", expanded=False):
                                    if not source_documents:
                                        st.info("No vector search results available.")
                                    else:
                                        # Show the top 3 results for better visibility
                                        top_results = source_documents[:3] if len(source_documents) > 3 else source_documents

                                        # Create a table with metadata and scores
                                        st.markdown("### Top Vector Search Results")
                                        result_data = []
                                        for i, doc in enumerate(top_results):
                                            # Extract metadata
                                            doc_id = doc.get("metadata", {}).get("job_id", "Unknown")
                                            chunk_index = doc.get("metadata", {}).get("chunk_index", 0)
                                            similarity = doc.get("metadata", {}).get("similarity", 0)
                                            document_name = doc.get("metadata", {}).get("document_name", "Unknown")

                                            # Add to result data
                                            result_data.append({
                                                "Rank": i+1,
                                                "Document": document_name,
                                                "Chunk": chunk_index,
                                                "Score": f"{similarity:.3f}"
                                            })

                                        # Display as dataframe
                                        if result_data:
                                            st.dataframe(pd.DataFrame(result_data), use_container_width=True)

                                        # Display document segments with metadata
                                        st.markdown("### Document Segments")
                                        for i, doc in enumerate(top_results):
                                            # Create a card-like container for each result
                                            with st.container(border=True):
                                                # Header with rank and score
                                                col1, col2 = st.columns([3, 1])
                                                with col1:
                                                    doc_name = doc.get("metadata", {}).get("document_name", "Unknown Document")
                                                    st.markdown(f"**Segment {i+1}**: {doc_name}")
                                                with col2:
                                                    similarity = doc.get("metadata", {}).get("similarity", 0)
                                                    st.markdown(f"**Score**: {similarity:.3f}")

                                                # Metadata section
                                                st.markdown("**Metadata:**")
                                                meta_cols = st.columns(3)
                                                with meta_cols[0]:
                                                    if "job_id" in doc.get("metadata", {}):
                                                        st.caption(f"Document ID: {doc['metadata'].get('job_id', 'Unknown')}")
                                                with meta_cols[1]:
                                                    if "chunk_index" in doc.get("metadata", {}):
                                                        st.caption(f"Chunk Index: {doc['metadata'].get('chunk_index', 0)}")
                                                with meta_cols[2]:
                                                    if "document_name" in doc.get("metadata", {}):
                                                        st.caption(f"Document: {doc['metadata'].get('document_name', 'Unknown')}")

                                                # Content section
                                                st.markdown("**Content:**")
                                                st.code(doc.get("content", "No content available"), language=None)

                                        # Show total count if more than 3 results
                                        if len(source_documents) > 3:
                                            st.caption(f"Showing top 3 of {len(source_documents)} total results. Expand 'View All Source Documents' below to see all.")

                                # Keep the original source documents expander for viewing all results
                                with st.expander("📄 View All Source Documents", expanded=False):
                                    if not source_documents:
                                        st.info("No source documents were used for this response.")
                                    else:
                                        for i, doc in enumerate(source_documents):
                                            st.markdown(f"**Source {i+1}**")

                                            # Display document metadata
                                            meta_cols = st.columns(2)
                                            with meta_cols[0]:
                                                if "job_id" in doc.get("metadata", {}):
                                                    st.caption(f"Document ID: {doc['metadata'].get('job_id', 'Unknown')}")
                                                if "chunk_index" in doc.get("metadata", {}):
                                                    st.caption(f"Chunk: {doc['metadata'].get('chunk_index', 0)}")

                                            with meta_cols[1]:
                                                if "similarity" in doc.get("metadata", {}):
                                                    similarity = doc["metadata"].get("similarity", 0)
                                                    st.caption(f"Relevance: {similarity:.2f}")

                                            # Display document content
                                            st.markdown("```")
                                            st.markdown(doc.get("content", "No content available"))
                                            st.markdown("```")

                                            if i < len(source_documents) - 1:
                                                st.markdown("---")

            # Input area for user message using Streamlit's native chat input
            # Use st.chat_input for user input
            if prompt := st.chat_input("Type your message here...", key="chat_input_main"):

                # Add original user input to history for display
                st.session_state.chat_history.append({"role": "user", "content": prompt})

                # Display the user message immediately using st.chat_message
                with st.chat_message("user"):
                     st.markdown(prompt)

                # Send the original message. Context is handled in send_message via metadata.
                # Show spinner while waiting for backend
                with st.spinner("Agent is thinking..."):
                     send_message(prompt, st.session_state.chat_session_id)
                # The page will rerun via send_message, displaying the new assistant message from history
        with st.sidebar:
            # Additional info about the system
            with st.expander("About Flame Audio Chat"):
                st.markdown("""
                **Flame Audio Chat** allows you to interact with your transcribed audio documents through natural language.

                You can:
                - Chat with any agent you've created
                - Search through specific documents or use all agent knowledge
                - Customize parameters and system prompts
                - Save and manage chat sessions

                The chat interface uses LangChain and LangGraph for advanced conversational abilities and
                connects with your Flame audio transcriptions for context-aware responses.
                """)

    # Add CSS to position chat input at the bottom (from user example)
    st.markdown(
        """
        <style>
            .stChatFloatingInputContainer {
                bottom: 20px;
                background-color: rgba(0, 0, 0, 0)
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.warning("Please sign in to use the chat interface.")