import streamlit as st

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Flame Audio Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

import streamlit_antd_components as sac
import requests
import json
import os
import sys
# import uuid - not used
from typing import Dict, List, Any
from datetime import datetime

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.abspath('.'))

# Import constants from environment or define them directly
import os

# Import the LangGraph client utilities
from utils.langgraph_client import (
    get_langgraph_client,
    create_session,
    chat_with_agent_rag
)

# Define constants that were previously imported from flameaudio.py
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000/api")
AUTH_ENABLED = os.environ.get("AUTH_ENABLED", "true").lower() == "true"
from authentication.auth_forms import auth_forms, handle_auth_callback, init_auth_session

# Initialize authentication session and handle callback
init_auth_session()
handle_auth_callback()

# Initialize basic session state variables first to avoid reference errors
# MCP URL is always loaded from environment, like BACKEND_URL
MCP_URL = os.environ.get("MCP_URL", "http://localhost:8001").split('#')[0].strip().rstrip('/')

if "mcp_status" not in st.session_state:
    st.session_state.mcp_status = {"status": "unknown"}

if "active_tools" not in st.session_state:
    st.session_state.active_tools = {}

if "remote_agents_enabled" not in st.session_state:
    st.session_state.remote_agents_enabled = False

if "remote_agents" not in st.session_state:
    st.session_state.remote_agents = []

if "workflow_enabled" not in st.session_state:
    st.session_state.workflow_enabled = False

if "mcp_tools" not in st.session_state:
    st.session_state.mcp_tools = []

if "messages" not in st.session_state:
    st.session_state.messages = []

if "mcp_chat_session_id" not in st.session_state:
    st.session_state.mcp_chat_session_id = None

if "sessions_reload_requested" not in st.session_state:
    st.session_state.sessions_reload_requested = False

# Add MCP activation toggle state
if "mcp_activated" not in st.session_state:
    st.session_state.mcp_activated = False

# Function to get authentication headers - moved to top for use in other functions
def get_auth_headers():
    if not AUTH_ENABLED:
        return {}

    # Get the auth token from session state
    headers = st.session_state.get("_request_headers_", {})

    # If headers are empty or don't contain Authorization, try to get the token directly
    if not headers or "Authorization" not in headers:
        token = st.session_state.get("_auth_token_")
        if token:
            headers = {"Authorization": f"Bearer {token}"}
        else:
            st.warning("Authentication token not found. Please log in again.")
            st.stop()

    return headers

# Function to load MCP configuration from backend
def load_mcp_config():
    try:
        # Get authentication headers
        headers = get_auth_headers()

        # Call the backend API
        response = requests.get(f"{BACKEND_URL}/mcp/config", headers=headers)

        if response.status_code == 200:
            config = response.json()
            return {
                "mcp_url": config.get("mcp_url", os.environ.get("MCP_URL", "http://localhost:8001")),
                "active_tools": config.get("active_tools", {}),
                "remote_agents_enabled": config.get("remote_agents_enabled", False),
                "workflow_enabled": config.get("workflow_enabled", False),
                "mcp_status": config.get("mcp_status", {"status": "unknown", "details": {"message": "Status not checked"}})
            }
        elif response.status_code == 401 or response.status_code == 403:
            st.warning("Authentication failed. Please log in again.")
            # Clear authentication state
            st.session_state.authenticated = False
            st.session_state._auth_token_ = None
            st.session_state._request_headers_ = None
            st.rerun()
        else:
            st.warning(f"Failed to load configuration from backend: {response.text}")
            return None
    except Exception as e:
        st.warning(f"Error loading configuration: {str(e)}")
        return None

# Function to save MCP configuration to backend
def save_mcp_config(config):
    try:
        # Get authentication headers
        headers = get_auth_headers()

        # Call the backend API
        response = requests.post(f"{BACKEND_URL}/mcp/config", json=config, headers=headers)

        if response.status_code == 200:
            return True
        else:
            print(f"Failed to save configuration: {response.text}")
            return False
    except Exception as e:
        print(f"Error saving configuration: {str(e)}")
        return False

# Function to check MCP service status using LangGraph client
def check_mcp_status(url: str) -> Dict[str, Any]:
    try:
        # Sanitize URL - remove any trailing spaces or comments
        url = url.split('#')[0].strip().rstrip('/')

        # Get authentication token
        auth_token = st.session_state.get("_auth_token_")

        # Debug info
        print(f"Checking MCP status at {url}")

        # Try a direct health check first (more reliable)
        try:
            print("Trying direct health check...")
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                status = {"status": "online", "details": response.json()}
                print(f"Direct health check successful: {status}")

                # Save the status to the backend
                if "config_loaded" in st.session_state and st.session_state.get("authenticated", False):
                    config = {
                        "mcp_url": MCP_URL,
                        "active_tools": st.session_state.active_tools,
                        "remote_agents_enabled": st.session_state.remote_agents_enabled,
                        "workflow_enabled": st.session_state.workflow_enabled,
                        "mcp_status": status
                    }
                    save_mcp_config(config)

                return status
        except Exception as direct_error:
            print(f"Direct health check failed: {str(direct_error)}")

        # Create LangGraph client
        print("Creating LangGraph client for health check...")
        client = get_langgraph_client(url, auth_token)

        # Check health
        print("Calling client.check_health()...")
        status = client.check_health()
        print(f"LangGraph client health check result: {status}")

        # Save the status to the backend
        if "config_loaded" in st.session_state and st.session_state.get("authenticated", False):
            config = {
                "mcp_url": MCP_URL,
                "active_tools": st.session_state.active_tools,
                "remote_agents_enabled": st.session_state.remote_agents_enabled,
                "workflow_enabled": st.session_state.workflow_enabled,
                "mcp_status": status
            }
            save_mcp_config(config)

        return status
    except Exception as e:
        # If there's an error in the client itself, create a fallback status
        error_msg = f"Error checking MCP status: {str(e)}"
        print(error_msg)
        import traceback
        print(traceback.format_exc())

        status = {"status": "error", "details": {"message": error_msg}}

        # Save the status to the backend
        if "config_loaded" in st.session_state and st.session_state.get("authenticated", False):
            config = {
                "mcp_url": MCP_URL,
                "active_tools": st.session_state.active_tools,
                "remote_agents_enabled": st.session_state.remote_agents_enabled,
                "workflow_enabled": st.session_state.workflow_enabled,
                "mcp_status": status
            }
            save_mcp_config(config)

        return status



# Page title
st.title("Flame Audio Chat")

# Add common elements to the sidebar
with st.sidebar:
    st.title("Flame Audio AI: Flame Audio Chat")

# Navigation menu (always visible)
with st.sidebar:
    sac.menu([
        sac.MenuItem('Home', icon='house-fill', href='/flamehome'),
        sac.MenuItem('Playground', icon='mic-fill', href='/flameaudio'),
        sac.MenuItem('Agents', icon='person-fill', href='/agents'),
        sac.MenuItem('Documents', icon='file-text-fill', href='/documents'),
        #sac.MenuItem('Chat', icon='chat-fill', href='/chat'),
        sac.MenuItem('MCP', icon='gear-fill', href='/flame_mcp'),
        sac.MenuItem('Flame Audio Chat', icon='chat-dots-fill'),
    ], open_all=True)

# Show authentication forms if not authenticated
if AUTH_ENABLED:
    # Check if user is authenticated
    authenticated = st.session_state.get("authenticated", False)

    if not authenticated:
        with st.sidebar:
            auth_forms()
        st.warning("Please log in to access MCP Chat.")
        st.stop()

    # Double-check that we have the token
    token = st.session_state.get("_auth_token_")
    if not token:
        with st.sidebar:
            auth_forms()
        st.warning("Authentication token not found. Please log in again.")
        st.stop()



# Chat Sessions Container
with st.sidebar:
    st.subheader("Chat Sessions")

    # --- AUTO-CREATE CHAT SESSION ON PAGE LOAD ---
    # If MCP is activated, online, an agent is selected, and no active chat session, create one automatically
    if (
        st.session_state.get("mcp_activated", False)
        and st.session_state.get("mcp_status", {}).get("status") == "online"
        and st.session_state.get("mcp_selected_agent")
        and not st.session_state.get("mcp_chat_session_id")
    ):
        try:
            agent_id = st.session_state.get("mcp_selected_agent")
            document_id = st.session_state.get("mcp_selected_document")
            title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            auth_token = st.session_state.get("_auth_token_")
            client = get_langgraph_client(MCP_URL, auth_token)
            session_data = {
                "agent_id": agent_id,
                "title": title,
                "metadata": {},
            }
            if document_id:
                session_data["document_id"] = document_id
            # Set session if exists
            if st.session_state.get("mcp_session_id"):
                client.set_session_id(st.session_state.mcp_session_id)
            result = client.call_tool_sync("create_chat_session", {
                "token": auth_token,
                "session_data": session_data
            })
            # Parse session ID from result
            session_id = None
            if result and isinstance(result, str):
                try:
                    result_json = json.loads(result)
                    session_id = result_json.get("session_id") or result_json.get("id")
                except json.JSONDecodeError:
                    session_id = None
            elif isinstance(result, dict):
                session_id = result.get("session_id") or result.get("id")
            if session_id:
                st.session_state.messages = []
                st.session_state.mcp_session_id = session_id
                st.session_state.mcp_chat_session_id = session_id
                st.toast("Started new chat session", icon="✅")
                st.rerun()
            else:
                st.error("Failed to auto-create new chat session.")
                if result:
                    st.error(f"Error details: {result}")
        except Exception as e:
            st.error(f"Exception during auto-creating chat session: {str(e)}")
    # Only show the New Chat button and sessions if an agent is selected
    if st.session_state.get("mcp_selected_agent"):
        # Create a New Chat button at the top level in the sidebar
        if st.button("New Chat", key="new_chat_btn", use_container_width=True):
            agent_id = st.session_state.get("mcp_selected_agent")
            document_id = st.session_state.get("mcp_selected_document")
            title = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            auth_token = st.session_state.get("_auth_token_")
            client = get_langgraph_client(MCP_URL, auth_token)
            # No need to import create_chat_session, we'll call the tool directly

            # Create the session data dictionary
            session_data = {
                "agent_id": agent_id,
                "title": title,
                "metadata": {},
                "chat_parameters": {}
            }

            # Add document_id if selected
            if document_id:
                session_data["document_id"] = document_id

            # Call the create_chat_session tool directly
            # Make sure to set the session ID first to avoid the "No session ID provided" error
            if "mcp_session_id" in st.session_state and st.session_state.mcp_session_id:
                client.set_session_id(st.session_state.mcp_session_id)

            # Debug output
            print(f"DEBUG: Creating chat session with data: {session_data}")

            result = client.call_tool_sync("create_chat_session", {
                "token": auth_token,
                "session_data": session_data
            })

            # Debug output
            print(f"DEBUG: create_chat_session result: {result}")

            # Parse the result
            if result and isinstance(result, str):
                try:
                    print(f"DEBUG: Parsing string result: {result}")
                    result_json = json.loads(result)
                    session_id = result_json.get("session_id") or result_json.get("id")
                    print(f"DEBUG: Parsed session_id from JSON: {session_id}")
                except json.JSONDecodeError as e:
                    print(f"DEBUG: JSON decode error: {str(e)}")
                    session_id = None
            elif isinstance(result, dict):
                print(f"DEBUG: Result is a dict: {result}")
                session_id = result.get("session_id") or result.get("id")
                print(f"DEBUG: Got session_id from dict: {session_id}")
            else:
                print(f"DEBUG: Result is neither string nor dict: {result}")
                session_id = None

            if session_id:
                print(f"DEBUG: Setting session_id in session state: {session_id}")
                st.session_state.messages = []
                st.session_state.mcp_session_id = session_id
                st.session_state.mcp_chat_session_id = session_id
                st.toast("Started new chat session", icon="✅")
                st.rerun()
            else:
                st.error("Failed to create new chat session.")
                if result:
                    st.error(f"Error details: {result}")

        # Function to load chat sessions for the current agent and document using MCP client
        def load_mcp_chat_sessions():
            try:
                # Get authentication token
                auth_token = st.session_state.get("_auth_token_")

                # Create LangGraph client
                client = get_langgraph_client(MCP_URL, auth_token)

                # If we have a session ID, use it
                if "mcp_session_id" in st.session_state:
                    client.set_session_id(st.session_state.mcp_session_id)

                # Get agent ID and document ID
                agent_id = st.session_state.get("mcp_selected_agent", "")
                document_id = st.session_state.get("mcp_selected_document", "")

                # Call the list_chat_sessions tool
                result = client.call_tool_sync("get_user_chat_sessions", {"token": auth_token, "agent_id": agent_id})
                # Handle both list and dict responses
                if result:
                    if isinstance(result, list):
                        sessions = result
                    else:
                        sessions = result.get("sessions", [])
                else:
                    sessions = []

                # If we got a new session ID from the client, save it
                st.session_state.mcp_session_id = client.session_id

                # If a document is selected, filter the sessions in the frontend
                if document_id and sessions:
                    sessions = [session for session in sessions if session.get('document_id') == document_id]

                return sessions or []
            except Exception as e:
                st.error(f"Error loading chat sessions: {str(e)}")
                return []

        # Function to load a specific chat session using MCP client
        def load_mcp_chat_session(session_id):
            try:
                # Get authentication token
                auth_token = st.session_state.get("_auth_token_")

                # Create LangGraph client
                client = get_langgraph_client(MCP_URL, auth_token)

                # If we have a session ID, use it
                if "mcp_session_id" in st.session_state:
                    client.set_session_id(st.session_state.mcp_session_id)

                # Call the get_chat_session tool
                result = client.call_tool_sync("get_chat_session", {"session_id": session_id})
                # Handle both dict and list responses (though this should always be a dict)
                session_data = result if result else None

                # If we got a new session ID from the client, save it
                st.session_state.mcp_session_id = client.session_id

                # Update messages in session state
                if session_data:
                    st.session_state.messages = session_data.get("messages", [])

                return session_data
            except Exception as e:
                st.error(f"Error loading chat session: {str(e)}")
                return None

        # Function to delete a chat session using MCP client
        def delete_mcp_chat_session(session_id):
            try:
                # Get authentication token
                auth_token = st.session_state.get("_auth_token_")

                # Create LangGraph client
                client = get_langgraph_client(MCP_URL, auth_token)

                # If we have a session ID, use it
                if "mcp_session_id" in st.session_state:
                    client.set_session_id(st.session_state.mcp_session_id)

                # Call the delete_chat_session tool
                result = client.call_tool_sync("delete_chat_session", {"session_id": session_id})
                success = result.get("success", False) if result else False

                # If we got a new session ID from the client, save it
                st.session_state.mcp_session_id = client.session_id

                return success
            except Exception as e:
                st.error(f"Error deleting chat session: {str(e)}")
                return False

        # Display session reload button
        if st.button("🔄 Reload Sessions", key="reload_sessions_btn", use_container_width=True):
            st.success("Reloading chat sessions...")
            st.session_state.sessions_reload_requested = True

        # Check if sessions reload was requested
        if st.session_state.get("sessions_reload_requested", False):
            # Clear the flag first
            st.session_state.sessions_reload_requested = False
            # Rerun for safer page refresh
            st.rerun()

        # Create a scrollable container for all chat sessions
        sessions_container = st.container(height=300, border=True)

        # Load and display existing chat sessions
        with sessions_container:
            sessions = load_mcp_chat_sessions()

            if not sessions:
                if st.session_state.get('mcp_selected_document'):
                    # Get document name for better user feedback
                    doc_name = ""
                    if "mcp_documents" in st.session_state:
                        for doc in st.session_state.mcp_documents:
                            if doc.get("id") == st.session_state.get('mcp_selected_document'):
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
                            load_mcp_chat_session(session_id)
                            st.session_state.mcp_chat_session_id = session_id
                            st.rerun()
                    with col2:
                        if st.button("Delete", key=f"delete_{session_id}"):
                            if delete_mcp_chat_session(session_id):
                                if st.session_state.get("mcp_chat_session_id") == session_id:
                                    st.session_state.messages = []
                                    st.session_state.mcp_chat_session_id = None
                                st.rerun()
                            else:
                                st.error(f"Failed to delete session")
    else:
        # If no agent is selected, show a message
        st.info("Select an agent in the settings below to see your chat sessions.")

# Agent Settings Container
if st.session_state.get("mcp_activated", False):
    st.sidebar.markdown("## Agent Settings")
    agent_settings_container = st.sidebar.container(height=350, border=True)
    with agent_settings_container:
        agent_settings_tab = st.radio(
            label="Agent Settings Section",
            options=["Agent Selection", "Model Config", "MCP Tools"],
            index=0,
            horizontal=True,
            key="agent_settings_tab_radio"
        )

        if agent_settings_tab == "Agent Selection":
            st.subheader("Agent Selection")

            # Function to load agents - using only MCP service
            def load_agents():
                try:
                    # Get authentication token
                    auth_token = st.session_state.get("_auth_token_")
                    if not auth_token:
                        st.error("Authentication token not found. Please log in again.")
                        return []

                    # Debug info
                    st.info(f"Loading agents from MCP service at {MCP_URL}")
                    print(f"Loading agents from MCP service at {MCP_URL}")
                    print(f"Auth token: {auth_token[:10]}...")

                    # Create LangGraph client
                    client = get_langgraph_client(MCP_URL, auth_token)

                    # If we have a session ID, use it
                    if "mcp_session_id" in st.session_state:
                        client.set_session_id(st.session_state.mcp_session_id)
                        print(f"Using existing session ID: {st.session_state.mcp_session_id}")
                    else:
                        print("No existing session ID found")

                    # Create a new session
                    print("Creating a new MCP session...")
                    user_id = st.session_state['user']['id'] if 'user' in st.session_state and 'id' in st.session_state['user'] else None
                    if not user_id:
                        st.error("User ID not found in session state. Please log in again.")
                        st.stop()
                    session_created = client.create_session(user_id=user_id)
                    print(f"Session creation result: {session_created}")
                    print(f"New session ID: {client.session_id}")

                    # Call the get_user_agents tool (previously named list_agents)
                    print("Calling get_user_agents tool...")
                    result = client.call_tool_sync("get_user_agents", {"token": auth_token})
                    # Handle both list and dict responses
                    if result:
                        if isinstance(result, list):
                            agents = result
                        else:
                            agents = result.get("agents", [])
                    else:
                        agents = []
                    print(f"Received {len(agents) if agents else 0} agents from LangGraph service")

                    # If we got a new session ID from the client, save it
                    st.session_state.mcp_session_id = client.session_id

                    # Try direct backend call as fallback if no agents returned
                    if not agents:
                        print("No agents returned from MCP service, trying direct backend call...")
                        response = requests.get(
                            f"{BACKEND_URL}/db/agents",
                            headers={
                                "Authorization": f"Bearer {auth_token}"
                            },
                            timeout=10
                        )
                        if response.status_code == 200:
                            direct_agents = response.json()
                            print(f"Received {len(direct_agents)} agents from direct backend call")
                            return direct_agents
                        else:
                            print(f"Direct backend call failed: {response.status_code} - {response.text}")

                    return agents or []
                except Exception as e:
                    error_msg = f"Error loading agents: {str(e)}"
                    st.error(error_msg)
                    print(error_msg)
                    import traceback
                    print(traceback.format_exc())
                    return []

            # Fetch agents
            if "mcp_agents" not in st.session_state or st.button("🔄 Refresh Agents", key="refresh_mcp_agents"):
                with st.spinner("Loading agents..."):
                    st.session_state.mcp_agents = load_agents()

            # Display agent selection dropdown
        agent_options = [{"label": "-- Select an Agent --", "value": ""}]
        if "mcp_agents" in st.session_state and st.session_state.mcp_agents:
            for agent in st.session_state.mcp_agents:
                agent_options.append({
                    "label": agent.get("name", "Unnamed Agent"),
                    "value": agent.get("id", ""),
                    "system_prompt": agent.get("system_message", "")
                })

        # Set default agent if not already set
        if "mcp_selected_agent" not in st.session_state:
            st.session_state.mcp_selected_agent = ""

        # Agent selection
        selected_agent = st.selectbox(
            "Select Agent",
            options=[a["value"] for a in agent_options],
            format_func=lambda x: next((a["label"] for a in agent_options if a["value"] == x), x),
            index=next((i for i, a in enumerate(agent_options) if a["value"] == st.session_state.mcp_selected_agent), 0),
            key="mcp_agent_selector"
        )

        # Update selected agent
        if selected_agent != st.session_state.mcp_selected_agent:
            st.session_state.mcp_selected_agent = selected_agent
            # Reset document selection
            if "mcp_selected_document" in st.session_state:
                st.session_state.mcp_selected_document = ""
            # Force refresh of documents
            if "mcp_documents" in st.session_state:
                del st.session_state.mcp_documents

        # Document Selection (moved from Model Config tab)
        st.subheader("Document Selection")
        if st.session_state.mcp_selected_agent:
            def get_documents(agent_id):
                try:
                    auth_token = st.session_state.get("_auth_token_")
                    if not auth_token:
                        st.error("Authentication token not found. Please log in again.")
                        return []
                    st.info(f"Loading documents for agent {agent_id} from MCP service at {MCP_URL}")
                    print(f"Loading documents for agent {agent_id} from MCP service at {MCP_URL}")
                    client = get_langgraph_client(MCP_URL, auth_token)
                    if "mcp_session_id" in st.session_state:
                        client.set_session_id(st.session_state.mcp_session_id)
                        print(f"Using existing session ID for documents: {st.session_state.mcp_session_id}")
                    print("Calling list_documents tool...")
                    result = client.call_tool_sync("get_user_documents", {"token": auth_token, "agent_id": agent_id})
                    if result:
                        if isinstance(result, list):
                            documents = result
                        else:
                            documents = result.get("documents", [])
                    else:
                        documents = []
                    print(f"Received {len(documents) if documents else 0} documents from LangGraph service")
                    st.session_state.mcp_session_id = client.session_id
                    if not documents:
                        print("No documents returned from MCP service, trying direct backend call...")
                        params = {}
                        if agent_id and agent_id != "all":
                            params["agent_id"] = agent_id
                        response = requests.get(
                            f"{BACKEND_URL}/db/documents",
                            params=params,
                            headers={
                                "Authorization": f"Bearer {auth_token}"
                            },
                            timeout=10
                        )
                        if response.status_code == 200:
                            direct_documents = response.json()
                            print(f"Received {len(direct_documents)} documents from direct backend call")
                            return direct_documents
                        else:
                            print(f"Direct backend call failed: {response.status_code} - {response.text}")
                    return documents or []
                except Exception as e:
                    error_msg = f"Error loading documents: {str(e)}"
                    st.error(error_msg)
                    print(error_msg)
                    import traceback
                    print(traceback.format_exc())
                    return []
            if "mcp_documents" not in st.session_state or st.button("🔄 Refresh Documents", key="refresh_mcp_documents"):
                with st.spinner("Loading documents..."):
                    st.session_state.mcp_documents = get_documents(st.session_state.mcp_selected_agent)
            doc_options = []
            if "mcp_documents" in st.session_state and st.session_state.mcp_documents:
                for doc in st.session_state.mcp_documents:
                    doc_options.append({
                        "label": doc.get("document_name", "Unnamed Document"),
                        "value": doc.get("id", "")
                    })
            if "mcp_selected_document" not in st.session_state:
                st.session_state.mcp_selected_document = ""
            if doc_options:
                selected_doc = st.selectbox(
                    "Select Document",
                    options=[d["value"] for d in doc_options],
                    format_func=lambda x: next((d["label"] for d in doc_options if d["value"] == x), x),
                    index=next((i for i, d in enumerate(doc_options) if d["value"] == st.session_state.mcp_selected_document), 0) if st.session_state.mcp_selected_document in [d["value"] for d in doc_options] else 0,
                    key="mcp_document_selector"
                )
            else:
                selected_doc = ""
                st.session_state.mcp_selected_document = ""
            if selected_doc != st.session_state.mcp_selected_document:
                st.session_state.mcp_selected_document = selected_doc
                st.session_state.mcp_selected_document_id = selected_doc
                if selected_doc:
                    doc_name = next((d["label"] for d in doc_options if d["value"] == selected_doc), "Selected Document")
                    st.session_state.mcp_selected_document_name = doc_name
                    print(f"Selected document: {doc_name} (ID: {selected_doc})")
                else:
                    st.session_state.mcp_selected_document_name = None
            if doc_options and selected_doc:
                doc_name = next((d["label"] for d in doc_options if d["value"] == selected_doc), "Selected Document")
                st.success(f"The agent will focus on document: {doc_name}")
            elif not doc_options:
                st.warning("No documents available for this agent. Please add documents first.")
        else:
            st.warning("Please select an agent first to view associated documents.")

        # Model Config Section
        if agent_settings_tab == "Model Config":
            st.subheader("Model Config")
            # Model Selection
            st.subheader("Model Selection")
            # Function to get chat models using MCP client
            def get_models():
                try:
                    # Get authentication token
                    auth_token = st.session_state.get("_auth_token_")
                    if not auth_token:
                        st.error("Authentication token not found. Please log in again.")
                        return []

                    # Debug info
                    st.info(f"Loading chat models from MCP service at {MCP_URL}")
                    print(f"Loading chat models from MCP service at {MCP_URL}")

                    # Create LangGraph client
                    client = get_langgraph_client(MCP_URL, auth_token)

                    # If we have a session ID, use it
                    if "mcp_session_id" in st.session_state:
                        client.set_session_id(st.session_state.mcp_session_id)
                        print(f"Using existing session ID for models: {st.session_state.mcp_session_id}")

                    # Call the get_chat_models tool
                    print("Calling get_chat_models tool...")
                    result = client.call_tool_sync("get_chat_models", {"token": auth_token})
                    # Handle both list and dict responses
                    if result:
                        if isinstance(result, list):
                            models = result
                        else:
                            models = result.get("models", [])
                    else:
                        models = []
                    print(f"Received {len(models) if models else 0} models from LangGraph service")

                    # If we got a new session ID from the client, save it
                    st.session_state.mcp_session_id = client.session_id

                    # Try direct backend call as fallback if no models returned
                    if not models:
                        print("No models returned from MCP service, trying direct backend call...")
                        response = requests.get(
                            f"{BACKEND_URL}/chat_models",
                            headers={
                                "Authorization": f"Bearer {auth_token}"
                            },
                            timeout=10
                        )
                        if response.status_code == 200:
                            direct_models = response.json()
                            print(f"Received {len(direct_models)} models from direct backend call")
                            return direct_models
                        else:
                            print(f"Direct backend call failed: {response.status_code} - {response.text}")
                    return models or []
                except Exception as e:
                    error_msg = f"Error loading chat models: {str(e)}"
                    st.error(error_msg)
                    print(error_msg)
                    import traceback
                    print(traceback.format_exc())
                    return []

            # Fetch models
            if "mcp_chat_models" not in st.session_state or st.button("🔄 Refresh Models", key="refresh_mcp_models"):
                with st.spinner("Loading chat models..."):
                    st.session_state.mcp_chat_models = get_models()

            # Display model selection dropdown
            model_options = []  # No default options, only use what comes from MCP
            model_descriptions = {}

            if "mcp_chat_models" in st.session_state and st.session_state.mcp_chat_models:
                for model in st.session_state.mcp_chat_models:
                    model_id = model.get("id")
                    if model_id:  # Only add if we have a valid ID
                        model_options.append(model_id)
                        model_descriptions[model_id] = model.get("description", "No description available")

            # Set default model if not already set
            if "mcp_selected_model" not in st.session_state:
                st.session_state.mcp_selected_model = model_options[0] if model_options else "default"

            # Model selection
            if model_options:
                # Determine the index to select
                if st.session_state.mcp_selected_model in model_options:
                    index = model_options.index(st.session_state.mcp_selected_model)
                else:
                    index = 0
                    # Reset selected model if it's not in the options
                    st.session_state.mcp_selected_model = model_options[0]

                # Display the model selection dropdown
                selected_model = st.selectbox(
                    "Select Model",
                    options=model_options,
                    index=index,
                    key="mcp_model_selector"
                )

                # Update selected model
                if selected_model != st.session_state.mcp_selected_model:
                    st.session_state.mcp_selected_model = selected_model

                # Show model description if available
                if selected_model in model_descriptions and model_descriptions[selected_model]:
                    st.info(model_descriptions[selected_model])
            else:
                st.warning("No models available from MCP service. Please check the MCP service configuration.")
                st.session_state.mcp_selected_model = ""
                            # --- System Prompt Section ---
            st.subheader("System Prompt")
            system_prompt = ""
            # Try to find the system prompt from the selected agent
            if (
                "mcp_agents" in st.session_state and st.session_state.mcp_agents and
                st.session_state.get("mcp_selected_agent")
            ):
                for agent in st.session_state.mcp_agents:
                    if agent.get("id") == st.session_state.mcp_selected_agent:
                        system_prompt = agent.get("system_message", "")
                        break

            # Display editable text area for system prompt
            edited_system_prompt = st.text_area(
                "Edit System Prompt for this Session",
                value=system_prompt,
                height=200,
                key="mcp_system_prompt"
            )
            # Warn if edited
            if edited_system_prompt != system_prompt:
                st.warning("System prompt has been modified from the agent's default. This change will only apply to this chat session.")
            # Do NOT assign to st.session_state['mcp_system_prompt'] after widget instantiation to avoid StreamlitAPIException.
            # Use 'edited_system_prompt' variable wherever the current prompt is needed.

            # --- Model Parameters Section ---
            st.subheader("Model Parameters")
            if "mcp_model_parameters" not in st.session_state:
                st.session_state.mcp_model_parameters = {
                    "temperature": 0.7,
                    "max_tokens": 1024,
                    "top_p": 0.9
                }
            # Sliders for model parameters
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.mcp_model_parameters.get("temperature", 0.7),
                step=0.05,
                help="Higher values make output more random, lower values make it more deterministic."
            )
            max_tokens = st.slider(
                "Max Output Tokens",
                min_value=128,
                max_value=4096,
                value=st.session_state.mcp_model_parameters.get("max_tokens", 1024),
                step=128,
                help="Maximum number of tokens to generate in the response."
            )
            top_p = st.slider(
                "Top P",
                min_value=0.1,
                max_value=1.0,
                value=st.session_state.mcp_model_parameters.get("top_p", 0.9),
                step=0.05,
                help="Controls diversity of generated text. Lower values generate more focused text."
            )
            # Update parameters in session state
            st.session_state.mcp_model_parameters = {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p
            }


        # MCP Tools Section
        if agent_settings_tab == "MCP Tools":
            st.subheader("MCP Tools")
            mcp_tools = st.session_state.get("mcp_tools", [])
            if not mcp_tools:
                # Try to load tools if not already loaded
                try:
                    mcp_tools = get_mcp_tools(MCP_URL)
                    st.session_state.mcp_tools = mcp_tools
                except Exception as e:
                    st.error(f"Could not load MCP tools: {e}")
            if mcp_tools:
                # Bulk select/deselect buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Select All Tools", key="select_all_tools_mcpchat", use_container_width=True):
                        for tool in mcp_tools:
                            st.session_state.active_tools[tool.get("name", "")] = True
                with col2:
                    if st.button("Deselect All Tools", key="deselect_all_tools_mcpchat", use_container_width=True):
                        for tool in mcp_tools:
                            st.session_state.active_tools[tool.get("name", "")] = False
                st.divider()
                # Display each tool with toggle
                for tool in mcp_tools:
                    tool_name = tool.get("name", "")
                    tool_desc = tool.get("description", "No description")
                    # Initialize tool state if not present
                    if tool_name not in st.session_state.active_tools:
                        st.session_state.active_tools[tool_name] = True
                    cols = st.columns([4, 1])
                    with cols[0]:
                        st.markdown(f"**{tool_name}**")
                        st.caption(tool_desc)
                    with cols[1]:
                        is_active = st.toggle("Active", value=st.session_state.active_tools.get(tool_name, True), key=f"toggle_mcpchat_{tool_name}")
                        if is_active != st.session_state.active_tools.get(tool_name, True):
                            # Call toggle_tool if available, else just update state
                            try:
                                toggle_tool(tool_name, is_active)
                            except Exception:
                                st.session_state.active_tools[tool_name] = is_active
                    st.divider()
            else:
                st.info("No MCP tools available or failed to load.")


# Function to get available MCP tools using MCP client
def get_mcp_tools(url=MCP_URL):
    try:
        # Get authentication token
        auth_token = st.session_state.get("_auth_token_")

        # Create LangGraph client
        client = get_langgraph_client(url, auth_token)

        # Get tools
        tools = client.get_tools()

        return tools
    except Exception as e:
        st.error(f"Error getting MCP tools: {str(e)}")
        return []

# Function to get available remote agents using MCP client
def get_remote_agents(url=MCP_URL):
    try:
        # Get authentication token
        auth_token = st.session_state.get("_auth_token_")

        # Create LangGraph client
        client = get_langgraph_client(url, auth_token)

        # Call the get_remote_agents tool (previously named list_remote_agents)
        # Check if the tool exists first
        tools = client.list_tools()
        tool_names = [tool.get("name") for tool in tools]

        if "get_remote_agents" in tool_names:
            result = client.call_tool_sync("get_remote_agents", {"token": auth_token})
        elif "list_remote_agents" in tool_names:
            result = client.call_tool_sync("list_remote_agents", {"token": auth_token})
        else:
            print("Remote agents tool not found")
            return []

        # Extract agents from result
        if result and "agents" in result:
            return result.get("agents", [])

        return []
    except Exception as e:
        st.error(f"Error getting remote agents: {str(e)}")
        return []

# Function to get available workflows using MCP client
def get_workflows(url=MCP_URL):
    try:
        # Get authentication token
        auth_token = st.session_state.get("_auth_token_")

        # Create LangGraph client
        client = get_langgraph_client(url, auth_token)

        # Call the list_workflows tool
        # Check if the tool exists first
        tools = client.list_tools()
        tool_names = [tool.get("name") for tool in tools]

        if "list_workflows" in tool_names:
            result = client.call_tool_sync("list_workflows", {"token": auth_token})
        else:
            print("Workflows tool not found")
            return []

        # Extract workflows from result
        if result and "workflows" in result:
            return result.get("workflows", [])

        return []
    except Exception as e:
        st.error(f"Error getting workflows: {str(e)}")
        return []

# Function to create a session with the MCP service using LangGraph client
def create_langgraph_session(url=MCP_URL, tools=None, model_name="default", system_message=None, document_id=None):
    # If MCP is offline, don't even try to create a session
    if st.session_state.mcp_status["status"] == "offline":
        return False

    try:
        # Get authentication token
        auth_token = st.session_state.get("_auth_token_")
        if not auth_token:
            st.error("Authentication token not found. Please log in again.")
            return False

        # Get user ID from session state
        user_id = st.session_state['user']['id'] if 'user' in st.session_state and 'id' in st.session_state['user'] else None
        if not user_id:
            st.error("User ID not found in session state. Please log in again.")
            return False

        # Create LangGraph client
        client = get_langgraph_client(url, auth_token)

        # Always create a new session to ensure we have a valid one
        print("Creating new LangGraph session...")

        # First, try to delete any existing session
        if "langgraph_session_id" in st.session_state and st.session_state.langgraph_session_id:
            try:
                # Try to delete the existing session
                client.set_session_id(st.session_state.langgraph_session_id)
                client.delete_session()
                print(f"Deleted existing session: {st.session_state.langgraph_session_id}")
            except Exception as close_error:
                print(f"Error deleting session: {str(close_error)}")

        # Get document_id if selected
        document_id = st.session_state.get("mcp_selected_document_id")

        # If a document is selected, create a custom system message
        if document_id and not system_message:
            # Get document name if available
            document_name = st.session_state.get("mcp_selected_document_name", "the selected document")
            system_message = f"You are a helpful AI assistant that can answer questions about {document_name}. Use the document context provided to give accurate answers. If the document context doesn't contain the information needed, you can say so and answer based on your general knowledge."
            print(f"Created custom system message for document: {document_name}")

        # Create a new session using the LangGraph client
        # Pass the user_id as the first parameter
        success = client.create_session(user_id, tools, model_name, system_message, document_id)

        # Save the session ID
        st.session_state.langgraph_session_id = client.session_id
        print(f"New LangGraph session ID: {st.session_state.langgraph_session_id}")

        # If successful, update the MCP status
        if success:
            # Update MCP status to online and save to database
            new_status = {"status": "online", "details": {"message": "Connected successfully to LangGraph"}}
            st.session_state.mcp_status = new_status

            # Save the updated status to the database
            if "config_loaded" in st.session_state and st.session_state.get("authenticated", False):
                config = {
                    "mcp_url": MCP_URL,
                    "active_tools": st.session_state.active_tools,
                    "remote_agents_enabled": st.session_state.remote_agents_enabled,
                    "workflow_enabled": st.session_state.workflow_enabled,
                    "mcp_status": new_status
                }
                save_mcp_config(config)

            # Store the client in session state
            st.session_state.langgraph_client = client
            return True
        else:
            st.error("Failed to create LangGraph session. Please check the MCP service logs.")
            return False
    except Exception as e:
        print(f"Error creating session: {str(e)}")
        st.error(f"Error creating MCP session: {str(e)}")

        # Update MCP status to error and save to database
        new_status = {"status": "error", "details": {"message": f"Error: {str(e)}"}}
        st.session_state.mcp_status = new_status

        # Save the updated status to the database
        if "config_loaded" in st.session_state and st.session_state.get("authenticated", False):
            config = {
                "mcp_url": MCP_URL,
                "active_tools": st.session_state.active_tools,
                "remote_agents_enabled": st.session_state.remote_agents_enabled,
                "workflow_enabled": st.session_state.workflow_enabled,
                "mcp_status": new_status
            }
            save_mcp_config(config)

        return False

# Function to send a chat message using the new structured RAG chat helper

def send_message(
    message: str,
    retrieved_chunks: list = None,
    system_prompt: str = None,
    agent_id: str = None,
    document_id: str = None,
    parameters: dict = None,
    metadata: dict = None,
    session_id: str = None,
) -> dict:
    """
    Send a chat message using the unified chat_with_agent_rag helper.
    """
    auth_token = st.session_state.get("_auth_token_")
    client = st.session_state.get("mcp_client")
    if not client:
        client = get_langgraph_client(MCP_URL, auth_token)
        st.session_state["mcp_client"] = client

    # Always use the correct chat session id
    if not session_id:
        # Prefer mcp_chat_session_id, fall back to mcp_session_id
        session_id = st.session_state.get("mcp_chat_session_id") or st.session_state.get("mcp_session_id")

    # Debug output
    print(f"DEBUG: Using session_id for send_message: {session_id}")

    # Make sure to set the session ID in the client to avoid the "No session ID provided" error
    if session_id:
        client.set_session_id(session_id)

    # Log which session_id is being used for debugging
    import logging
    logger = logging.getLogger("langgraph_client")
    logger.info(f"Using session_id for send_message: {session_id}")
    return chat_with_agent_rag(
        client,
        token=auth_token,
        question=message,
        retrieved_chunks=retrieved_chunks,
        system_prompt=system_prompt,
        agent_id=agent_id,
        document_id=document_id,
        parameters=parameters,
        metadata=metadata,
        session_id=session_id,
    )

# Load configuration from backend if not already loaded
if "config_loaded" not in st.session_state:
    # Try to load config from backend
    config = load_mcp_config()
    if config:
        # Do not load MCP URL from backend config; always use environment
        st.session_state.active_tools = config.get("active_tools", {})
        st.session_state.remote_agents_enabled = config.get("remote_agents_enabled", False)
        st.session_state.workflow_enabled = config.get("workflow_enabled", False)
        st.session_state.mcp_status = config.get("mcp_status", {"status": "unknown", "details": {"message": "Status not checked"}})
        st.toast("Configuration loaded from backend", icon="ℹ️")

    st.session_state.config_loaded = True

if "workflows" not in st.session_state:
    st.session_state.workflows = []

# Load MCP status from database and check actual connection if MCP is activated
if isinstance(st.session_state.mcp_status, str) or st.session_state.mcp_status.get("status") == "unknown":
    # First try to use the status from the database without checking the actual connection
    config = load_mcp_config()
    if config and config.get("mcp_status", {}).get("status") in ["online", "offline", "error"]:
        st.session_state.mcp_status = config.get("mcp_status")
        print(f"Using MCP status from database: {st.session_state.mcp_status['status']}")
    else:
        # Only check the actual connection if we don't have a valid status from the database
        st.session_state.mcp_status = check_mcp_status(MCP_URL)
        print(f"Checked actual MCP connection, status: {st.session_state.mcp_status['status']}")

# If MCP is activated, always check the actual connection status
import time

CACHE_SECONDS = 60
now = time.time()
last_checked = st.session_state.get("mcp_status_last_checked", 0)

if st.session_state.get("mcp_activated", False):
    # Only check MCP status if cache expired or unknown
    if (
        not st.session_state.get("mcp_status")
        or st.session_state.mcp_status.get("status") == "unknown"
        or now - last_checked > CACHE_SECONDS
    ):
        st.session_state.mcp_status = check_mcp_status(MCP_URL)
        st.session_state.mcp_status_last_checked = now
        print(f"MCP status checked at {now}, status: {st.session_state.mcp_status['status']}")
    else:
        print(f"MCP status cache hit, last checked at {last_checked}, status: {st.session_state.mcp_status['status']}")

    # Save the status to the backend
    if "config_loaded" in st.session_state and st.session_state.get("authenticated", False):
        config = {
            "mcp_url": MCP_URL,
            "active_tools": st.session_state.active_tools,
            "remote_agents_enabled": st.session_state.remote_agents_enabled,
            "workflow_enabled": st.session_state.workflow_enabled,
            "mcp_status": st.session_state.mcp_status
        }
        save_mcp_config(config)

    # Load tools and other resources if online
    if st.session_state.mcp_status["status"] == "online":
        st.session_state.mcp_tools = get_mcp_tools(MCP_URL)

        # Get remote agents if enabled
        if st.session_state.remote_agents_enabled:
            st.session_state.remote_agents = get_remote_agents(MCP_URL)

        # Get workflows if enabled
        if st.session_state.workflow_enabled:
            st.session_state.workflows = get_workflows(MCP_URL)

# Create a session if we don't have one and MCP is online and activated
# But only try to connect if the user has explicitly activated MCP
if st.session_state.mcp_activated and st.session_state.mcp_status["status"] == "online":
    # Always create a new session when the page loads to ensure we have a valid session
    print("Creating a new MCP session...")

    # Get authentication token
    auth_token = st.session_state.get("_auth_token_")

    if auth_token:
        # Create MCP client
        client = get_langgraph_client(MCP_URL, auth_token)

        # Create a new session
        user_id = st.session_state['user']['id'] if 'user' in st.session_state and 'id' in st.session_state['user'] else None
        if not user_id:
            print("User ID not found in session state. Cannot create MCP session.")
        else:
            if client.create_session(user_id=user_id):
                # Save the session ID and client
                st.session_state.mcp_session_id = client.session_id
                st.session_state.mcp_client = client  # Store the client to keep the SSE connection alive
                print(f"Successfully created MCP session: {st.session_state.mcp_session_id}")
            else:
                print("Failed to create MCP session")




# Main chat interface
st.markdown("Ask questions about your documents or use the available tools to help you with your tasks.")

# Add a horizontal rule for visual separation
st.markdown("---")

# Create the chat container
chat_container = st.container()

# Add a clear chat button
col1, col2 = st.columns([5, 1])
with col2:
    if st.button("🗑️ Clear Chat", key="clear_chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Chat history cleared. How can I help you today?"}
        ]
        st.rerun()

# Display chat messages
with chat_container:
    # Add some custom CSS for better styling
    st.markdown("""
    <style>
    .stChatMessage {border-radius: 10px; padding: 10px;}
    .stChatMessage[data-testid="stChatMessageUser"] {background-color: #e6f7ff;}
    .stChatMessage[data-testid="stChatMessageAssistant"] {background-color: #f0f2f6;}
    </style>
    """, unsafe_allow_html=True)

    # Debug the messages state
    print(f"DEBUG: Current messages in session state: {st.session_state.messages}")

    # Display messages
    for message in st.session_state.messages:
        role = message.get("role", "")
        content = message.get("content", "")
        print(f"DEBUG: Displaying message - role: {role}, content: {content[:50]}...")

        if role == "user":
            st.chat_message("user", avatar="👤").write(content)
        elif role == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.write(content)

                # --- Source Documents Expander ---
                if "metadata" in message and "source_documents" in message["metadata"]:
                    source_documents = message["metadata"]["source_documents"]
                    with st.expander("📄 Source Documents", expanded=False):
                        if not source_documents:
                            st.info("No source documents were used for this response.")
                        else:
                            for i, doc in enumerate(source_documents):
                                # Extract metadata, handling possible locations
                                metadata = doc.get("metadata", {})

                                # Check if we have a nested metadata structure (common in Qdrant responses)
                                nested_metadata = metadata.get("metadata", {})

                                # First try to get values from the nested metadata (most complete)
                                if nested_metadata:
                                    doc_name = nested_metadata.get("document_name", "Unknown Document")
                                    chunk_index = nested_metadata.get("chunk_index", 0)
                                    start_time = nested_metadata.get("start_time")
                                    end_time = nested_metadata.get("end_time")
                                    score = metadata.get("similarity", 0)  # Score is usually in the parent metadata
                                    segment = nested_metadata.get("segment", "Unknown")
                                else:
                                    # Fallback to the parent metadata
                                    doc_name = metadata.get("document_name") or doc.get("document_name", "Unknown Document")
                                    chunk_index = metadata.get("chunk_index") or doc.get("chunk_index", 0)
                                    start_time = metadata.get("start_time") or doc.get("start_time")
                                    end_time = metadata.get("end_time") or doc.get("end_time")
                                    score = metadata.get("similarity") or doc.get("similarity", 0)
                                    segment = metadata.get("segment") or doc.get("segment", "Unknown")

                                # Print debug info
                                print(f"DEBUG: Document metadata extraction:")
                                print(f"  - doc_name: {doc_name}")
                                print(f"  - chunk_index: {chunk_index}")
                                print(f"  - segment: {segment}")
                                print(f"  - start_time: {start_time}")
                                print(f"  - end_time: {end_time}")
                                print(f"  - score: {score}")

                                # Try to extract segment from content if not found in metadata
                                if segment == "Unknown" and "content" in doc:
                                    content = doc.get("content", "")
                                    if "Segment" in content:
                                        import re
                                        segment_match = re.search(r"Segment\s+(\d+)", content)
                                        if segment_match:
                                            segment = segment_match.group(1)
                                            print(f"DEBUG: Extracted segment {segment} from content")

                                # If segment is still unknown, use chunk_index + 1 as segment
                                if segment == "Unknown" and chunk_index is not None:
                                    segment = str(int(chunk_index) + 1) if isinstance(chunk_index, (int, float)) or (isinstance(chunk_index, str) and chunk_index.isdigit()) else "1"
                                    print(f"DEBUG: Using chunk_index {chunk_index} as segment {segment}")

                                # If still unknown, default to 1
                                if segment == "Unknown":
                                    segment = "1"
                                    print(f"DEBUG: Using default segment 1")

                                # Format values
                                start_time_disp = f"{float(start_time):.2f}s" if start_time is not None else "N/A"
                                end_time_disp = f"{float(end_time):.2f}s" if end_time is not None else "N/A"
                                score_disp = f"{float(score):.3f}" if score is not None else "N/A"

                                # Header
                                st.markdown(f"**Source {i+1}: {doc_name}**")
                                # Always display chunk/segment as 1-based (never 0-based)
                                chunk_display = int(chunk_index) + 1 if isinstance(chunk_index, int) or (isinstance(chunk_index, str) and chunk_index.isdigit()) else chunk_index
                                col1, col2, col3, col4, col5 = st.columns(5)
                                with col1:
                                    st.caption(f"Segment: {segment}")
                                with col2:
                                    st.caption(f"Chunk: {chunk_display}")
                                with col3:
                                    st.caption(f"Start Time: {start_time_disp}")
                                with col4:
                                    st.caption(f"End Time: {end_time_disp}")
                                with col5:
                                    st.caption(f"Score: {score_disp}")
                                st.code(doc.get("content", "No content available"))
                                if i < len(source_documents) - 1:
                                    st.markdown("---")

                # Display tool calls if present
                if "tool_calls" in message:
                    tool_calls = message.get("tool_calls", [])
                    for tool_call in tool_calls:
                        tool_name = tool_call.get("name", "")
                        arguments = tool_call.get("arguments", "{}")

                        with st.expander(f"🔧 Tool Call: {tool_name}"):
                            st.json(json.loads(arguments))

# Message input (always visible, disables as appropriate)
chat_disabled = not st.session_state.mcp_activated or st.session_state.mcp_status["status"] != "online"
chat_submission = st.chat_input(
    "Type your message or attach an audio file...",
    accept_file=True,
    file_type=["wav", "mp3", "m4a", "ogg", "flac"],
    disabled=chat_disabled,
)

if chat_disabled:
    if not st.session_state.mcp_activated:
        st.info("MCP is currently deactivated. Please activate MCP in the sidebar to use the chat functionality.")
    elif st.session_state.mcp_status["status"] != "online":
        st.warning("MCP service is not connected. Please check the connection in the sidebar.")

if chat_submission:
    # chat_submission is a dict-like object with 'text' and 'files' attributes
    user_message = chat_submission.text
    uploaded_files = chat_submission.files  # List of UploadedFile objects

    # Check if MCP service is online and activated
    if chat_disabled:
        st.error("MCP Service is not available. Please check the connection.")
    else:
        # Get active tools from configuration
        active_tools = [tool for tool, is_active in st.session_state.active_tools.items() if is_active]

        # Add chat_with_agent tool if an agent is selected
        if st.session_state.get("mcp_selected_agent") and "chat_with_agent" not in active_tools:
            active_tools.append("chat_with_agent")

        # If audio files are uploaded, navigate to flameaudio.py and load the user's agent
        if uploaded_files:
            # Set the agent in session state so flameaudio.py can load it, and preload agent details
            if st.session_state.get("mcp_selected_agent"):
                agent_id = st.session_state["mcp_selected_agent"]
                st.session_state.current_agent_id = agent_id

                # Fetch agent details directly to avoid Streamlit import collision
                try:
                    import os
                    import requests
                    BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
                    auth_token = st.session_state.get('_auth_token_', '')
                    response = requests.get(
                        f"{BACKEND_URL}/db/agents/{agent_id}",
                        headers={"Authorization": f"Bearer {auth_token}"},
                        timeout=10
                    )
                    if response.status_code == 200:
                        agent_data = response.json()
                        st.session_state.agent_name = agent_data.get("name", "")
                        st.session_state.system_message = agent_data.get("system_message", "")
                except Exception as e:
                    st.warning(f"Could not load agent details: {e}")
            # Optionally store the uploaded files in session state for flameaudio.py to use
            st.session_state.uploaded_audio_files = uploaded_files
            # Navigate to flameaudio.py
            st.switch_page("pages/flameaudio.py")
            st.stop()
        # The rest of the chat logic (sending message to backend, etc.) continues below.

        # Show debugging information
        # with st.expander("Debug Information", expanded=True):
        #     st.write("MCP URL:", MCP_URL)
        #     st.write("Session ID:", st.session_state.get("mcp_session_id", "Not set"))
        #     st.write("Active Tools:", active_tools)
        #     st.write("Auth Headers:", get_auth_headers())
        #     st.write("Current Messages:", st.session_state.get("messages", []))

        # Add user message to session state and UI
        if user_message not in [msg.get("content", "") for msg in st.session_state.messages if msg.get("role") == "user"]:
            st.session_state.messages.append({"role": "user", "content": user_message})
            print(f"DEBUG: Added user message to session state: {user_message}")
        st.chat_message("user", avatar="👤").write(user_message)

        # Add a typing indicator
        with st.chat_message("assistant", avatar="🤖"):
            typing_placeholder = st.empty()
            typing_placeholder.markdown("_Thinking..._")

            # Send message to MCP service
            # Gather chat parameters
            agent_id = st.session_state.get("mcp_selected_agent")
            document_id = st.session_state.get("mcp_selected_document")
            system_prompt = st.session_state.get("edited_system_prompt")
            parameters = st.session_state.get("mcp_model_parameters")
            session_id = st.session_state.get("mcp_chat_session_id")
            # For now, assume no retrieved_chunks or metadata (add if needed)

            # Debug output
            print(f"DEBUG: Preparing to send message with session_id: {session_id}")
            print(f"DEBUG: agent_id: {agent_id}, document_id: {document_id}")
            print(f"DEBUG: All session state keys: {list(st.session_state.keys())}")

            if not session_id:
                st.error("No active chat session. Please start a new chat before sending messages.")
                typing_placeholder.empty()
                result = {"success": False, "error": "No active chat session. Please start a new chat before sending messages."}
            else:
                try:
                    print(f"DEBUG: Sending message to MCP service with session_id: {session_id}")
                    print(f"DEBUG: agent_id: {agent_id}, document_id: {document_id}")

                    # Create a new client for this request to ensure session ID is set
                    auth_token = st.session_state.get("_auth_token_")
                    mcp_url = st.session_state.get("mcp_url", "http://localhost:8001")
                    client = get_langgraph_client(mcp_url, auth_token)

                    # Explicitly set the session ID
                    client.set_session_id(session_id)
                    print(f"DEBUG: Set session_id in client to: {session_id}")

                    # Call the chat_with_agent tool directly
                    chat_request = {
                        "message": user_message,
                        "agent_id": agent_id,
                        "document_id": document_id,
                        "system_prompt": system_prompt,
                        "parameters": parameters,
                        "session_id": session_id
                    }

                    # Remove None values
                    chat_request = {k: v for k, v in chat_request.items() if v is not None}

                    # Use the chat_with_agent_rag helper to ensure correct payload structure
                    from utils.langgraph_client import chat_with_agent_rag
                    result = chat_with_agent_rag(
                        client,
                        token=auth_token,
                        question=user_message,
                        retrieved_chunks=None,
                        system_prompt=system_prompt,
                        agent_id=agent_id,
                        document_id=document_id,
                        parameters=parameters,
                        metadata=None,
                        session_id=session_id
                    )
                    print(f"DEBUG: Message sent successfully, result: {result}")
                except Exception as e:
                    print(f"DEBUG: Exception during send_message: {str(e)}")
                    result = {"success": False, "error": f"Exception during send_message: {str(e)}"}

            # Clear the typing indicator
            typing_placeholder.empty()

            # Debug expander removed as per user request

            if not isinstance(result, dict):
                st.error("Internal error: Backend did not return a valid response object.")
            elif not result.get("success", False):
                error_msg = result.get("error", "Unknown error occurred")
                st.error(f"Error: {error_msg}")
                # Show raw error if present
                if isinstance(result, dict) and "message" in result:
                    st.error(f"Backend error message: {result['message']}")
                # Handle authentication errors
                if "authentication failed" in str(error_msg).lower():
                    st.warning("Please try refreshing the page and logging in again.")
                    # Clear authentication state
                    if st.button("Refresh Authentication"):
                        st.session_state.authenticated = False
                        st.session_state._auth_token_ = None
                        st.session_state._request_headers_ = None
                        st.rerun()
            else:
                # Add the assistant's response to the chat history
                print(f"DEBUG: Processing successful result: {result}")
                if "response" in result and result["response"]:
                    print(f"DEBUG: Found response in result: {result['response']}")
                    # If the result contains a messages array, use the last assistant message (with full metadata)
                    if "messages" in result and result["messages"]:
                        assistant_msgs = [msg for msg in result["messages"] if msg.get("role") == "assistant"]
                        if assistant_msgs:
                            last_assistant = assistant_msgs[-1]
                            assistant_message = {
                                "role": "assistant",
                                "content": last_assistant.get("content", ""),
                                "metadata": last_assistant.get("metadata", {})
                            }
                            st.session_state.messages.append(assistant_message)
                            print(f"DEBUG: Added assistant message to session state (from messages): {assistant_message}")
                            typing_placeholder.markdown(last_assistant.get("content", ""))
                            st.success("Message processed successfully")
                        else:
                            # Fallback if no assistant message found
                            assistant_message = {
                                "role": "assistant",
                                "content": result["response"],
                                "metadata": result.get("metadata", {})
                            }
                            st.session_state.messages.append(assistant_message)
                            print(f"DEBUG: Added assistant message to session state: {assistant_message}")
                            typing_placeholder.markdown(result["response"])
                            st.success("Message processed successfully")
                    else:
                        assistant_message = {
                            "role": "assistant",
                            "content": result["response"],
                            "metadata": result.get("metadata", {})
                        }
                        st.session_state.messages.append(assistant_message)
                        print(f"DEBUG: Added assistant message to session state: {assistant_message}")
                        typing_placeholder.markdown(result["response"])
                        st.success("Message processed successfully")
                elif "messages" in result and result["messages"]:
                    # If the response contains a messages array, add the last assistant message
                    print(f"DEBUG: Found messages in result: {result['messages']}")
                    for msg in result["messages"]:
                        if msg.get("role") == "assistant":
                            assistant_message = {
                                "role": "assistant",
                                "content": msg.get("content", ""),
                                "metadata": msg.get("metadata", {})
                            }
                            st.session_state.messages.append(assistant_message)
                            print(f"DEBUG: Added assistant message from messages array: {assistant_message}")
                            typing_placeholder.markdown(msg.get("content", ""))
                    st.success("Message processed successfully")
                else:
                    # If no response or messages, create a simple response
                    print("DEBUG: No response or messages found in result, creating default response")
                    default_response = "I received your message but couldn't generate a proper response. Please try again."
                    assistant_message = {
                        "role": "assistant",
                        "content": default_response
                    }
                    st.session_state.messages.append(assistant_message)
                    typing_placeholder.markdown(default_response)
                    st.warning("No response content found in the result")

        # Force UI refresh
        st.rerun()

# MCP Settings Container: Always visible for authenticated users
if AUTH_ENABLED and st.session_state.get("authenticated", False):
    st.sidebar.markdown("## Chat Settings")
    mcp_settings_container = st.sidebar.container(border=True)
    with mcp_settings_container:
        mcp_activated = st.toggle(
            "Activate MCP",
            value=st.session_state.get("mcp_activated", False),
            help="Toggle to activate or deactivate MCP features."
        )

        # Update session state if the toggle value changed
        if mcp_activated != st.session_state.get("mcp_activated", False):
            st.session_state.mcp_activated = mcp_activated

            # If MCP is activated, check the status and update it
            if mcp_activated:
                # Check the actual MCP status
                new_status = check_mcp_status(MCP_URL)
                st.session_state.mcp_status = new_status

                # If MCP is online, try to create a session
                if new_status.get("status") == "online":
                    print("MCP is online, creating a session...")
                    # Get authentication token
                    auth_token = st.session_state.get("_auth_token_")
                    if auth_token:
                        # Create LangGraph client
                        client = get_langgraph_client(MCP_URL, auth_token)

                        # Create a new session
                        # Get user ID from session state
                        user_id = st.session_state['user']['id'] if 'user' in st.session_state and 'id' in st.session_state['user'] else None
                        if not user_id:
                            print("User ID not found in session state. Cannot create MCP session.")
                        else:
                            session_created = create_session(client, user_id)
                            print(f"Session creation result: {session_created}")
                            if session_created:
                                # Save the session ID
                                st.session_state.mcp_session_id = client.session_id
                                print(f"Created new session: {st.session_state.mcp_session_id}")

                                # Store the client in session state
                                st.session_state.mcp_client = client
                            else:
                                print("Failed to create MCP session")
                    else:
                        print("No authentication token found, cannot create MCP session")

                # Save the status to the backend
                if "config_loaded" in st.session_state and st.session_state.get("authenticated", False):
                    config = {
                        "mcp_url": MCP_URL,
                        "active_tools": st.session_state.active_tools,
                        "remote_agents_enabled": st.session_state.remote_agents_enabled,
                        "workflow_enabled": st.session_state.workflow_enabled,
                        "mcp_status": new_status
                    }
                    save_mcp_config(config)

                # Force a rerun to update the UI
                st.rerun()
        else:
            st.session_state.mcp_activated = mcp_activated
        # MCP Service URL input
        st.text_input(
            "MCP Service URL",
            value=MCP_URL,
            key="mcp_url_input",
            disabled=True
        )
        # Remove update button and config save logic for MCP URL

        # MCP status display - only show if activated
        if st.session_state.get("mcp_activated", False):
            with st.spinner("Checking MCP status..."):
                # Use cache for MCP status unless user forces refresh
                now = time.time()
                last_checked = st.session_state.get("mcp_status_last_checked", 0)
                if (
                    not st.session_state.get("mcp_status")
                    or st.session_state.mcp_status.get("status") == "unknown"
                    or now - last_checked > CACHE_SECONDS
                ):
                    current_status = check_mcp_status(MCP_URL)
                    st.session_state.mcp_status = current_status
                    st.session_state.mcp_status_last_checked = now
                    print(f"MCP status checked at {now}, status: {current_status['status']}")
                else:
                    current_status = st.session_state.mcp_status
                    print(f"MCP status cache hit, last checked at {last_checked}, status: {current_status['status']}")

                # Save the status to the backend
                if "config_loaded" in st.session_state and st.session_state.get("authenticated", False):
                    config = {
                        "mcp_url": MCP_URL,
                        "active_tools": st.session_state.active_tools,
                        "remote_agents_enabled": st.session_state.remote_agents_enabled,
                        "workflow_enabled": st.session_state.workflow_enabled,
                        "mcp_status": current_status
                    }
                    save_mcp_config(config)

                # Display the current status
                if current_status.get("status") == "online":
                    st.success("MCP Service is online")
                else:
                    st.warning(f"MCP Service is offline or has errors: {current_status.get('details', {}).get('message', 'Unknown error')}")

                # Add a button to check the status manually
                if st.button("Check Status Again", key="check_mcp_status"):
                    # Force a fresh MCP status check
                    new_status = check_mcp_status(MCP_URL)
                    st.session_state.mcp_status = new_status
                    st.session_state.mcp_status_last_checked = time.time()

                    # Save the status to the backend
                    if "config_loaded" in st.session_state and st.session_state.get("authenticated", False):
                        config = {
                            "mcp_url": MCP_URL,
                            "active_tools": st.session_state.active_tools,
                            "remote_agents_enabled": st.session_state.remote_agents_enabled,
                            "workflow_enabled": st.session_state.workflow_enabled,
                            "mcp_status": new_status
                        }
                        save_mcp_config(config)

                    # Force a rerun to update the UI
                    st.rerun()
        else:
            st.info("MCP is currently deactivated. Activate MCP to check status.")

#connect with us
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



