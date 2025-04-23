import streamlit as st

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Flame Audio MCP Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

import streamlit_antd_components as sac
import requests
import json
import os
import sys
import uuid
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
    send_message
)

# Define constants that were previously imported from flameaudio.py
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000/api")
AUTH_ENABLED = os.environ.get("AUTH_ENABLED", "true").lower() == "true"
from authentication.auth_forms import auth_forms, handle_auth_callback, init_auth_session

# Initialize authentication session and handle callback
init_auth_session()
handle_auth_callback()

# Initialize basic session state variables first to avoid reference errors
if "mcp_url" not in st.session_state:
    st.session_state.mcp_url = "http://localhost:8001"

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
                "mcp_url": config.get("mcp_url", "http://localhost:8001"),
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
                        "mcp_url": st.session_state.mcp_url,
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
                "mcp_url": st.session_state.mcp_url,
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
                "mcp_url": st.session_state.mcp_url,
                "active_tools": st.session_state.active_tools,
                "remote_agents_enabled": st.session_state.remote_agents_enabled,
                "workflow_enabled": st.session_state.workflow_enabled,
                "mcp_status": status
            }
            save_mcp_config(config)

        return status



# Page title
st.title("MCP Chat")

# Add common elements to the sidebar
with st.sidebar:
    st.title("Flame Audio AI: MCP Chat")

# Navigation menu (always visible)
with st.sidebar:
    sac.menu([
        sac.MenuItem('Home', icon='house-fill', href='/flamehome'),
        sac.MenuItem('Playground', icon='mic-fill', href='/flameaudio'),
        sac.MenuItem('Agents', icon='person-fill', href='/agents'),
        sac.MenuItem('Documents', icon='file-text-fill', href='/documents'),
        sac.MenuItem('Chat', icon='chat-fill', href='/chat'),
        sac.MenuItem('MCP', icon='gear-fill', href='/flame_mcp'),
        sac.MenuItem('MCP Chat', icon='chat-dots-fill'),
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
                new_status = check_mcp_status(st.session_state.mcp_url)
                st.session_state.mcp_status = new_status

                # If MCP is online, try to create a session
                if new_status.get("status") == "online":
                    print("MCP is online, creating a session...")
                    # Get authentication token
                    auth_token = st.session_state.get("_auth_token_")
                    if auth_token:
                        # Create LangGraph client
                        client = get_langgraph_client(st.session_state.mcp_url, auth_token)

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
                        "mcp_url": st.session_state.mcp_url,
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
        st.session_state.mcp_url = st.text_input(
            "MCP Service URL",
            value=st.session_state.get("mcp_url", "http://localhost:8001"),
            key="mcp_url_input"
        )
        # Update button with actual functionality
        if st.button("Update Connection", key="update_mcp_url"):
            # Check the actual MCP status
            new_status = check_mcp_status(st.session_state.mcp_url)
            st.session_state.mcp_status = new_status

            # Save the status to the backend
            if "config_loaded" in st.session_state and st.session_state.get("authenticated", False):
                config = {
                    "mcp_url": st.session_state.mcp_url,
                    "active_tools": st.session_state.active_tools,
                    "remote_agents_enabled": st.session_state.remote_agents_enabled,
                    "workflow_enabled": st.session_state.workflow_enabled,
                    "mcp_status": new_status
                }
                save_mcp_config(config)

            # Force a rerun to update the UI
            st.rerun()

        # MCP status display
        if st.session_state.get("mcp_status", {}).get("status") == "online":
            st.success("MCP Service is online")
        else:
            st.warning("MCP Service is offline or status unknown")

            # Add a button to check the status manually
            if st.button("Check Status", key="check_mcp_status"):
                # Check the actual MCP status
                new_status = check_mcp_status(st.session_state.mcp_url)
                st.session_state.mcp_status = new_status

                # Save the status to the backend
                if "config_loaded" in st.session_state and st.session_state.get("authenticated", False):
                    config = {
                        "mcp_url": st.session_state.mcp_url,
                        "active_tools": st.session_state.active_tools,
                        "remote_agents_enabled": st.session_state.remote_agents_enabled,
                        "workflow_enabled": st.session_state.workflow_enabled,
                        "mcp_status": new_status
                    }
                    save_mcp_config(config)

                # Force a rerun to update the UI
                st.rerun()

# Agent Settings Container
if st.session_state.get("mcp_activated", False):
    st.sidebar.markdown("## Agent Settings")
    agent_settings_container = st.sidebar.container(border=True)

    with agent_settings_container:
        if not AUTH_ENABLED or st.session_state.get("authenticated", False):
            # Agent Selection
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
                    st.info(f"Loading agents from MCP service at {st.session_state.mcp_url}")
                    print(f"Loading agents from MCP service at {st.session_state.mcp_url}")
                    print(f"Auth token: {auth_token[:10]}...")

                    # Create LangGraph client
                    client = get_langgraph_client(st.session_state.mcp_url, auth_token)

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
                st.info(f"Loading chat models from MCP service at {st.session_state.mcp_url}")
                print(f"Loading chat models from MCP service at {st.session_state.mcp_url}")

                # Create LangGraph client
                client = get_langgraph_client(st.session_state.mcp_url, auth_token)

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

        # Document Selection
        st.subheader("Document Selection")

        # Only show document selection if an agent is selected
        if st.session_state.mcp_selected_agent:
            # Function to get documents for the selected agent using MCP client
            def get_documents(agent_id):
                try:
                    # Get authentication token
                    auth_token = st.session_state.get("_auth_token_")
                    if not auth_token:
                        st.error("Authentication token not found. Please log in again.")
                        return []

                    # Debug info
                    st.info(f"Loading documents for agent {agent_id} from MCP service at {st.session_state.mcp_url}")
                    print(f"Loading documents for agent {agent_id} from MCP service at {st.session_state.mcp_url}")

                    # Create LangGraph client
                    client = get_langgraph_client(st.session_state.mcp_url, auth_token)

                    # If we have a session ID, use it
                    if "mcp_session_id" in st.session_state:
                        client.set_session_id(st.session_state.mcp_session_id)
                        print(f"Using existing session ID for documents: {st.session_state.mcp_session_id}")

                    # Call the list_documents tool
                    print("Calling list_documents tool...")
                    result = client.call_tool_sync("get_user_documents", {"token": auth_token, "agent_id": agent_id})
                    # Handle both list and dict responses
                    if result:
                        if isinstance(result, list):
                            documents = result
                        else:
                            documents = result.get("documents", [])
                    else:
                        documents = []
                    print(f"Received {len(documents) if documents else 0} documents from LangGraph service")

                    # If we got a new session ID from the client, save it
                    st.session_state.mcp_session_id = client.session_id

                    # Try direct backend call as fallback if no documents returned
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

            # Fetch documents for the current agent
            if "mcp_documents" not in st.session_state or st.button("🔄 Refresh Documents", key="refresh_mcp_documents"):
                with st.spinner("Loading documents..."):
                    st.session_state.mcp_documents = get_documents(st.session_state.mcp_selected_agent)

            # Display document selection dropdown
            doc_options = []
            if "mcp_documents" in st.session_state and st.session_state.mcp_documents:
                for doc in st.session_state.mcp_documents:
                    doc_options.append({
                        "label": doc.get("document_name", "Unnamed Document"),
                        "value": doc.get("id", "")
                    })

            # Set default document if not already set
            if "mcp_selected_document" not in st.session_state:
                st.session_state.mcp_selected_document = ""

            # Document selection
            if doc_options:  # Only show dropdown if there are documents
                selected_doc = st.selectbox(
                    "Select Document",
                    options=[d["value"] for d in doc_options],
                    format_func=lambda x: next((d["label"] for d in doc_options if d["value"] == x), x),
                    index=next((i for i, d in enumerate(doc_options) if d["value"] == st.session_state.mcp_selected_document), 0) if st.session_state.mcp_selected_document in [d["value"] for d in doc_options] else 0,
                    key="mcp_document_selector"
                )
            else:
                # No documents available
                selected_doc = ""
                st.session_state.mcp_selected_document = ""

            # Update selected document
            if selected_doc != st.session_state.mcp_selected_document:
                st.session_state.mcp_selected_document = selected_doc

                # Also store the document ID for LangGraph session creation
                st.session_state.mcp_selected_document_id = selected_doc

                # Store the document name for better context
                if selected_doc:
                    doc_name = next((d["label"] for d in doc_options if d["value"] == selected_doc), "Selected Document")
                    st.session_state.mcp_selected_document_name = doc_name
                    print(f"Selected document: {doc_name} (ID: {selected_doc})")
                else:
                    st.session_state.mcp_selected_document_name = None

            # Show info about document selection
            if doc_options and selected_doc:
                doc_name = next((d["label"] for d in doc_options if d["value"] == selected_doc), "Selected Document")
                st.success(f"The agent will focus on document: {doc_name}")
            elif not doc_options:
                st.warning("No documents available for this agent. Please add documents first.")
        else:
            st.warning("Please select an agent first to view associated documents.")

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


# Function to get available MCP tools using MCP client
def get_mcp_tools(url: str) -> List[Dict[str, Any]]:
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
def get_remote_agents(url: str) -> List[Dict[str, Any]]:
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
def get_workflows(url: str) -> List[Dict[str, Any]]:
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
def create_langgraph_session(url: str, tools: List[str] = None, model_name: str = "default", system_message: str = None, document_id: str = None) -> bool:
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
                    "mcp_url": st.session_state.mcp_url,
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
                "mcp_url": st.session_state.mcp_url,
                "active_tools": st.session_state.active_tools,
                "remote_agents_enabled": st.session_state.remote_agents_enabled,
                "workflow_enabled": st.session_state.workflow_enabled,
                "mcp_status": new_status
            }
            save_mcp_config(config)

        return False

# Function to send a message to the LangGraph service
def send_langgraph_message(url: str, message: str, tools: List[str] = None, document_id: str = None, model_name: str = None) -> Dict[str, Any]:
    try:
        # Get authentication token
        auth_token = st.session_state.get("_auth_token_")
        if not auth_token:
            return {"success": False, "error": "Authentication token not found. Please log in again."}

        # Initialize messages if not in session state
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hello! I'm your AI assistant. How can I help you today?"}
            ]

        # Get or create LangGraph client
        if "langgraph_client" in st.session_state:
            client = st.session_state.langgraph_client
        else:
            client = get_langgraph_client(url, auth_token)

        # Get model name if selected
        model_name = st.session_state.get("mcp_selected_model", "default")

        # Get document_id if not provided
        if document_id is None:
            document_id = st.session_state.get("mcp_selected_document_id")

        # Get tool names if any
        tool_names = tools

        # Always try to create/verify the session before sending a message
        if "langgraph_session_id" not in st.session_state or not st.session_state.langgraph_session_id:
            # Create a new session with the selected model, tools, and document
            # Create a custom system message for document context if needed
            custom_system_message = None
            if document_id:
                document_name = st.session_state.get("mcp_selected_document_name", "the selected document")
                custom_system_message = f"You are a helpful AI assistant that can answer questions about {document_name}. Use the document context provided to give accurate answers. If the document context doesn't contain the information needed, you can say so and answer based on your general knowledge."
                print(f"Created custom system message for document: {document_name}")

            if not create_langgraph_session(url, tool_names, model_name, system_message=custom_system_message, document_id=document_id):
                return {"success": False, "error": "Failed to create LangGraph session"}

        # Make sure the client has the correct session ID
        client.set_session_id(st.session_state.langgraph_session_id)
        print(f"Using LangGraph session ID for message: {st.session_state.langgraph_session_id}")

        # Add user message to session state
        st.session_state.messages.append({"role": "user", "content": message})

        # Send the message using the LangGraph client
        print(f"Sending message to LangGraph with model={model_name}, document_id={document_id}")
        # Use positional arguments instead of keyword arguments
        response = client.send_message(message, None, document_id, model_name)

        # Process the response
        if response:
            # Debug the response structure in detail
            print(f"Response from LangGraph: {response}")
            print(f"Response type: {type(response)}")

            # Print the result field structure
            result = response.get("result", {})
            print(f"Result field: {result}")
            print(f"Result type: {type(result)}")

            # If result is a dict, print its keys
            if isinstance(result, dict):
                print(f"Result keys: {result.keys()}")

                # Check for source documents
                if "source_documents" in result:
                    print(f"Source documents found in result: {len(result['source_documents'])}")
                elif "document_sources" in result:
                    print(f"Document sources found in result: {len(result['document_sources'])}")
                elif "metadata" in result and "source_documents" in result["metadata"]:
                    print(f"Source documents found in result.metadata: {len(result['metadata']['source_documents'])}")
                else:
                    print("No source documents found in result")

            # Extract the response content from the result field
            result = response.get("result", {})
            content = ""

            # Try different ways to extract the content
            if isinstance(result, dict):
                # Try to get content from the result dictionary
                content = result.get("content", "")

                # If no content found, try to get it from the response field
                if not content and "response" in result:
                    content = result["response"]

                # If still no content, check if there's a messages field with an assistant message
                if not content and "messages" in result:
                    messages = result["messages"]
                    if isinstance(messages, list):
                        for msg in messages:
                            if isinstance(msg, dict) and msg.get("role") == "assistant":
                                content = msg.get("content", "")
                                break
            elif isinstance(result, str):
                # If result is a string, use it directly
                content = result

            # Create the assistant message
            assistant_message = {
                "role": "assistant",
                "content": content,
                "metadata": {}
            }

            # Add metadata if available
            if isinstance(result, dict):
                # Check for source documents in the result
                if "source_documents" in result:
                    assistant_message["metadata"]["source_documents"] = result["source_documents"]
                    print(f"Added {len(result['source_documents'])} source documents from result.source_documents")
                elif "document_sources" in result:
                    assistant_message["metadata"]["source_documents"] = result["document_sources"]
                    print(f"Added {len(result['document_sources'])} source documents from result.document_sources")
                elif "metadata" in result and "source_documents" in result["metadata"]:
                    assistant_message["metadata"]["source_documents"] = result["metadata"]["source_documents"]
                    print(f"Added {len(result['metadata']['source_documents'])} source documents from result.metadata.source_documents")
                # Check for messages field that might contain metadata
                elif "messages" in result:
                    messages = result["messages"]
                    if isinstance(messages, list) and len(messages) > 0:
                        for msg in messages:
                            if isinstance(msg, dict) and msg.get("role") == "assistant" and "metadata" in msg:
                                msg_metadata = msg["metadata"]
                                if isinstance(msg_metadata, dict) and "source_documents" in msg_metadata:
                                    assistant_message["metadata"]["source_documents"] = msg_metadata["source_documents"]
                                    print(f"Added {len(msg_metadata['source_documents'])} source documents from result.messages[].metadata")
                                    break
                # Check for response field that might contain metadata
                elif "response" in result and isinstance(result["response"], dict):
                    response_obj = result["response"]
                    if "metadata" in response_obj and "source_documents" in response_obj["metadata"]:
                        assistant_message["metadata"]["source_documents"] = response_obj["metadata"]["source_documents"]
                        print(f"Added {len(response_obj['metadata']['source_documents'])} source documents from result.response.metadata")

            # Add tool calls if any
            if "tool_calls" in response and response["tool_calls"]:
                assistant_message["tool_calls"] = response["tool_calls"]

            # Add the message to session state
            st.session_state.messages.append(assistant_message)

            # Return success
            return {
                "success": True,
                "messages": st.session_state.messages,
                "response": response
            }
        else:
            # Return error
            return {
                "success": False,
                "error": "No response from LangGraph",
                "messages": st.session_state.messages
            }
    except Exception as e:
        print(f"Exception in send_langgraph_message: {str(e)}")
        return {"success": False, "error": f"Error sending message: {str(e)}"}

# Keep the original function for backward compatibility
def send_message(url: str, message: str, tools: List[str] = None, document_id: str = None, model_name: str = None) -> Dict[str, Any]:
    # Use the LangGraph message function instead
    try:
        return send_langgraph_message(url, message, tools, document_id, model_name)
    except Exception as e:
        print(f"Exception in send_message: {str(e)}")
        return {"success": False, "error": f"Error sending message: {str(e)}"}

# Load configuration from backend if not already loaded
if "config_loaded" not in st.session_state:
    # Try to load config from backend
    config = load_mcp_config()
    if config:
        st.session_state.mcp_url = config.get("mcp_url", "http://localhost:8001")
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
        st.session_state.mcp_status = check_mcp_status(st.session_state.mcp_url)
        print(f"Checked actual MCP connection, status: {st.session_state.mcp_status['status']}")

# If MCP is activated, always check the actual connection status
if st.session_state.get("mcp_activated", False):
    # Check the actual MCP status
    st.session_state.mcp_status = check_mcp_status(st.session_state.mcp_url)
    print(f"MCP is activated, checked actual connection, status: {st.session_state.mcp_status['status']}")

    # Save the status to the backend
    if "config_loaded" in st.session_state and st.session_state.get("authenticated", False):
        config = {
            "mcp_url": st.session_state.mcp_url,
            "active_tools": st.session_state.active_tools,
            "remote_agents_enabled": st.session_state.remote_agents_enabled,
            "workflow_enabled": st.session_state.workflow_enabled,
            "mcp_status": st.session_state.mcp_status
        }
        save_mcp_config(config)

    # Load tools and other resources if online
    if st.session_state.mcp_status["status"] == "online":
        st.session_state.mcp_tools = get_mcp_tools(st.session_state.mcp_url)

        # Get remote agents if enabled
        if st.session_state.remote_agents_enabled:
            st.session_state.remote_agents = get_remote_agents(st.session_state.mcp_url)

        # Get workflows if enabled
        if st.session_state.workflow_enabled:
            st.session_state.workflows = get_workflows(st.session_state.mcp_url)

# Create a session if we don't have one and MCP is online and activated
# But only try to connect if the user has explicitly activated MCP
if st.session_state.mcp_activated and st.session_state.mcp_status["status"] == "online":
    # Always create a new session when the page loads to ensure we have a valid session
    print("Creating a new MCP session...")

    # Get authentication token
    auth_token = st.session_state.get("_auth_token_")

    if auth_token:
        # Create MCP client
        client = get_langgraph_client(st.session_state.mcp_url, auth_token)

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

# Chat Sessions Container
with st.sidebar:
    st.subheader("Chat Sessions")

    # Only show the New Chat button and sessions if an agent is selected
    if st.session_state.get("mcp_selected_agent"):
        # Create a New Chat button at the top level in the sidebar
        if st.button("New Chat", key="new_chat_btn", use_container_width=True):
            # Clear chat history and session ID
            st.session_state.messages = []
            st.session_state.mcp_session_id = str(uuid.uuid4())
            st.toast("Started new chat session", icon="✅")
            st.rerun()

        # Function to load chat sessions for the current agent and document using MCP client
        def load_mcp_chat_sessions():
            try:
                # Get authentication token
                auth_token = st.session_state.get("_auth_token_")

                # Create LangGraph client
                client = get_langgraph_client(st.session_state.mcp_url, auth_token)

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
                client = get_langgraph_client(st.session_state.mcp_url, auth_token)

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
                client = get_langgraph_client(st.session_state.mcp_url, auth_token)

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
        st.info("Select an agent in the settings above to see your chat sessions.")


# Main chat interface
st.markdown("## 💬 Chat with MCP")
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

    # Display messages
    for message in st.session_state.messages:
        role = message.get("role", "")
        content = message.get("content", "")

        if role == "user":
            st.chat_message("user", avatar="👤").write(content)
        elif role == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.write(content)

                # Display source documents if available
                if "metadata" in message and "source_documents" in message["metadata"]:
                    with st.expander("📄 Sources"):
                        for i, doc in enumerate(message["metadata"]["source_documents"]):
                            st.markdown(f"**Source {i+1}:**")
                            st.markdown(doc["content"])
                            st.markdown("---")

                # Display tool calls if present
                if "tool_calls" in message:
                    tool_calls = message.get("tool_calls", [])
                    for tool_call in tool_calls:
                        tool_name = tool_call.get("name", "")
                        arguments = tool_call.get("arguments", "{}")

                        with st.expander(f"🔧 Tool Call: {tool_name}"):
                            st.json(json.loads(arguments))

# Message input
if not st.session_state.mcp_activated:
    st.info("MCP is currently deactivated. Please activate MCP in the sidebar to use the chat functionality.")
    user_message = st.chat_input("Type your message here...", disabled=True)
elif st.session_state.mcp_status["status"] != "online":
    st.warning("MCP service is not connected. Please check the connection in the sidebar.")
    user_message = st.chat_input("Type your message here...", disabled=True)
else:
    user_message = st.chat_input("Type your message here...")

if user_message:
    # Check if MCP service is online and activated
    if not st.session_state.mcp_activated or st.session_state.mcp_status["status"] != "online":
        st.error("MCP Service is not available. Please check the connection.")
    else:
        # Get active tools from configuration
        active_tools = [tool for tool, is_active in st.session_state.active_tools.items() if is_active]

        # Add chat_with_agent tool if an agent is selected
        if st.session_state.get("mcp_selected_agent") and "chat_with_agent" not in active_tools:
            active_tools.append("chat_with_agent")

        # Show debugging information
        with st.expander("Debug Information", expanded=True):
            st.write("MCP URL:", st.session_state.mcp_url)
            st.write("Session ID:", st.session_state.get("mcp_session_id", "Not set"))
            st.write("Active Tools:", active_tools)
            st.write("Auth Headers:", get_auth_headers())
            st.write("Current Messages:", st.session_state.get("messages", []))

        # Add user message to UI immediately
        st.chat_message("user", avatar="👤").write(user_message)

        # Add a typing indicator
        with st.chat_message("assistant", avatar="🤖"):
            typing_placeholder = st.empty()
            typing_placeholder.markdown("_Thinking..._")

            # Send message to MCP service
            result = send_message(st.session_state.mcp_url, user_message, active_tools)

            # Clear the typing indicator
            typing_placeholder.empty()

            # Show result for debugging
            with st.expander("Response Debug", expanded=True):
                st.json(result)

            if not result.get("success", False):
                error_msg = result.get("error", "Unknown error occurred")
                st.error(f"Error: {error_msg}")

                # Handle authentication errors
                if "authentication failed" in error_msg.lower():
                    st.warning("Please try refreshing the page and logging in again.")
                    # Clear authentication state
                    if st.button("Refresh Authentication"):
                        st.session_state.authenticated = False
                        st.session_state._auth_token_ = None
                        st.session_state._request_headers_ = None
                        st.rerun()
            else:
                st.success("Message processed successfully")

        # Force UI refresh
        st.rerun()
