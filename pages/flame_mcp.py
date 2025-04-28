import streamlit as st

# Page configuration must be the first Streamlit command
st.set_page_config(
    page_title="Flame Audio MCP",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

import streamlit_antd_components as sac
import requests
import os
import sys
from typing import Dict, List, Any

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.abspath('.'))

# Import constants from environment or define them directly
import os

# Define constants that were previously imported from flameaudio.py
BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000/api")
AUTH_ENABLED = os.environ.get("AUTH_ENABLED", "true").lower() == "true"
from authentication.auth_forms import auth_forms, handle_auth_callback, init_auth_session

# Initialize authentication session and handle callback
init_auth_session()
handle_auth_callback()

# Page title
st.title("MCP Service Configuration")

# Add common elements to the sidebar
with st.sidebar:
    st.title("Flame Audio AI: MCP")

# Navigation menu (always visible)
with st.sidebar:
    sac.menu([
        sac.MenuItem('Home', icon='house-fill', href='/flamehome'),
        sac.MenuItem('Playground', icon='mic-fill', href='/flameaudio'),
        sac.MenuItem('Agents', icon='person-fill', href='/agents'),
        sac.MenuItem('Documents', icon='file-text-fill', href='/documents'),
        #sac.MenuItem('Chat', icon='chat-fill', href='/chat'),
        sac.MenuItem('MCP', icon='gear-fill'),
        sac.MenuItem('Flame Audio Chat', icon='chat-dots-fill', href='/mcp_chat'),
    ], open_all=True)

# Show authentication forms if not authenticated
if AUTH_ENABLED and not st.session_state.get("authenticated", False):
    with st.sidebar:
        auth_forms()
    st.warning("Please log in to access MCP configuration.")
    st.stop()

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

# Import the LangGraph client utilities
from utils.langgraph_client import (
    get_langgraph_client,
    list_tools as langgraph_list_tools
)

# Function to get available MCP tools using LangGraph
def get_mcp_tools(url: str) -> List[Dict[str, Any]]:
    try:
        # Log that we're getting tools
        print(f"Getting tools from LangGraph at {url}/langgraph/tools") # Ensure no space in URL

        # Get authentication token
        auth_token = st.session_state.get("_auth_token_")
        if not auth_token:
            print("Authentication token not found. Please log in again.")
            return []

        # Create LangGraph client
        client = get_langgraph_client(url, auth_token) # URL is passed correctly, ensure no space when using

        # Get tools using the LangGraph client
        tools = langgraph_list_tools(client) # The client should use URLs without spaces
        print(f"Found {len(tools)} tools from LangGraph service")

        # If we have tools in active_tools but they're not in the response,
        # create dummy tools for them so they can be displayed
        if 'active_tools' in st.session_state and st.session_state.active_tools and tools:
            for tool_name in st.session_state.active_tools:
                if not any(tool.get("name") == tool_name for tool in tools):
                    tools.append({
                        "name": tool_name,
                        "description": f"Tool from configuration: {tool_name}"
                    })
            print(f"Added missing tools from configuration, total now: {len(tools)}")

        return tools
    except Exception as e:
        error_msg = f"Error getting LangGraph tools: {str(e)}"
        print(error_msg)

        # If we have active_tools, create dummy tools from them
        if 'active_tools' in st.session_state and st.session_state.active_tools:
            print(f"Creating dummy tools from active_tools ({len(st.session_state.active_tools)} tools)")
            dummy_tools = []
            for tool_name, is_active in st.session_state.active_tools.items():
                # Only include active tools
                if is_active:
                    dummy_tools.append({
                        "name": tool_name,
                        "description": f"Tool from configuration: {tool_name}"
                    })
            return dummy_tools

        st.error(error_msg)
        return []

# Function to check MCP service status using LangGraph
def check_mcp_status(url: str) -> Dict[str, Any]:
    try:
        # Sanitize URL - remove any trailing spaces or comments
        url = url.split('#')[0].strip().rstrip('/')

        # First try the LangGraph endpoint
        try:
            # Get authentication token
            auth_token = st.session_state.get("_auth_token_")
            if not auth_token:
                print("Authentication token not found. Please log in again.")
                return {"status": "error", "details": {"message": "Authentication token not found"}}

            # Create LangGraph client
            client = get_langgraph_client(url, auth_token) # URL is passed correctly, ensure no space when using

            # List sessions to check if LangGraph is working
            sessions = client.list_sessions()

            # If we get here, LangGraph is working
            status = {"status": "online", "details": {"message": "LangGraph is online", "sessions": len(sessions)}}
            return status
        except Exception as langgraph_error:
            # If LangGraph fails, try the regular health endpoint
            print(f"LangGraph check failed: {str(langgraph_error)}")

            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                status = {"status": "online", "details": response.json()}
                # We'll save the status to the backend later when save_mcp_config is defined
                return status
            else:
                status = {"status": "error", "details": {"message": f"HTTP {response.status_code}: {response.text}"}}
                return status
    except Exception as e:
        status = {"status": "offline", "details": {"message": str(e)}}
        return status

# Function to load configuration from backend API
def load_config():
    try:
        # Only attempt to load if authenticated
        if AUTH_ENABLED and not st.session_state.get("authenticated", False):
            return None

        # Get the auth token
        headers = st.session_state.get("_request_headers_", {})

        # If headers are empty or don't contain Authorization, try to get the token directly
        if not headers or "Authorization" not in headers:
            token = st.session_state.get("_auth_token_")
            if token:
                headers = {"Authorization": f"Bearer {token}"}
                print("Using direct token for authentication")

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
        else:
            st.warning(f"Failed to load configuration from backend: {response.text}")
            return None
    except Exception as e:
        st.warning(f"Error loading configuration: {str(e)}")
        return None

# Initialize session state variables
if "config_loaded" not in st.session_state:
    # Try to load config from backend
    config = load_config()
    if config:
        # Always sanitize loaded MCP URL
        st.session_state.mcp_url = config.get("mcp_url", os.environ.get("MCP_URL", "http://localhost:8001")).strip().rstrip('/')
        st.session_state.active_tools = config.get("active_tools", {})
        st.session_state.remote_agents_enabled = config.get("remote_agents_enabled", False)
        st.session_state.workflow_enabled = config.get("workflow_enabled", False)
        st.session_state.mcp_status = config.get("mcp_status", {"status": "unknown", "details": {"message": "Status not checked"}})
        st.toast("Configuration loaded from backend", icon="ℹ️")
    else:
        # Default values
        st.session_state.mcp_url = os.environ.get("MCP_URL", "http://localhost:8001")
        st.session_state.active_tools = {}
        st.session_state.remote_agents_enabled = False
        st.session_state.workflow_enabled = False
        st.session_state.mcp_status = {"status": "unknown", "details": {"message": "Status not checked"}}

    st.session_state.config_loaded = True

if "mcp_tools" not in st.session_state:
    st.session_state.mcp_tools = []
    # Try to load tools from MCP service if we have a URL
    if "mcp_url" in st.session_state:
        print(f"Trying to load tools from MCP service at {st.session_state.mcp_url}")
        st.session_state.mcp_tools = get_mcp_tools(st.session_state.mcp_url)
        print(f"Loaded {len(st.session_state.mcp_tools)} tools from MCP service")

if "mcp_status" not in st.session_state:
    st.session_state.mcp_status = {"status": "unknown", "details": {"message": "Status not checked"}}

# Function to save MCP configuration to backend
def save_mcp_config(config):
    try:
        # Get the auth token
        headers = st.session_state.get("_request_headers_", {})

        # If headers are empty or don't contain Authorization, try to get the token directly
        if not headers or "Authorization" not in headers:
            token = st.session_state.get("_auth_token_")
            if token:
                headers = {"Authorization": f"Bearer {token}"}
            else:
                st.error("Authentication token not found. Please log in again.")
                return False

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

# Function to check MCP service status (moved to the top of the file)

# Function to get available MCP tools (moved to the top of the file)

# Function to toggle tool activation
def toggle_tool(tool_name: str, is_active: bool):
    st.session_state.active_tools[tool_name] = is_active
    # In a real implementation, you would call an API to activate/deactivate the tool
    st.toast(f"Tool '{tool_name}' {'activated' if is_active else 'deactivated'}")

# Function to update MCP URL
def update_mcp_url():
    # Sanitize URL before using it
    sanitized_url = st.session_state.mcp_url.split('#')[0].strip().rstrip('/')
    print(f"Updating MCP URL to {sanitized_url}")

    # Store the sanitized URL back in session state
    st.session_state.mcp_url = sanitized_url

    # Check MCP status
    st.session_state.mcp_status = check_mcp_status(sanitized_url)
    print(f"MCP status: {st.session_state.mcp_status['status']}")

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

    # Get tools if MCP is online
    if st.session_state.mcp_status["status"] == "online":
        print("MCP is online, getting tools")
        st.session_state.mcp_tools = get_mcp_tools(st.session_state.mcp_url)
        print(f"Got {len(st.session_state.mcp_tools)} tools from MCP service")

        # Initialize active_tools for new tools
        for tool in st.session_state.mcp_tools:
            tool_name = tool.get("name", "")
            if tool_name and tool_name not in st.session_state.active_tools:
                print(f"Initializing new tool: {tool_name}")
                st.session_state.active_tools[tool_name] = True
    else:
        print(f"MCP is not online, status: {st.session_state.mcp_status['status']}")

        # If we have active_tools but no mcp_tools, create dummy tools
        if st.session_state.active_tools and not st.session_state.mcp_tools:
            print(f"Creating dummy tools from active_tools ({len(st.session_state.active_tools)} tools)")
            dummy_tools = []
            for tool_name, is_active in st.session_state.active_tools.items():
                # Only include active tools
                if is_active:
                    dummy_tools.append({
                        "name": tool_name,
                        "description": f"Tool from configuration: {tool_name}"
                    })
            st.session_state.mcp_tools = dummy_tools

# Main content with two columns
col1, col2 = st.columns(2)

# Column 1: MCP Configuration
with col1:
    st.subheader("MCP Service Configuration")

    # MCP URL input
    mcp_url = st.text_input("MCP Service URL", value=st.session_state.mcp_url.strip().rstrip('/'), key="mcp_url_input")

    # Update button
    if st.button("Update Connection", use_container_width=True):
        sanitized_url = mcp_url.strip().rstrip('/')
        st.session_state.mcp_url = sanitized_url
        update_mcp_url()

    # Check status on page load
    if st.session_state.mcp_status.get("status", "unknown") == "unknown":
        update_mcp_url()

    # If we have active tools but no mcp_tools, try to load them
    if st.session_state.active_tools and not st.session_state.mcp_tools:
        st.session_state.mcp_tools = get_mcp_tools(st.session_state.mcp_url)

    # Always check the actual MCP status before displaying
    with st.spinner("Checking MCP status..."):
        # Check the actual MCP status
        current_status = check_mcp_status(st.session_state.mcp_url)
        st.session_state.mcp_status = current_status

        # Save the status to the backend
        if "config_loaded" in st.session_state and st.session_state.get("authenticated", False):
            config = {
                "mcp_url": st.session_state.mcp_url,
                "active_tools": st.session_state.active_tools,
                "remote_agents_enabled": st.session_state.remote_agents_enabled,
                "workflow_enabled": st.session_state.workflow_enabled,
                "mcp_status": current_status
            }
            save_mcp_config(config)

        # Display the current status
        if current_status["status"] == "online":
            st.success("MCP Service is online")
            if "details" in current_status and isinstance(current_status["details"], dict):
                with st.expander("Service Details"):
                    st.json(current_status["details"])
        elif current_status["status"] == "error":
            st.warning(f"MCP Service has errors: {current_status['details'].get('message', 'Unknown error')}")
        else:
            st.error(f"MCP Service is offline: {current_status['details'].get('message', 'Unknown error')}")

        # Add a button to check the status manually
        if st.button("Check Status Again", key="check_mcp_status_again"):
            # Check the actual MCP status with sanitized URL
            sanitized_url = st.session_state.mcp_url.split('#')[0].strip().rstrip('/')
            new_status = check_mcp_status(sanitized_url)
            st.session_state.mcp_status = new_status
            st.rerun()

    # Additional configuration options
    st.subheader("Service Options")

    # Backend URL
    backend_url = st.text_input("Backend API URL", value=BACKEND_URL, disabled=True)
    st.caption("The backend URL is configured in the environment and cannot be changed here.")

    # Authentication toggle
    auth_enabled = st.toggle("Authentication Required", value=AUTH_ENABLED, disabled=True)
    st.caption("Authentication settings are configured in the environment and cannot be changed here.")

    # Advanced features section
    st.subheader("Advanced Features")

    # Remote Agents toggle
    remote_agents_enabled = st.toggle("Enable Remote Agents", value=st.session_state.remote_agents_enabled, key="remote_agents_toggle")
    st.session_state.remote_agents_enabled = remote_agents_enabled
    st.caption("Enable integration with remote agent services that extend MCP capabilities.")

    # Workflow toggle
    workflow_enabled = st.toggle("Enable Workflow Orchestration", value=st.session_state.workflow_enabled, key="workflow_toggle")
    st.session_state.workflow_enabled = workflow_enabled
    st.caption("Enable workflow orchestration for creating and executing complex workflows.")

# Column 2: Tools Management
with col2:
    st.subheader("MCP Tools Management")

    if not st.session_state.mcp_tools:
        st.info("No tools available. Check the MCP service connection.")
    else:
        # Add search box for filtering tools
        search_query = st.text_input("Search tools", key="tool_search", placeholder="Type to filter tools...")

        # Initialize filtered tools list
        filtered_tools = st.session_state.mcp_tools

        # Filter tools based on search query
        if search_query:
            filtered_tools = [tool for tool in st.session_state.mcp_tools
                             if search_query.lower() in tool.get("name", "").lower()
                             or search_query.lower() in tool.get("description", "").lower()]

            # Show how many tools match the search
            if len(filtered_tools) == 0:
                st.warning(f"No tools match '{search_query}'")
            else:
                st.success(f"Found {len(filtered_tools)} tools matching '{search_query}'")
        # Group tools by category
        tool_categories = {
            "Authentication": [],
            "Agent": [],
            "Document": [],
            "Chat": [],
            "Transcription": [],
            "Filesystem": [],
            "Workflow": [],
            "Remote Agent": [],
            "Other": []
        }

        # Categorize tools
        for tool in filtered_tools:
            name = tool.get("name", "")
            if name.startswith(("login", "signup", "auth")):
                category = "Authentication"
            elif name.startswith(("agent", "create_agent", "list_agents")):
                category = "Agent"
            elif name.startswith(("document", "list_documents", "search_documents")):
                category = "Document"
            elif name.startswith(("chat", "get_chat")):
                category = "Chat"
            elif name.startswith(("transcribe", "translate", "export")):
                category = "Transcription"
            elif name.startswith(("file", "list_files", "read_", "write_", "search_files")):
                category = "Filesystem"
            elif name.startswith(("workflow", "create_sequential", "create_parallel", "create_loop", "execute_workflow", "list_workflows", "get_workflow", "delete_workflow")):
                category = "Workflow"
            elif name.startswith(("remote_agent", "list_remote_agents", "get_remote_agent", "invoke_remote_agent")):
                category = "Remote Agent"
            else:
                category = "Other"

            tool_categories[category].append(tool)

        # Add global Select All / Deselect All buttons
        global_col1, global_col2 = st.columns(2)
        with global_col1:
            if st.button("Select All Tools", key="select_all_global", use_container_width=True):
                for tool in st.session_state.mcp_tools:
                    st.session_state.active_tools[tool.get("name", "")] = True
        with global_col2:
            if st.button("Deselect All Tools", key="deselect_all_global", use_container_width=True):
                for tool in st.session_state.mcp_tools:
                    st.session_state.active_tools[tool.get("name", "")] = False

        # Create a scrollable container for tools
        with st.container(height=500, border=False):
            # Display tools by category
            for category, tools in tool_categories.items():
                if tools:  # Only show categories with tools
                    with st.expander(f"{category} Tools ({len(tools)})", expanded=True):
                        # Add Select All / Deselect All buttons
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(f"Select All {category}", key=f"select_all_{category}", use_container_width=True):
                                for tool in tools:
                                    st.session_state.active_tools[tool.get("name", "")] = True
                        with col2:
                            if st.button(f"Deselect All {category}", key=f"deselect_all_{category}", use_container_width=True):
                                for tool in tools:
                                    st.session_state.active_tools[tool.get("name", "")] = False
                        for tool in tools:
                            tool_name = tool.get("name", "")
                            tool_desc = tool.get("description", "No description available")

                            # Initialize tool state if not present
                            if tool_name not in st.session_state.active_tools:
                                st.session_state.active_tools[tool_name] = True

                            # Create a row for each tool
                            cols = st.columns([4, 1])
                            with cols[0]:
                                st.markdown(f"**{tool_name}**")
                                st.caption(tool_desc)
                            with cols[1]:
                                is_active = st.toggle("Active", value=st.session_state.active_tools.get(tool_name, True), key=f"toggle_{tool_name}")
                                if is_active != st.session_state.active_tools.get(tool_name, True):
                                    toggle_tool(tool_name, is_active)

                            st.divider()

# Footer with save button
st.markdown("---")

# Prepare configuration data
config = {
    "mcp_url": st.session_state.mcp_url,
    "active_tools": st.session_state.active_tools,
    "remote_agents_enabled": st.session_state.remote_agents_enabled,
    "workflow_enabled": st.session_state.workflow_enabled,
    "mcp_status": st.session_state.mcp_status
}

# Show configuration preview
with st.expander("Configuration Preview", expanded=False):
    st.json(config)

# Save and Reset buttons
col1, col2 = st.columns(2)

# Save button
with col1:
    if st.button("Save Configuration", type="primary", use_container_width=True):
        try:
            # Only save if authenticated
            if AUTH_ENABLED and not st.session_state.get("authenticated", False):
                st.error("You must be logged in to save configuration.")
                st.stop()

            # Get the auth token
            headers = st.session_state.get("_request_headers_", {})

            # If headers are empty or don't contain Authorization, try to get the token directly
            if not headers or "Authorization" not in headers:
                token = st.session_state.get("_auth_token_")
                if token:
                    headers = {"Authorization": f"Bearer {token}"}
                else:
                    st.error("Authentication token not found. Please log in again.")
                    st.stop()

            # Call the backend API to save the configuration
            response = requests.post(
                f"{BACKEND_URL}/mcp/config",
                json={
                    "mcp_url": config["mcp_url"],
                    "active_tools": config["active_tools"],
                    "remote_agents_enabled": config["remote_agents_enabled"],
                    "workflow_enabled": config["workflow_enabled"],
                    "mcp_status": config["mcp_status"]
                },
                headers=headers
            )

            if response.status_code == 200:
                st.success("Configuration saved successfully to the backend!")
                # Show a toast notification
                st.toast("MCP configuration updated", icon="✅")
            else:
                st.error(f"Failed to save configuration: {response.text}")
        except Exception as e:
            st.error(f"Error saving configuration: {str(e)}")

# Reset button
with col2:
    if st.button("Reset Configuration", type="secondary", use_container_width=True):
        try:
            # Only reset if authenticated
            if AUTH_ENABLED and not st.session_state.get("authenticated", False):
                st.error("You must be logged in to reset configuration.")
                st.stop()

            # Get the auth token
            headers = st.session_state.get("_request_headers_", {})

            # If headers are empty or don't contain Authorization, try to get the token directly
            if not headers or "Authorization" not in headers:
                token = st.session_state.get("_auth_token_")
                if token:
                    headers = {"Authorization": f"Bearer {token}"}
                else:
                    st.error("Authentication token not found. Please log in again.")
                    st.stop()

            # Call the backend API to delete the configuration
            response = requests.delete(f"{BACKEND_URL}/mcp/config", headers=headers)

            if response.status_code == 200:
                # Reset session state
                st.session_state.mcp_url = "http://localhost:8001"
                st.session_state.active_tools = {}
                st.session_state.config_loaded = False

                st.success("Configuration reset successfully!")
                # Show a toast notification
                st.toast("MCP configuration reset", icon="✅")
                # Rerun to refresh the page
                st.rerun()
            else:
                st.error(f"Failed to reset configuration: {response.text}")
        except Exception as e:
            st.error(f"Error resetting configuration: {str(e)}")
