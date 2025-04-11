import streamlit as st
import requests
import os
from dotenv import load_dotenv
import streamlit_antd_components as sac
from authentication.auth_forms import auth_forms, handle_auth_callback, init_auth_session

# Page configuration
st.set_page_config(
    page_title="Flame Audio AI: Agents",
    page_icon="logos/flame logo.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize authentication session and handle callback
init_auth_session()
handle_auth_callback()

# Load environment variables
load_dotenv()

# Set up backend URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Check if authentication is enabled
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() == "true"

# Sidebar configuration
st.sidebar.title("Flame Audio AI: Agents")

# Navigation menu (always visible)
with st.sidebar:
    sac.menu([
        sac.MenuItem('Home', icon='house-fill', href='/flamehome'),
        sac.MenuItem('Playground', icon='mic-fill', href='/flameaudio'),
        sac.MenuItem('Agents', icon='person-fill'),
        sac.MenuItem('Documents', icon='file-text-fill', href='/documents'),
        sac.MenuItem('Chat', icon='chat-fill', href='/chat'),
    ], open_all=True)

# Show authentication forms if not authenticated
if AUTH_ENABLED and not st.session_state.get("authenticated", False):
    with st.sidebar:
        auth_forms()



# Main content area
st.title("Flame Audio AI: Agents")
st.subheader("Manage your AI agents for transcription, translation, and chat")

# Function to load agents from the database
def load_agents():
    if not AUTH_ENABLED or not st.session_state.get("authenticated", False):
        return []

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

# Function to create a new agent
def create_new_agent():
    st.session_state.agent_name = ""
    st.session_state.system_message = ""
    st.session_state.current_agent_id = None
    st.switch_page("pages/flameaudio.py")

# Function to view/edit an existing agent
def view_agent(agent_id):
    # Set the agent ID in session state so flameaudio.py can load it
    st.session_state.current_agent_id = agent_id
    
    # Get agent details to set in session state
    agent_data = get_agent_details(agent_id)
    if agent_data:
        st.session_state.agent_name = agent_data.get("name", "")
        st.session_state.system_message = agent_data.get("system_message", "")
        
        # Switch to the playground page
        st.switch_page("pages/flameaudio.py")

# Function to chat with an agent
def chat_with_agent(agent_id):
    # Set the chat agent in session state so chat.py can load it
    st.session_state.chat_agent = agent_id
    
    # Switch to the chat page
    st.switch_page("pages/chat.py")

# Check if user is authenticated
if AUTH_ENABLED and not st.session_state.get("authenticated", False):
    st.warning("Please sign in to view and manage your agents.")
else:
    # Create a button to create a new agent
    if st.button("Create New Agent", type="primary", key="create_agent_btn"):
        create_new_agent()
    
    # Load agents
    agents = load_agents()
    
    if not agents:
        st.info("You don't have any agents yet. Click 'Create New Agent' to get started.")
    else:
        st.markdown("### Your Agents")
        
        # Create a grid layout for agent cards
        # Calculate number of columns (3 for desktop, fewer for smaller screens)
        cols = st.columns(3)
        
        # Display each agent as a card
        for i, agent in enumerate(agents):
            col_idx = i % 3
            with cols[col_idx]:
                # Create a card-like container for each agent
                with st.container(border=True):
                    # Agent name as title
                    st.markdown(f"### {agent.get('name', 'Unnamed Agent')}")
                    
                    # Show a preview of the system message
                    system_message = agent.get("system_message", "")
                    preview = system_message[:100] + "..." if len(system_message) > 100 else system_message
                    st.markdown(f"**System Message:** {preview}")
                    
                    # Show agent metadata if available
                    if "created_at" in agent:
                        st.caption(f"Created: {agent.get('created_at', '')}")
                    
                    # Add buttons for actions
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("View Agent", key=f"view_{agent.get('id')}", use_container_width=True):
                            view_agent(agent.get("id"))
                    with col2:
                        if st.button("Chat", key=f"chat_{agent.get('id')}", use_container_width=True):
                            chat_with_agent(agent.get("id"))
        
        # Add some spacing at the bottom
    with st.sidebar:    
        st.markdown("---")
        st.markdown("""
        ### About Agents
        
        Agents are AI assistants that can be customized with specific personalities and knowledge.
        Each agent can be used for:
        
        - **Transcription & Translation**: Process audio files with agent-specific context
        - **Document Organization**: Group related documents under the same agent
        - **Specialized Chat**: Chat with the agent about documents it has access to
        
        Create different agents for different projects or topics to keep your work organized.
        """)

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

# User Profile Container (only show when authenticated)
if not AUTH_ENABLED or st.session_state.get("authenticated", False):
    st.sidebar.markdown("## User Profile")
    user_profile_container = st.sidebar.container(border=True)
    with user_profile_container:
        if "user" in st.session_state:
            email = st.session_state['user'].get('email', '')
            st.markdown(f"**Signed in as:**")
            st.info(email)
            if st.button("Sign Out", key="sign_out_btn", use_container_width=True):
                # Use the proper logout function from auth_forms.py
                from authentication.auth_forms import logout
                logout()
