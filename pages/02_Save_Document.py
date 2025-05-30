import streamlit as st
# Page title and description
# Page title and description
st.set_page_config(page_title="Save Document", page_icon="📊", layout="wide")
import streamlit_antd_components as sac




import os
import sys
import json
import requests
import time
import tempfile
import traceback

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.abspath('.'))

# Try importing from flameaudio directly - no fallback since streamlit_app.py no longer exists
from pages.flameaudio import load_agents, save_document_to_database, BACKEND_URL, AUTH_ENABLED

# Add a back button at the top of the page
back_col1, back_col2 = st.columns([1, 9])
with back_col1:
    if st.button("⬅️ Back to Flame Audio", key="back_btn"):
        # Instead of direct switch_page, set a flag and use experimental_rerun
        st.session_state.navigate_to_main = True

with back_col2:
    st.title("Save Document to Vector Store")

# Debug session state - Now restored as requested
# st.write("Debug - Available session state keys:", list(st.session_state.keys()))

# Commented out debug information in sidebar
# st.sidebar.write("### Session State Debug")
# st.sidebar.write("Available keys in session state:")
# st.sidebar.json([key for key in st.session_state.keys()])

# Commented out agent ID debug information
# if "current_agent_id" in st.session_state:
#     st.sidebar.success(f"Found agent ID: {st.session_state.current_agent_id}")
# else:
#     st.sidebar.warning("No agent ID found in session state")
#     # Try to find it under alternative names
#     for possible_key in ['agent_id', 'selected_agent_id']:
#         if possible_key in st.session_state:
#             st.sidebar.info(f"Found alternative key: {possible_key} = {st.session_state[possible_key]}")

# Check if transcription results exist in session state
if "transcription_result" not in st.session_state or st.session_state.get("transcription_result") is None:
    # Try to get data from the _temp_transcription variable
    if "_temp_transcription" in st.session_state and st.session_state.get("_temp_transcription") is not None:
        st.session_state["transcription_result"] = st.session_state["_temp_transcription"]
        st.success("Successfully loaded transcription data")
    # Try to load from temp file (legacy approach)
    elif "temp_transcription_path" in st.session_state and os.path.exists(st.session_state["temp_transcription_path"]):
        try:
            # Load transcription data from the temporary file
            with open(st.session_state["temp_transcription_path"], "r") as f:
                st.session_state["transcription_result"] = json.load(f)
            st.success("Successfully loaded transcription data from temporary file")
        except Exception as e:
            st.error(f"Failed to load transcription data from temporary file: {str(e)}")
            st.stop()
    # Check for our persistent transcription variable as fallback
    elif "_persistent_transcription" in st.session_state and st.session_state.get("_persistent_transcription") is not None:
        # Restore it to the normal location
        st.session_state["transcription_result"] = st.session_state["_persistent_transcription"]
        st.success("Successfully loaded transcription data from session state")
    else:
        st.warning("No transcription results found. Please transcribe content on the main page first.")
        st.stop()

# Ensure transcription data is available for debugging
transcription_text = ""
if "transcription_result" in st.session_state and st.session_state["transcription_result"] is not None:
    # Get the full text from the transcription result
    for segment in st.session_state["transcription_result"].get("segments", []):
        if segment.get("text"):
            transcription_text += segment.get("text") + " "

# Show a preview of the transcription at the top
with st.expander("Transcription Preview", expanded=False):
    st.text_area("Content to be saved", transcription_text[:500] + "..." if len(transcription_text) > 500 else transcription_text, height=100)

# Create a two-column layout for the form
col1, col2 = st.columns([1, 1])

with col1:
    # Basic information section
    st.subheader("Document Information")

    # Document name field (required)
    doc_name = st.text_input("Document Name (required)",
                          st.session_state.get("last_filename", "").split("/")[-1].split("\\")[-1].split(".")[0] if st.session_state.get("last_filename", "") else "")

    # Document description field (optional)
    doc_description = st.text_area("Document Description (optional)",
                                 height=100,
                                 help="Add a description to help you find this document later")

    # Display the currently selected agent
    if "current_agent_id" in st.session_state:
        # Get the agent information
        if st.session_state.current_agent_id is not None:
            # Find the agent name from the loaded agents
            current_agent_name = "All Agents (No Specific Agent)"
            for agent in load_agents():
                if agent["id"] == st.session_state.current_agent_id:
                    current_agent_name = agent["name"]
                    break

            # Show the agent selection information in the UI
            st.info(f" Document will be associated with agent: **{current_agent_name}**")
            st.caption("To change the agent, select a different one from the sidebar.")
        else:
            st.info(" Document will not be associated with any specific agent.")
            st.caption("To associate this document with an agent, select one from the sidebar.")
    else:
        st.info(" Document will not be associated with any specific agent.")
        st.caption("To associate this document with an agent, select one from the sidebar.")

with col2:
    # Vector store configuration section
    st.subheader("Vector Store Configuration")
    st.info("Documents will be stored in Qdrant using collections named after the document ID.")

    # Add embedding model configuration
    with st.expander("Embedding Model Settings", expanded=True):
        # Fetch available embedding models from API
        try:
            embedding_models_response = requests.get(f"{BACKEND_URL}/vectorstore/models")
            if embedding_models_response.status_code == 200:
                embedding_models = embedding_models_response.json()
                model_names = [model["name"] for model in embedding_models]
                model_info = {model["name"]: model for model in embedding_models}
            else:
                # Fallback to default models if API fails
                model_names = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-multilingual-MiniLM-L12-v2"]
                model_info = {}
        except Exception as e:
            st.warning(f"Could not fetch embedding models: {e}. Using defaults.")
            model_names = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "paraphrase-multilingual-MiniLM-L12-v2"]
            model_info = {}

        # Get current vector store settings
        try:
            vs_settings_response = requests.get(f"{BACKEND_URL}/vectorstore/settings",
                                            headers={"Authorization": f"Bearer {st.session_state.get('_auth_token_', '')}"} if AUTH_ENABLED and st.session_state.get("authenticated", False) else {})
            if vs_settings_response.status_code == 200:
                vs_settings = vs_settings_response.json()
            else:
                # Fallback to default settings
                vs_settings = {
                    "embedding_model": "all-MiniLM-L6-v2",
                    "similarity_threshold": 0.2,
                    "chunk_size": 1000,
                    "chunk_overlap": 200,
                    "match_count": 10,
                    "enabled": True
                }
        except Exception as e:
            st.warning(f"Could not fetch vector store settings: {e}. Using defaults.")
            vs_settings = {
                "embedding_model": "all-MiniLM-L6-v2",
                "similarity_threshold": 0.2,
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "match_count": 10,
                "enabled": True
            }

        # Create model selection
        selected_embedding_model = st.selectbox(
            "Embedding Model",
            options=model_names,
            index=model_names.index(vs_settings.get("embedding_model", "all-MiniLM-L6-v2")) if vs_settings.get("embedding_model") in model_names else 0,
            help="Select the model to use for generating embeddings."
        )

        # Add a note about what this means
        st.info("The selected embedding model will be used to encode the text for semantic search. Higher dimension models may provide better search accuracy but use more memory.")

        # Add chunking configuration with sliders
        chunk_size = st.slider(
            "Chunk Size",
            min_value=100,
            max_value=2000,
            value=int(vs_settings.get("chunk_size", 1000)),
            step=100,
            help="Size of text chunks for embedding (larger chunks provide more context but less precision)"
        )

        chunk_overlap = st.slider(
            "Chunk Overlap",
            min_value=0,
            max_value=500,
            value=int(vs_settings.get("chunk_overlap", 200)),
            step=50,
            help="Overlap between chunks to ensure context continuity"
        )

# Add common elements to the sidebar
with st.sidebar:
    st.title("Flame Audio AI: Save Document")

# Navigation menu (always visible)
with st.sidebar:
    sac.menu([
        sac.MenuItem('Home', icon='house-fill', href='/flamehome'),
	    sac.MenuItem('Playground', icon='mic-fill', href='/flameaudio'),
        sac.MenuItem('Documents', icon='file-text-fill', href='/documents'),
        #sac.MenuItem('Chat', icon='chat-fill', href='/chat'),
        sac.MenuItem('MCP', icon='gear-fill', href='/flame_mcp'),
        sac.MenuItem('Flame Audio Chat', icon='chat-dots-fill', href='/mcp_chat'),
    ], open_all=True)




with st.sidebar:
    # User Profile Container at the bottom of sidebar
    st.sidebar.markdown("## User Profile")
    user_profile_container = st.sidebar.container(border=True)
    with user_profile_container:
        if AUTH_ENABLED and st.session_state.get("authenticated", False) and "user" in st.session_state:
            email = st.session_state['user'].get('email', '')
            st.markdown(f"**Signed in as:**")
            st.info(email)
            if st.button("Sign Out", key="save_doc_sign_out_btn", use_container_width=True):
                # Use the proper logout function from auth_forms.py
                from authentication.auth_forms import logout
                logout()


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

# Sign out is now handled by the logout() function from auth_forms.py

# Check if we should navigate back to main
if st.session_state.get('navigate_to_main', False):
    # Clear the flag
    st.session_state.navigate_to_main = False
    # Try to detect which main file exists and navigate to it
    main_file = None
    if os.path.exists('pages/flameaudio.py'):
        main_file = 'pages/flameaudio.py'

    if main_file:
        st.switch_page(main_file)
    else:
        st.error("Could not find main application file. Please check your installation.")

# Create a status area for messages
status_area = st.empty()

# Add action buttons in a row
save_col1, save_col2 = st.columns(2)

with save_col1:
    # Option to enable/disable vector storage
    vectorize_enabled = st.toggle(
        "Enable Vector Search",
        value=True,
        help="Store embeddings for semantic search. Disable for faster saving but no semantic search capability."
    )

# Initialize indexing status in session state if not present
if 'document_indexed' not in st.session_state:
    st.session_state.document_indexed = False

# Add the index button and save button based on indexing status
if not st.session_state.document_indexed:
    with save_col2:
        index_btn = st.button("Index & Save Document", type="primary", use_container_width=True,
                            help="Index document to Qdrant and save metadata to database in one step")

    # Process indexing when the Index Document button is clicked
    if index_btn:
        # Check if transcription result exists and has segments
        if "transcription_result" not in st.session_state or not st.session_state.get("transcription_result"):
            status_area.error("No transcription results found to index. Please transcribe content first.")
            st.stop()

        # Check if segments exist in the transcription result
        segments = st.session_state.get("transcription_result", {}).get("segments", [])
        if not segments:
            status_area.warning("No segments found in transcription results. Nothing to index.")
            st.session_state.document_indexed = True  # Still allow saving metadata
            st.rerun()

        if not doc_name.strip():
            status_area.error("Document name is required")
        elif AUTH_ENABLED and st.session_state.get("authenticated", False) and '_auth_token_' in st.session_state:
            user_id = None

            # Get user ID from token
            try:
                user_id = st.session_state.get("user", {}).get("id")
                if not user_id:
                    status_area.error("Could not retrieve user ID from session. Please log in again.")
                    st.stop()
            except Exception as e:
                status_area.error(f"Authentication error: {e}")
                st.stop()

            # Create vector store settings for the request
            vector_store_settings = {
                "embedding_model": selected_embedding_model,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "enabled": vectorize_enabled
            }

            # Get agent ID if available
            if "current_agent_id" in st.session_state:
                agent_id = st.session_state.current_agent_id
                # Find agent name if available
                agent_name = "Default Agent"
                for agent in load_agents():
                    if agent["id"] == agent_id:
                        agent_name = agent["name"]
                        break
            else:
                agent_id = None
                agent_name = "Default Agent"

            if not vectorize_enabled:
                status_area.warning("Vector search is disabled. No indexing will be performed.")

            # COMBINED APPROACH: Index and save in one step
            try:
                # Make sure selected_language is defined
                selected_language = None
                if "selected_language" in st.session_state:
                    selected_language = st.session_state.get("selected_language")

                # Create a progress indicator
                combined_progress = st.progress(0)

                with st.spinner("Indexing and saving document..."):
                    # This is a simplified approach that does both indexing and saving
                    success = save_document_to_database(
                        st.session_state.get("transcription_result"),
                        doc_name,
                        user_id,
                        selected_language if st.session_state.get("current_task") == "transcribe" else None,
                        st.session_state.get("selected_model"),
                        agent_id=agent_id,
                        status_area=status_area,
                        vector_store_settings=vector_store_settings
                    )

                    # Update progress to show completion
                    combined_progress.progress(1.0)

                if success:
                    # Show success message
                    status_area.success("✅ Document successfully indexed and saved to database!")
                    st.session_state.document_indexed = True
                    # Show a balloons celebration
                    st.balloons()
                else:
                    status_area.error("Error saving document. Please check the debug information.")
            except Exception as e:
                status_area.error(f"Error saving document: {str(e)}")
                st.code(traceback.format_exc())
        else:
            status_area.error("Please sign in to index and save documents")
else:
    # Show reset button to redo the process if needed
    reset_col1, reset_col2 = st.columns([3, 1])
    with reset_col2:
        if st.button("Start Over", type="secondary", use_container_width=True):
            st.session_state.document_indexed = False
            st.rerun()

    with reset_col1:
        st.success("✅ Document has been successfully indexed and saved to the database!")


# Define footer HTML
footer = """
<footer style="margin-top: 5rem; padding: 2.5rem 0; border-top: 1px solid rgba(0,0,0,0.05); width: 100%;">
    <div style="max-width: 1200px; margin: 0 auto; padding: 0 1.5rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div style="display: flex; align-items: center;">
            <div style="font-size: 0.9rem; color: #6B7280; margin-right: 1rem;">
                2025 FlameheadLabs
            </div>
        </div>
        <div style="font-size: 0.85rem; color: #9CA3AF;">
            <a href="privacy.html" target="_blank" style="color: #6B7280; margin-right: 1rem; text-decoration: none;">Privacy Policy</a>
            <a href="terms.html" target="_blank" style="color: #6B7280; text-decoration: none;">Terms of Service</a>
        </div>
    </div>
</footer>
"""

st.markdown(footer, unsafe_allow_html=True)
