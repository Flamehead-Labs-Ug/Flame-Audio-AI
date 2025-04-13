import streamlit as st

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Flame Audio Documents",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import requests
from datetime import datetime
import os
import sys
import streamlit_antd_components as sac

from dotenv import load_dotenv

# Add the parent directory to the path so we can import from the root
sys.path.append(os.path.abspath('.'))

# Import from flameaudio.py in the pages directory
from pages.flameaudio import BACKEND_URL, AUTH_ENABLED, load_agents
from authentication.auth_forms import auth_forms, handle_auth_callback, init_auth_session

# Load environment variables for any other env vars still needed
load_dotenv()

# Initialize authentication session and handle callback
init_auth_session()
handle_auth_callback()

# Page title
st.title("Documents")

with st.sidebar:
    st.title("Flame Audio AI: Documents")

# Navigation menu (always visible)
with st.sidebar:
    sac.menu([
        sac.MenuItem('Home', icon='house-fill', href='/flamehome'),
	    sac.MenuItem('Playground', icon='mic-fill', href='/flameaudio'),
        sac.MenuItem('Agents', icon='person-fill', href='/agents'),
        sac.MenuItem('Documents', icon='file-text-fill'),
        sac.MenuItem('Chat', icon='chat-fill', href='/chat'),
    ], open_all=True)
# Show authentication forms if not authenticated
if AUTH_ENABLED and not st.session_state.get("authenticated", False):
    with st.sidebar:
        auth_forms()


# Sidebar for agent filtering


    # Display agent selection if authenticated
with st.sidebar:
    with st.expander("Agent Settings", expanded=True):
        if not AUTH_ENABLED or st.session_state.get("authenticated", False):
            # Fetch agents for the current user
            if "agents" not in st.session_state or st.button("Refresh Agents"):
                with st.spinner("Loading agents..."):
                    try:
                        response = requests.get(
                            f"{BACKEND_URL}/db/agents",
                            headers={
                                "Authorization": f"Bearer {st.session_state.get('_auth_token_', '')}"
                            },
                            timeout=10
                        )

                        if response.status_code == 200:
                            st.session_state.agents = response.json()
                        else:
                            st.error(f"Failed to fetch agents: {response.text}")
                            st.session_state.agents = []
                    except Exception as e:
                        st.error(f"Error fetching agents: {str(e)}")
                        st.session_state.agents = []

            # Display agent selection dropdown
            agent_options = [{"label": "All Documents", "value": "all"}]
            if "agents" in st.session_state and st.session_state.agents:
                for agent in st.session_state.agents:
                    agent_options.append({
                        "label": agent.get("name", "Unnamed Agent"),
                        "value": agent.get("id", "")
                    })

            # Set default agent if not already set
            if "selected_agent" not in st.session_state:
                st.session_state.selected_agent = "all"

            # Agent selection
            selected_agent = st.selectbox(
                "Select Agent",
                options=[a["value"] for a in agent_options],
                format_func=lambda x: next((a["label"] for a in agent_options if a["value"] == x), x),
                index=next((i for i, a in enumerate(agent_options) if a["value"] == st.session_state.selected_agent), 0),
                key="agent_selector"
            )

            # Update selected agent
            if selected_agent != st.session_state.selected_agent:
                st.session_state.selected_agent = selected_agent
                # Force refresh of documents
                st.session_state.loading_documents = True
                st.experimental_rerun()
        else:
            st.info("Please sign in to access agent settings")

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


# Sign out is now handled by the logout() function from auth_forms.py

# Main content area
if not AUTH_ENABLED or st.session_state.get("authenticated", False):
    # Set up a container for the documents list
    documents_container = st.container()

    with documents_container:
        # Refresh button and loading indicator
        col1, col2 = st.columns([9, 1])
        with col2:
            if st.button("Refresh", key="refresh_documents", use_container_width=True):
                st.session_state.loading_documents = True
                st.experimental_rerun()

        # Load documents if needed
        if "loading_documents" not in st.session_state:
            st.session_state.loading_documents = True

        if st.session_state.loading_documents:
            with st.spinner("Loading documents..."):
                try:
                    # Construct the API endpoint with optional agent filter
                    agent_filter = f"?agent_id={st.session_state.selected_agent}" if st.session_state.selected_agent != "all" else ""

                    # Fetch documents from the API
                    response = requests.get(
                        f"{BACKEND_URL}/db/documents{agent_filter}",
                        headers={
                            "Authorization": f"Bearer {st.session_state.get('_auth_token_', '')}"
                        },
                        timeout=10
                    )

                    if response.status_code == 200:
                        st.session_state.documents = response.json()
                    else:
                        st.error(f"Failed to fetch documents: {response.text}")
                        st.session_state.documents = []
                except Exception as e:
                    st.error(f"Error fetching documents: {str(e)}")
                    st.session_state.documents = []

                st.session_state.loading_documents = False

        # Display documents in a table
        if "documents" in st.session_state and st.session_state.documents:
            # Format the data for display
            table_data = []
            for doc in st.session_state.documents:
                # Format creation date
                created_at = datetime.fromisoformat(doc["created_at"].replace("Z", "+00:00")) if "created_at" in doc else datetime.now()
                formatted_date = created_at.strftime("%Y-%m-%d %H:%M")

                # Description handling
                description = "None"
                if doc.get("description") and doc["description"] != "null":
                    description = doc["description"]
                    if len(description) > 20:
                        description = description[:20] + "..."

                # Extract needed fields
                table_data.append({
                    "ID": doc["id"][:8] + "...",  # Truncate ID for display
                    "Document Name": doc.get("document_name", "Unknown"),
                    "Type": doc.get("file_type", "Unknown"),
                    "Task": doc.get("task", "transcribe").capitalize(),
                    "Language": doc.get("language", "auto-detect"),
                    "Vector Status": doc.get("vector_status", "Not Indexed"),
                    "Created": formatted_date,
                    "Description": description,
                    "Delete": False,  # Add Delete checkbox field with default value of False
                    "Full ID": doc["id"]  # Hidden column for delete action
                })

            # Create dataframe
            df = pd.DataFrame(table_data)

            # Function to delete document
            def delete_document(doc_id):
                try:
                    # Make API request to delete document
                    response = requests.delete(
                        f"{BACKEND_URL}/db/documents/{doc_id}",
                        headers={
                            "Authorization": f"Bearer {st.session_state.get('_auth_token_', '')}"
                        },
                        timeout=10
                    )

                    if response.status_code == 200:
                        # Remove from session state and show success message
                        st.session_state.documents = [doc for doc in st.session_state.documents if doc["id"] != doc_id]
                        return True
                    else:
                        st.error(f"Failed to delete document: {response.text}")
                        return False
                except Exception as e:
                    st.error(f"Error deleting document: {str(e)}")
                    return False

            # Create interactive table with delete buttons
            edited_df = st.data_editor(
                df.drop(columns=["Full ID"]),  # Don't show the Full ID column
                column_config={
                    "ID": st.column_config.TextColumn("ID", width="small"),
                    "Document Name": st.column_config.TextColumn("Document Name", width="medium"),
                    "Type": st.column_config.TextColumn("Type", width="small"),
                    "Task": st.column_config.TextColumn("Task", width="small"),
                    "Language": st.column_config.TextColumn("Language", width="small"),
                    "Vector Status": st.column_config.TextColumn("Vector Status", width="small"),
                    "Created": st.column_config.TextColumn("Created", width="small"),
                    "Description": st.column_config.TextColumn("Description", width="medium"),
                    "Delete": st.column_config.CheckboxColumn("Delete", default=False, width="small"),
                },
                hide_index=True,
                use_container_width=True,
            )

            # Handle document deletion
            if st.button("Delete Selected Documents", key="delete_docs"):
                if "Delete" in edited_df.columns and edited_df["Delete"].any():
                    with st.spinner("Deleting selected documents..."):
                        # Get IDs of documents to delete
                        docs_to_delete = []
                        for i, row in edited_df.iterrows():
                            if row.get("Delete", False):
                                docs_to_delete.append(df.iloc[i]["Full ID"])

                        # Delete each document
                        success_count = 0
                        for doc_id in docs_to_delete:
                            if delete_document(doc_id):
                                success_count += 1

                        if success_count > 0:
                            st.success(f"Successfully deleted {success_count} document(s)")
                            # Refresh the documents list
                            st.session_state.loading_documents = True
                            st.experimental_rerun()
                else:
                    st.warning("No documents selected for deletion")
        else:
            st.info("No documents found. Start transcribing audio to create your first document!")
else:
    st.warning("Please sign in to view your documents.")
