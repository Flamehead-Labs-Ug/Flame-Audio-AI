import streamlit as st
import os
import requests
import uuid
import tempfile
import io
import time
import json
import base64
import zipfile
import subprocess
import random
import re
from dotenv import load_dotenv
from datetime import datetime
import threading
import queue
import copy
#from database.vector_store import get_vector_store
#from database.vector_store_realtime import initialize_vector_store_realtime

# Configure page settings with title, favicon, and description
st.set_page_config(
    page_title="Flame Audio AI",
    page_icon="logos/flame logo.jpg",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': """# Flame Audio
A powerful speech transcription and translation application built by FlameheadLabs. Convert spoken language to text through recorded audio or uploaded files with support for multiple languages and advanced processing options."""
    }
)


import pandas as pd  # Import pandas
import streamlit_antd_components as sac
from audio_recorder_streamlit import audio_recorder
import uuid
from authentication.auth_forms import auth_forms, handle_auth_callback, init_auth_session, load_session_data, logout

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:8501")  # Default to localhost if not set

# Check if authentication is enabled
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() == "true"

# Check if API key input is enabled (allows hiding the API key field when false)
API_KEY_INPUT_ENABLED = os.getenv("API_KEY_INPUT_ENABLED", "true").lower() == "true"

# FastAPI backend URL - must be specified in .env file
BACKEND_URL = os.getenv("BACKEND_URL")
if not BACKEND_URL:
    st.error("BACKEND_URL not found in environment variables. Please set it in the .env file.")
    st.stop()

# Configure page layout

# Initialize authentication session
init_auth_session()

# Check for authentication callback
handle_auth_callback()

# Auth Debug expander
#with st.sidebar.expander("Auth Debug Info", expanded=False):
    #st.write(f"Authenticated: {st.session_state.get('authenticated', False)}")
    #if st.session_state.get('authenticated', False):
        #st.write(f"User: {st.session_state.get('user')}")

    # Show if session file exists
    #session_token = load_session_data()
    #st.write(f"Session file exists: {session_token is not None}")

    # Display all session state items
    #st.write("### Session State Contents:")
    #for key, value in st.session_state.items():
        # Skip displaying large objects or sensitive information
        #if key in ['_auth_token_', 'transcription_result']:
            #st.write(f"{key}: [Content hidden]")
        #else:
            #st.write(f"{key}: {value}")

    #if st.button("Force Reload"):
        #st.rerun()

# Initialize session state
if 'init_completed' not in st.session_state:
    st.session_state.init_completed = False

# Initialize session state for model if not already set
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None  # No default model

if 'transcription_result' not in st.session_state:
    st.session_state.transcription_result = None  # Initialize result storage

# Initialize default values for advanced settings if not already set
if 'chunk_length' not in st.session_state:
    st.session_state.chunk_length = 600  # Default to 10 minutes
if 'overlap' not in st.session_state:
    st.session_state.overlap = 5  # Default overlap
if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.0  # Default temperature (explicitly set as float)

# Initialize task options in session state
if "task_options" not in st.session_state:
    try:
        # Try to get task options from backend
        response = requests.get(f"{BACKEND_URL}/tasks")
        if response.status_code == 200:
            st.session_state.task_options = response.json()["tasks"]
        else:
            # Fallback to defaults if backend doesn't have this endpoint
            st.session_state.task_options = ["transcribe", "translate"]
    except Exception:
        # Fallback to defaults if backend request fails
        st.session_state.task_options = ["transcribe", "translate"]

# Initialize current task in session state
if "current_task" not in st.session_state:
    st.session_state.current_task = "transcribe"  # Default task

# Initialize groq_api_key_input variable with a default empty string
groq_api_key_input = ""

# Initialize session state for agents
if "current_agent_id" not in st.session_state:
    st.session_state.current_agent_id = None

# Initialize session state for agent name and system message
if "agent_name" not in st.session_state:
    st.session_state.agent_name = ""

if "system_message" not in st.session_state:
    st.session_state.system_message = ""

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

# Function to save agent to the database
def save_agent(name, system_message, agent_id=None):
    if not AUTH_ENABLED or not st.session_state.get("authenticated", False):
        st.error("You must be logged in to save agents")
        return None

    if not name:
        st.error("Agent name is required")
        return None

    try:
        # Prepare agent data
        agent_data = {
            "name": name,
            "system_message": system_message,
            "settings": {}
        }

        # Add ID if updating existing agent
        if agent_id:
            agent_data["id"] = agent_id

        # Make API request
        response = requests.post(
            f"{BACKEND_URL}/db/agents",
            json=agent_data,
            headers={
                "Authorization": f"Bearer {st.session_state.get('_auth_token_', '')}"
            },
            timeout=10
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("id")
        else:
            st.error(f"Failed to save agent: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error saving agent: {str(e)}")
        return None

# Define function to save to database using the new consolidated endpoint with polling
def save_document_to_database(result, filename, user_id, selected_language, selected_model, progress_bar=None, agent_id=None, status_area=None, vector_store_settings=None):
    """Save a complete document with segments and embeddings in a single operation using backend processing"""
    if not AUTH_ENABLED or not user_id:
        st.warning("Authentication is required to save transcriptions to the database.")
        return False

    try:
        # Add debug logging for agent_id
        if status_area:
            status_area.info(f"Processing document with agent_id: {agent_id}")

        # Gather document data for the API call
        document_data = {
            "filename": filename,
            "task_type": st.session_state.get("current_task", "transcribe"),
            "original_language": selected_language or "auto",
            "model": selected_model,
            "duration": result.get("duration", 0.0),
            "metadata": result.get("metadata", {}),
            "agent_id": agent_id,
            "segments": result.get("segments", []),
            "embedding_settings": vector_store_settings,
            "file_type": result.get("metadata", {}).get("file_extension", "audio")  # Add file type from metadata or default to audio
        }

        # Ensure agent_id is included in embedding metadata for each segment
        if agent_id:
            # Update metadata for each segment to include agent_id
            for segment in document_data["segments"]:
                if "metadata" not in segment:
                    segment["metadata"] = {}
                segment["metadata"]["agent_id"] = agent_id

        # Make API request to start the document saving process
        auth_token = st.session_state.get('_auth_token_', '')
        if not auth_token:
            if status_area:
                status_area.error(" No authentication token found. Please sign in again.")
            return False

        # Use streamlit's spinner for a cleaner UI experience
        with st.spinner("Saving document to database..."):
            try:
                # Send the document data to the server
                response = requests.post(
                    f"{BACKEND_URL}/db/save_document",
                    json=document_data,
                    headers={
                        "Authorization": f"Bearer {auth_token}"
                    },
                    timeout=60
                )

                # Check if the request was successful
                if response.status_code != 200:
                    if status_area:
                        try:
                            error_detail = response.json().get("detail", "Unknown error")
                            status_area.error(f" Error saving document: {error_detail}")
                        except:
                            status_area.error(f" Error saving document: Status code {response.status_code}")
                    return False

                # Extract job ID from response
                response_data = response.json()
                job_id = response_data.get("job_id")

                if not job_id:
                    if status_area:
                        status_area.error(" No job ID returned from server")
                    return False

                # Poll for job completion with simpler logic
                poll_interval = 2.0  # seconds
                max_polls = 150  # maximum number of status checks (5 minutes)
                for _ in range(max_polls):
                    # Sleep before making the request to prevent excessive polling
                    time.sleep(poll_interval)

                    # Get the current job status
                    status_response = requests.get(
                        f"{BACKEND_URL}/db/document_status/{job_id}",
                        headers={
                            "Authorization": f"Bearer {auth_token}"
                        },
                        timeout=5
                    )

                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        job_status = status_data.get("status", "processing")

                        # If job completed successfully, break out of the loop
                        if job_status == "completed":
                            if status_area:
                                status_area.success(" Document saved successfully to the database!")
                            # Show balloons for celebration
                            st.balloons()
                            return True

                # If we've reached here, the maximum number of polls was reached
                if status_area:
                    status_area.warning(" Document processing is taking longer than expected. It may still complete in the background.")
                return False

            except Exception as e:
                if status_area:
                    status_area.error(f" Error during document saving: {str(e)}")
                return False

    except Exception as e:
        # Show any exceptions that occurred
        if status_area:
            status_area.error(f" Error: {str(e)}")
        return False

# Sidebar configuration


# Title after the logo
st.sidebar.title("Flame Audio AI: Playground")

# Navigation menu (always visible)
with st.sidebar:
    sac.menu([
        sac.MenuItem('Home', icon='house-fill', href='/flamehome'),
	    sac.MenuItem('Playground', icon='mic-fill'),
        sac.MenuItem('Documents', icon='file-text-fill', href='/documents'),
        sac.MenuItem('Chat', icon='chat-fill', href='/chat'),
    ], open_all=True)

# Show authentication status based on AUTH_ENABLED setting
if AUTH_ENABLED:
    # Authentication UI only when enabled
    if not st.session_state.get("authenticated", False):
        # Display authentication forms if not authenticated
        with st.sidebar:
            auth_forms()

# Removed the duplicate sign out button here since it's in the user profile container

# Only show API key input and model selection if authenticated or auth is disabled
if st.session_state.get("authenticated", False) or not AUTH_ENABLED:
    # Set up the API key - either from input or env var
    if API_KEY_INPUT_ENABLED:
        # Groq API Key input
        groq_api_key_input = st.sidebar.text_input(
            "Groq API Key",
            type="password",
            value=groq_api_key if groq_api_key else "",
            placeholder="Enter your Groq API key here"
        )
    else:
        # Use the API key from environment variable
        groq_api_key_input = os.getenv("GROQ_API_KEY", "")

    # Function to load available models
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_models(api_key):
        try:
            response = requests.get(f"{BACKEND_URL}/models", headers={"X-API-KEY": api_key})
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                st.error("Invalid API key. Please check your Groq API key.")
                return []
            else:
                st.error(f"Error loading models: {response.text}")
                return []
        except Exception as e:
            st.error(f"Error connecting to backend: {str(e)}")
            return []

    # Initialize models variable
    models = []

    # Only try to load models if we have an API key
    if groq_api_key_input:
        with st.spinner("Loading available models..."):
            models = load_models(groq_api_key_input)
            if not models:
                st.warning("No audio models available. Please check your API key and connection.")

    # Model selection
    if models:
            # Get the model info for the current selection
            selected_model_info = None
            for model in models:
                if model["id"] == st.session_state.get("selected_model"):
                    selected_model_info = model
                    break

            # Get the list of model options
            model_options = [model["id"] for model in models]

            # Find the index of the currently selected model in the options
            default_index = None
            if st.session_state.get("selected_model") in model_options:
                default_index = model_options.index(st.session_state.get("selected_model"))

            # Model selection dropdown
            selected_model = st.sidebar.selectbox(
                "Model",
                options=model_options,
                index=default_index,  # Use the index of the currently selected model
                placeholder="Select a model",
                help="Select the model to use for transcription/translation"
            )

            # Update session state when model is changed
            st.session_state.selected_model = selected_model

            # Show model description if available
            if selected_model_info and "description" in selected_model_info:
                st.sidebar.info(selected_model_info["description"])
    else:
        st.sidebar.error("No audio models available. Please check your API key and connection.")
        selected_model = None
        selected_model_info = None

    # Agents dropdown (only show when authenticated)
    st.sidebar.subheader("Agents")
    # Load agents
    agents = load_agents()
    agent_options = ["Create New Agent"] + [(agent["name"], agent["id"]) for agent in agents]

    # Create selectbox for agents
    selected_agent = st.sidebar.selectbox(
        "Select Agent",
        options=range(len(agent_options)),
        format_func=lambda i: agent_options[i][0] if i > 0 else agent_options[i],
        index=0
    )

    # Handle agent selection
    if selected_agent > 0:
        # User selected an existing agent
        agent_name, agent_id = agent_options[selected_agent]

        # Load agent details if not already loaded or if different agent selected
        if st.session_state.get("current_agent_id") != agent_id:
            try:
                response = requests.get(
                    f"{BACKEND_URL}/db/agents/{agent_id}",
                    headers={
                        "Authorization": f"Bearer {st.session_state.get('_auth_token_', '')}"
                    },
                    timeout=10
                )

                if response.status_code == 200:
                    agent_data = response.json()
                    st.session_state.agent_name = agent_data.get("name", "")
                    st.session_state.system_message = agent_data.get("system_message", "")
                    st.session_state.current_agent_id = agent_id

                    # Force document reload when agent changes
                    st.session_state.loading_documents = True

                    # Complete UI refresh when agent changes
                    st.rerun()
                else:
                    st.error(f"Failed to load agent details: {response.text}")
            except Exception as e:
                st.error(f"Error loading agent details: {str(e)}")
    else:
        # Create new agent selected, reset fields if different from current
        if st.session_state.get("current_agent_id") is not None:
            st.session_state.agent_name = ""
            st.session_state.system_message = ""
            st.session_state.current_agent_id = None

            # Force document reload when agent changes
            st.session_state.loading_documents = True

            # Complete UI refresh when agent is cleared
            st.rerun()

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
                # This function handles clearing session state, session file, and backend logout
                logout()

# Social media links are now integrated into the Ant Design menu above

# Create main UI layout
title_col1, title_col2, title_col3 = st.columns([1, 2, 1])
with title_col2:
    st.title("Flame Audio AI Playground")

# Create a row with four columns for the action buttons
button_col1, button_col2, button_col3, button_col4 = st.columns(4)

# Add each button to its own column
with button_col1:
    save_button = st.button("Save Agent", type="primary", use_container_width=True, key='save_agent_btn')

#with button_col2:
    #view_code_button = st.button("View Code", type="secondary", use_container_width=True, key='view_code_btn')

#with button_col3:
    #chat_button = st.button("Chat", type="secondary", use_container_width=True, key='chat_btn')

with button_col4:
    delete_button = st.button("Delete Agent", type="secondary", use_container_width=True, key='delete_agent_btn')

# Main layout with three columns
col1, col2, col3 = st.columns([1, 1.5, 1])

# Right column - Parameters
with col3:
    parameters_container = st.container(border=True)
    with parameters_container:
        st.subheader("PARAMETERS")
        # Task selection - transcribe or translate
        task = st.selectbox(
            "Task",
            options=st.session_state.get("task_options", ["transcribe", "translate"]),
            index=st.session_state.get("task_options", ["transcribe", "translate"]).index(st.session_state.get("current_task", "transcribe")) if st.session_state.get("current_task", "transcribe") in st.session_state.get("task_options", ["transcribe", "translate"]) else 0,
            help="transcribe: Keep the original language, translate: Convert any language to English"
        )

        # Update the session state when task changes
        if task != st.session_state.get("current_task", "transcribe"):
            st.session_state.current_task = task

        # Advanced settings expander
        with st.expander("Advanced Settings"):
            chunk_length = st.number_input(
                "Chunk Length (seconds)",
                min_value=60,
                max_value=3600,
                value=st.session_state.get("chunk_length", 600),  # Default to 10 minutes for better processing
                help="Length of each audio chunk in seconds. Smaller chunks process faster but may have more boundary issues."
            )

            overlap = st.number_input(
                "Chunk Overlap (seconds)",
                min_value=0,
                max_value=60,
                value=st.session_state.get("overlap", 5),
                help="Overlap between chunks in seconds to ensure smooth transitions."
            )

            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get("temperature", 0.0),
                step=0.1,
                help="Controls randomness in output. Higher values make output more random."
            )

            # Update session state with new values
            st.session_state.chunk_length = chunk_length
            st.session_state.overlap = overlap
            st.session_state.temperature = float(temperature)  # Ensure temperature is stored as a float

    # Load available languages from backend
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_languages():
        try:
            response = requests.get(f"{BACKEND_URL}/languages")
            if response.status_code == 200:
                return response.json()["languages"]
            else:
                st.error(f"Error loading languages: {response.text}")
                return []
        except Exception as e:
            st.error(f"Error connecting to backend: {str(e)}")
            return []

    # Language selection
    selected_language = None
    if st.session_state.get("current_task", "transcribe") == "transcribe":  # Use session state variable
        languages = load_languages()
        language_names = [lang["name"] for lang in languages]
        display_language = st.selectbox(
            "Audio Language",
            options=["Auto-detect"] + language_names,
            help="Select the language spoken in your audio file to improve transcription accuracy. Auto-detect will let the model determine the language."
        )
        if display_language != "Auto-detect":
            selected_language = next(lang["code"] for lang in languages if lang["name"] == display_language)
    else:  # Translation
        selected_language = None
        st.info("The audio will be automatically translated to English text, regardless of the source language.")

    # Note about language support
    st.info("Note: The Models currently only support transcribing in the original language or translating to English. Multi-language translation is not yet supported.")

# Left column - System
with col1:
    system_container = st.container(border=True)
    with system_container:
        st.subheader("Agent Configuration")
        agent_name = st.text_input("Enter an agent name", value=st.session_state.get("agent_name", ""), key="agent_name_input")
        system_message = st.text_area("Enter a system message", value=st.session_state.get("system_message", ""), height=150, key="system_message_input")

        # Update session state when inputs change
        st.session_state.agent_name = agent_name
        st.session_state.system_message = system_message

# Middle column - Speech
with col2:
    speech_container = st.container(border=True)
    with speech_container:
        st.subheader("SPEECH")
        # Audio recorder component with extended recording duration
        audio_bytes = audio_recorder(
            key="recorder",
            pause_threshold=120.0,  # Set a longer pause threshold (2 minutes)
            sample_rate=44100  # Higher quality audio recording
        )

        # Initialize delete button state if not already set
        if 'delete_recording_clicked' not in st.session_state:
            st.session_state.delete_recording_clicked = False

        # Function to handle deletion
        def delete_recorded_audio():
            st.session_state.delete_recording_clicked = True

        if audio_bytes is not None:
            st.session_state.recorded_audio = audio_bytes
            st.success("Recording complete!")

            # Save the recorded audio to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_bytes)
                st.session_state.recorded_file = tmp_file.name

                # Create a download button for the recorded audio
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Download Recorded Audio",
                        data=audio_bytes,
                        file_name="recorded_audio.wav",
                        mime="audio/wav",
                        use_container_width=True
                    )

                with col2:
                    # Add delete button - using a callback function for better state management
                    st.button(
                        "Delete Recording",
                        type="secondary",
                        key="delete_recording_button",
                        on_click=delete_recorded_audio,
                        use_container_width=True
                    )

                # Add playback of recorded audio
                st.audio(audio_bytes, format="audio/wav")

        # Check if delete button was clicked and handle deletion
        if st.session_state.delete_recording_clicked:
            # Remove the temporary file
            try:
                if 'recorded_file' in st.session_state and os.path.exists(st.session_state.recorded_file):
                    os.remove(st.session_state.recorded_file)
                    st.info("Temporary file removed")
            except Exception as e:
                st.error(f"Error removing temporary file: {e}")

            # Clear session state
            if 'recorded_audio' in st.session_state:
                del st.session_state.recorded_audio
            if 'recorded_file' in st.session_state:
                del st.session_state.recorded_file

            # Reset the delete flag
            st.session_state.delete_recording_clicked = False

            st.success("Recording deleted!")
            st.rerun()  # Refresh the page
        # Add transcribe button for recorded audio if audio was recorded
        if 'recorded_audio' in st.session_state and st.session_state.recorded_audio is not None:
            # Generate button text based on the selected task
            button_text = "Translate Recording" if st.session_state.get("current_task", "transcribe") == "translate" else "Transcribe Recording"

            if st.button(button_text, type="primary", key="transcribe_recording", use_container_width=True):
                # Check if a model is selected
                if not st.session_state.get("selected_model"):
                    st.error("Please select a model from the dropdown menu in the sidebar.")
                else:
                    # Create form data
                    form_data = {
                        "model": st.session_state.get("selected_model"),
                        "response_format": "verbose_json",  # Always request verbose_json
                        "task": st.session_state.get("current_task", "transcribe"),  # Use session state variable
                        "chunk_length": st.session_state.get("chunk_length", 600),
                        "overlap": st.session_state.get("overlap", 5),
                        "temperature": st.session_state.get("temperature", 0.0)
                    }

                    # Add language if specified, but only if it's not None
                    if selected_language is not None:
                        form_data["language"] = selected_language

                    # Create multipart form data
                    audio_bytes = st.session_state.recorded_audio
                    files = {"file": ("recorded_audio.wav", audio_bytes, "audio/wav")}

                    # Show progress
                    with st.spinner(f"Initializing {st.session_state.get('current_task', 'transcribe')}..."):
                        # Create a progress bar
                        progress_bar = st.progress(0)

                        try:
                            # Make request with timeout - send each parameter individually, not as a JSON string
                            progress_bar.progress(10, "Uploading file...")

                            # Start making the request
                            operation_type = "translation" if st.session_state.get("current_task", "transcribe") == "translate" else "transcription"
                            progress_bar.progress(30, f"Processing audio for {operation_type}...")

                            # Determine which endpoint to use based on the task
                            endpoint = "translate" if st.session_state.get("current_task", "transcribe") == "translate" else "transcribe"

                            response = requests.post(
                                f"{BACKEND_URL}/audio/{endpoint}",
                                files=files,
                                data=form_data,  # Send as form fields, not as JSON
                                headers={
                                    "X-API-KEY": groq_api_key_input,
                                    "Authorization": f"Bearer {st.session_state.get('_auth_token_', '')}"
                                },
                                timeout=600  # 10 minute timeout
                            )

                            progress_bar.progress(80, "Finalizing transcription...")

                            # Check response
                            if response.status_code == 200:
                                st.session_state.transcription_result = response.json() # Store the result
                                st.session_state["transcription_done"] = True
                                operation_type = "translation" if st.session_state.get("current_task", "transcribe") == "translate" else "transcription"
                                progress_bar.progress(100, f"{operation_type.capitalize()} complete!")

                                # Store filename instead of automatically saving to database
                                st.session_state.last_filename = "recorded_audio.wav"
                                st.session_state.selected_model = st.session_state.get("selected_model")
                                st.session_state.selected_language = selected_language
                                st.session_state.last_file_content = audio_bytes

                                # Show save documents message
                                st.info("Transcription complete! Click 'Save Documents' below to generate embeddings and save to the database.")
                            else:
                                progress_bar.empty()
                                st.error(f"Error: {response.status_code} - {response.text}")
                                st.session_state.transcription_result = None # Clear any previous result

                                # Show more detailed error information
                                try:
                                    error_json = response.json()
                                    st.json(error_json)
                                except:
                                    st.code(response.text)
                        except requests.exceptions.Timeout:
                            st.error("The transcription request timed out after 10 minutes. The audio file might be too large or the server might be busy.")
                            st.info("Try using a smaller audio file or increasing the chunk size in the advanced settings.")
                            st.session_state.transcription_result = None

                        except requests.exceptions.ConnectionError:
                            st.error("Could not connect to the backend server. Make sure the FastAPI server is running.")
                            st.code("Run this command in a terminal to start the server: python -m uvicorn main:app --reload")
                            st.session_state.transcription_result = None

                        except Exception as e:
                            st.error(f"An unexpected error occurred: {str(e)}")
                            st.session_state.transcription_result = None

# File upload section
        st.write("")
        uploaded_file = st.file_uploader(
            "Select File",
            type=["flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"],
            help="40MB max. flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm supported"
        )
        # Process audio when file is uploaded
        if uploaded_file is not None:
            # Check authentication status if AUTH_ENABLED is true
            if AUTH_ENABLED and not st.session_state.get("authenticated", False):
                st.error("Please sign in to use this feature")
            elif not groq_api_key_input:
                # Check for API key
                st.warning("Please enter your Groq API key to continue")
            else:
                # Generate button text based on the selected task
                button_text = "Translate" if st.session_state.get("current_task", "transcribe") == "translate" else "Transcribe"
                if st.button(button_text, type="primary", use_container_width=True):
                    # Check if a model is selected
                    if not st.session_state.get("selected_model"):
                        st.error("Please select a model from the dropdown menu in the sidebar.")
                    else:
                        # Create form data
                        form_data = {
                            "model": st.session_state.get("selected_model"),
                            "response_format": "verbose_json",  # Always request verbose_json
                            "task": st.session_state.get("current_task", "transcribe"),  # Use session state variable
                            "chunk_length": st.session_state.get("chunk_length", 600),
                            "overlap": st.session_state.get("overlap", 5),
                            "temperature": st.session_state.get("temperature", 0.0)
                        }

                        # Add language if specified, but only if it's not None
                        if selected_language is not None:
                            form_data["language"] = selected_language

                        # Create multipart form data
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}

                        # Show progress
                        with st.spinner(f"Initializing {st.session_state.get('current_task', 'transcribe')}..."):
                            # Create a progress bar
                            progress_bar = st.progress(0)

                            try:
                                # Make request with timeout - send each parameter individually, not as a JSON string
                                progress_bar.progress(10, "Uploading file...")

                                # Start making the request
                                operation_type = "translation" if st.session_state.get("current_task", "transcribe") == "translate" else "transcription"
                                progress_bar.progress(30, f"Processing audio for {operation_type}...")

                                # Determine which endpoint to use based on the task
                                endpoint = "translate" if st.session_state.get("current_task", "transcribe") == "translate" else "transcribe"

                                response = requests.post(
                                    f"{BACKEND_URL}/audio/{endpoint}",
                                    files=files,
                                    data=form_data,  # Send as form fields, not as JSON
                                    headers={
                                        "X-API-KEY": groq_api_key_input,
                                        "Authorization": f"Bearer {st.session_state.get('_auth_token_', '')}"
                                    },
                                    timeout=600  # 10 minute timeout
                                )

                                progress_bar.progress(80, "Finalizing transcription...")

                                # Check response
                                if response.status_code == 200:
                                    st.session_state.transcription_result = response.json() # Store the result
                                    st.session_state["transcription_done"] = True
                                    operation_type = "translation" if st.session_state.get("current_task", "transcribe") == "translate" else "transcription"
                                    progress_bar.progress(100, f"{operation_type.capitalize()} complete!")

                                    # Store filename instead of automatically saving to database
                                    st.session_state.last_filename = uploaded_file.name
                                    st.session_state.selected_model = st.session_state.get("selected_model")
                                    st.session_state.selected_language = selected_language
                                    st.session_state.last_file_content = uploaded_file.getvalue()

                                    # Show save documents message
                                    st.info("Transcription complete! Click 'Save Documents' below to generate embeddings and save to the database.")
                                else:
                                    progress_bar.empty()
                                    st.error(f"Error: {response.status_code} - {response.text}")
                                    st.session_state.transcription_result = None # Clear any previous result

                                    # Show more detailed error information
                                    try:
                                        error_json = response.json()
                                        st.json(error_json)
                                    except:
                                        st.code(response.text)
                            except requests.exceptions.Timeout:
                                st.error("The transcription request timed out after 10 minutes. The audio file might be too large or the server might be busy.")
                                st.info("Try using a smaller audio file or increasing the chunk size in the advanced settings.")
                                st.session_state.transcription_result = None

                            except requests.exceptions.ConnectionError:
                                st.error("Could not connect to the backend server. Make sure the FastAPI server is running.")
                                st.code("Run this command in a terminal to start the server: python -m uvicorn main:app --reload")
                                st.session_state.transcription_result = None

                            except Exception as e:
                                st.error(f"An unexpected error occurred: {str(e)}")
                                st.session_state.transcription_result = None

# Display transcription results if available - only show this if authenticated or auth is disabled
if st.session_state.get("authenticated", False) or not AUTH_ENABLED:
    results_container = st.container(border=True)
    with results_container:
        # Determine operation type for labels
        operation_type = "Translation" if st.session_state.get("current_task", "transcribe") == "translate" else "Transcription"

        # Add buttons at the top in a row
        if "transcription_result" in st.session_state and st.session_state.get("transcription_result") is not None:
            # Create a single column for the Clear button (removed Save Document button)
            clear_col = st.columns(1)[0]

            # Clear results button
            with clear_col:
                if st.button(f"Clear Results", type="secondary", use_container_width=True):
                    st.session_state["transcription_result"] = None
                    st.session_state["transcription_done"] = False
                    # Also clear any file content to prevent reuse
                    if "last_file_content" in st.session_state:
                        st.session_state.pop("last_file_content")
                    if "last_filename" in st.session_state:
                        st.session_state.pop("last_filename")
                    st.experimental_rerun()

            # Define helper function for generating filenames
            def get_base_filename():
                """Get a base filename from the original file, removing extension and sanitizing"""
                # Get original filename from session state, or use a default
                original_filename = st.session_state.get("last_filename", "transcription")

                # Remove file extension if present
                base_name = os.path.splitext(original_filename)[0]

                # Replace invalid filename characters with underscores
                base_name = re.sub(r'[\\/*?:"<>|]', "_", base_name)

                return base_name

            # Get base filename for downloads
            base_filename = get_base_filename()

            # Create the tabs
            text_tab, segments_tab, json_tab = st.tabs(["Text", "Segments", "JSON"])

            with text_tab:
                # Display plain text in the text tab
                st.markdown(f"### {operation_type} Text")
                full_text = ""
                for segment in st.session_state["transcription_result"].get("segments", []):
                    if segment.get("text"):
                        full_text += segment.get("text") + " "

                st.text_area("Full Text", full_text, height=400)

                # Add a download button for the text
                download_text_btn = st.download_button(
                    label="Download Text",
                    data=full_text,
                    file_name=f"{base_filename}.txt",
                    mime="text/plain"
                )

            with segments_tab:
                # Display segments with time information
                st.markdown(f"### {operation_type} Segments with Timestamps")

                segments = st.session_state["transcription_result"].get("segments", [])
                if segments:
                    # Create a DataFrame for the segments
                    segments_data = []
                    for i, segment in enumerate(segments):
                        start_time = segment.get("start", 0)
                        end_time = segment.get("end", 0)
                        segments_data.append({
                            "Segment": i + 1,
                            "Start Time": f"{int(start_time // 60)}:{int(start_time % 60):02d}",
                            "End Time": f"{int(end_time // 60)}:{int(end_time % 60):02d}",
                            "Duration": f"{int((end_time - start_time) // 60)}:{int((end_time - start_time) % 60):02d}",
                            "Text": segment.get("text", "")
                        })

                    # Convert to DataFrame and display
                    df = pd.DataFrame(segments_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)

                    # Create functions for generating different export formats

                    def generate_srt(segments):
                        srt_content = ""
                        for i, segment in enumerate(segments):
                            start_time = segment.get("start", 0)
                            end_time = segment.get("end", 0)

                            # Format timestamps as HH:MM:SS,mmm
                            start_formatted = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{int(start_time%60):02d},{int((start_time%1)*1000):03d}"
                            end_formatted = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{int(end_time%60):02d},{int((end_time%1)*1000):03d}"

                            srt_content += f"{i+1}\n{start_formatted} --> {end_formatted}\n{segment.get('text', '')}\n\n"

                        return srt_content

                    def generate_vtt(segments):
                        vtt_content = "WEBVTT\n\n"
                        for i, segment in enumerate(segments):
                            start_time = segment.get("start", 0)
                            end_time = segment.get("end", 0)

                            # Format timestamps as HH:MM:SS.mmm
                            start_formatted = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{int(start_time%60):02d}.{int((start_time%1)*1000):03d}"
                            end_formatted = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{int(end_time%60):02d}.{int((end_time%1)*1000):03d}"

                            vtt_content += f"{start_formatted} --> {end_formatted}\n{segment.get('text', '')}\n\n"

                        return vtt_content

                    def generate_premiere_markers_csv(segments):
                        # Create CSV content with format compatible with Premiere Pro markers
                        csv_content = "Name,Description,In,Out,Duration,Marker Type\n"

                        for i, segment in enumerate(segments):
                            start_time = segment.get("start", 0)
                            end_time = segment.get("end", 0)
                            duration = end_time - start_time

                            # Format times as HH:MM:SS:FF (assuming 30fps)
                            start_formatted = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{int(start_time%60):02d}:{int((start_time%1)*30):02d}"
                            end_formatted = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{int(end_time%60):02d}:{int((start_time%1)*30):02d}"
                            duration_formatted = f"{int(duration//3600):02d}:{int((duration%3600)//60):02d}:{int(duration%60):02d}:{int((duration%1)*30):02d}"

                            text = segment.get("text", "").replace('"', '""')  # Escape quotes for CSV

                            csv_content += f'"Segment {i+1}","{text}","{start_formatted}","{end_formatted}","{duration_formatted}","Comment"\n'

                        return csv_content

                    def generate_anki_csv(segments):
                        # Create CSV content with format: audio_segment_filename,transcript
                        csv_content = "audio,text,start_time,end_time\n"

                        for i, segment in enumerate(segments):
                            segment_filename = f"segment_{i+1:03d}.mp3"
                            text = segment.get("text", "").replace('"', '""')  # Escape quotes for CSV
                            start_time = segment.get("start", 0)
                            end_time = segment.get("end", 0)

                            csv_content += f'"{segment_filename}","{text}","{start_time}","{end_time}"\n'

                        return csv_content

                    # Create download section with multiple options
                    st.markdown("### Download Options")

                    # Get base filename for downloads
                    base_filename = get_base_filename()

                    # Create columns for download buttons
                    col1, col2 = st.columns(2)

                    with col1:
                        # Standard CSV download
                        download_csv_btn = st.download_button(
                            label="Download Segments as CSV",
                            data=df.to_csv(index=False),
                            file_name=f"{base_filename}_segments.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                        # SRT subtitle format
                        download_srt_btn = st.download_button(
                            label="Download as SRT Subtitles",
                            data=generate_srt(st.session_state["transcription_result"].get("segments", [])),
                            file_name=f"{base_filename}.srt",
                            mime="text/plain",
                            use_container_width=True
                        )

                    with col2:
                        # WebVTT subtitle format
                        download_vtt_btn = st.download_button(
                            label="Download as WebVTT Subtitles",
                            data=generate_vtt(st.session_state["transcription_result"].get("segments", [])),
                            file_name=f"{base_filename}.vtt",
                            mime="text/vtt",
                            use_container_width=True
                        )

                        # Premiere Pro markers
                        download_premiere_btn = st.download_button(
                            label="Download as Premiere Markers",
                            data=generate_premiere_markers_csv(st.session_state["transcription_result"].get("segments", [])),
                            file_name=f"{base_filename}_premiere_markers.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    # Advanced options in an expander
                    with st.expander("Advanced Download Options"):
                        col3, col4 = st.columns(2)

                        with col3:
                            # Anki flashcards format
                            download_anki_btn = st.download_button(
                                label="Download as Anki Flashcards CSV",
                                data=generate_anki_csv(st.session_state["transcription_result"].get("segments", [])),
                                file_name=f"{base_filename}_anki_flashcards.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                        with col4:
                            # Audio segments extraction button
                            if st.button("Extract Audio Segments", key="extract_segments_btn", use_container_width=True):
                                with st.spinner("Preparing audio segments..."):
                                    st.info("This feature requires ffmpeg to be installed on your system.")
                                    st.info(f"For a complete implementation, audio segments from '{base_filename}' would be extracted and provided as a zip file.")
                                    st.info("This would include each segment as a separate audio file along with its transcript.")
                else:
                    st.info("No segments available.")

            with json_tab:
                # Display the raw JSON for developers
                st.markdown(f"### Raw JSON Output")
                st.json(st.session_state["transcription_result"])

                # Add a download button for the JSON
                download_json_btn = st.download_button(
                    label="Download JSON",
                    data=json.dumps(st.session_state["transcription_result"], indent=2),
                    file_name=f"{base_filename}.json",
                    mime="application/json"
                )

            # Save functionality is implemented in a separate container at the end of this file
            # (see the container with border=True at the bottom of this script)

# Check if the user is logged in / authenticated
if AUTH_ENABLED and st.session_state.get("authenticated", False) and "user" in st.session_state:
    # Set API key from user profile if available
    if "api_key" in st.session_state["user"] and st.session_state["user"]["api_key"]:
        groq_api_key_input = st.session_state["user"]["api_key"]

# Initialize agents list in session state
if "agents" not in st.session_state:
    st.session_state.agents = load_agents()

# Sign out is now handled by the logout() function from auth_forms.py

# Create a separate container for the save functionality with a visual separator
if "transcription_result" in st.session_state and st.session_state.get("transcription_result") is not None:
    # Add vertical space for clear visual separation
    st.write("")
    st.write("")
    st.markdown("---")

    # Create a simpler container
    save_nav_container = st.container(border=True)
    with save_nav_container:
        st.markdown("## Save Your Transcription")

        # Create columns for the layout
        col1, col2 = st.columns([3, 1])

        with col1:
            st.info("Your transcription is ready to be saved. Click the button to proceed to the Save Document page.")

            # Set the flag automatically since we already have transcription results
            st.session_state["save_requested"] = True

        with col2:
            st.write("")

            # Create a direct approach - just prepare the data and use the standard navigation
            if st.button("Save Document", key="save_doc_btn", type="primary", use_container_width=True):
                # Just make sure the transcription is in session state and navigate
                if "transcription_result" in st.session_state and st.session_state["transcription_result"] is not None:
                    # Set the flag to indicate a save is requested
                    st.session_state["save_requested"] = True

                    # Store transcription data in both variables for compatibility
                    st.session_state["_temp_transcription"] = st.session_state["transcription_result"]
                    st.session_state["_persistent_transcription"] = st.session_state["transcription_result"]

                    # Ensure all required session variables are preserved
                    # These are the variables needed by the Save Document page as listed by the user
                    required_vars = [
                        "current_agent_id",
                        "temperature",
                        "agent_name",
                        "overlap",
                        "task_options",
                        "init_completed",
                        "current_task",
                        "authenticated",
                        "transcription_result",
                        "system_message",
                        "auth_state",
                        "selected_model",
                        "user",
                        "_auth_state_",
                        "_auth_token_",
                        "chunk_length"
                    ]

                    # Log the session state before navigation for debugging
                    print("Session state before navigation:", [key for key in st.session_state.keys()])

                    # Navigate to the Save Document page
                    st.switch_page("pages/02_Save_Document.py")
                else:
                    st.error("No transcription data available to save.")

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
