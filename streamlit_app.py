import streamlit as st

# Configure page settings with title, favicon, and description
st.set_page_config(
    page_title="Flame Audio",
    page_icon="logos/flame logo.jpg",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': """# Flame Audio
A powerful speech transcription and translation application built by FlameheadLabs. Convert spoken language to text through recorded audio or uploaded files with support for multiple languages and advanced processing options."""
    }
)

import requests
from dotenv import load_dotenv
import os
import json
import pandas as pd  # Import pandas
import streamlit_antd_components as sac
from audio_recorder_streamlit import audio_recorder
import tempfile
import time
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

# Initialize authentication session
init_auth_session()

# Check for authentication callback
handle_auth_callback()

# Initialize session state variables
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

# Initialize selected model in session state
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Sidebar - Company info and auth controls
with st.sidebar:
    # Title
    st.image("logos/flame logo.jpg", width=80)
    st.markdown("### Flame Audio")
    
    # Authentication UI
    if not st.session_state.authenticated and AUTH_ENABLED:
        auth_forms()
    elif AUTH_ENABLED:
        st.success(f"Logged in as: {st.session_state.user.get('email', '')}")
        if st.button("Logout", use_container_width=True, type="primary"):
            logout()
    
    st.markdown("---")

# Only show API key input and model selection if authenticated or auth is disabled
if st.session_state.authenticated or not AUTH_ENABLED:
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
                if model["id"] == st.session_state.selected_model:
                    selected_model_info = model
                    break

            # Get the list of model options
            model_options = [model["id"] for model in models]
            
            # Find the index of the currently selected model in the options
            default_index = None
            if st.session_state.selected_model in model_options:
                default_index = model_options.index(st.session_state.selected_model)

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

# Create three columns for the layout
col1, col2, col3 = st.columns([1, 1.5, 1])

# Right column - Parameters
with col3:
    parameters_container = st.container(border=True)
    with parameters_container:
        st.subheader("PARAMETERS")
        # Task selection - transcribe or translate
        task = st.selectbox(
            "Task",
            options=st.session_state.task_options,
            index=st.session_state.task_options.index(st.session_state.current_task) if st.session_state.current_task in st.session_state.task_options else 0,
            help="transcribe: Keep the original language, translate: Convert any language to English"
        )
        
        # Update the session state when task changes
        if task != st.session_state.current_task:
            st.session_state.current_task = task
        
        # Advanced settings expander
        with st.expander("Advanced Settings"):
            chunk_length = st.number_input(
                "Chunk Length (seconds)", 
                min_value=60, 
                max_value=3600,
                value=st.session_state.chunk_length,  # Default to 10 minutes for better processing
                help="Length of each audio chunk in seconds. Smaller chunks process faster but may have more boundary issues."
            )
            
            overlap = st.number_input(
                "Chunk Overlap (seconds)",
                min_value=0,
                max_value=60,
                value=st.session_state.overlap,
                help="Overlap between chunks in seconds to ensure smooth transitions."
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.temperature,
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
    if st.session_state.current_task == "transcribe":  # Use session state variable
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
        st.subheader("SYSTEM")
        system_message = st.text_area("Enter a system message", height=150)

# Middle column - Speech
with col2:
    speech_container = st.container(border=True)
    with speech_container:
        st.subheader("SPEECH")
        # Recording interface
        st.subheader("Click to record")
        
        # Use the audio recorder with explicit parameters to avoid loading issues
        audio_bytes = audio_recorder(
            energy_threshold=0.01,  # Lower threshold for easier recording
            pause_threshold=2.0,    # 2 seconds of silence to stop recording
            sample_rate=44100,      # Standard sample rate
            text="Click to record",  # Explicit button text
            recording_color="#e8b62c",
            neutral_color="#6aa36f"
        )
        
        if audio_bytes:
            st.session_state.recorded_audio = audio_bytes
            st.success("✅ Recording complete!")
            
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
                        "🗑️ Delete Recording", 
                        type="secondary", 
                        key="delete_recording_button",
                        on_click=lambda: st.session_state.update({"delete_recording_clicked": True}),
                        use_container_width=True
                    )
                
                # Add playback of recorded audio
                st.audio(audio_bytes, format="audio/wav")
                
        # Check if delete button was clicked and handle deletion
        if st.session_state.get("delete_recording_clicked", False):
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
            button_text = "Translate Recording" if st.session_state.current_task == "translate" else "Transcribe Recording"
            
            if st.button(button_text, type="primary", key="transcribe_recording", use_container_width=True):
                # Check if a model is selected
                if not selected_model:
                    st.error("Please select a model from the dropdown menu in the sidebar.")
                else:
                    # Create form data
                    form_data = {
                        "model": selected_model,
                        "response_format": "verbose_json",  # Always request verbose_json
                        "task": st.session_state.current_task,  # Use session state variable
                        "chunk_length": st.session_state.chunk_length,
                        "overlap": st.session_state.overlap,
                        "temperature": st.session_state.temperature
                    }
                    
                    # Add language if specified, but only if it's not None
                    if selected_language is not None:
                        form_data["language"] = selected_language

                    # Create multipart form data
                    audio_bytes = st.session_state.recorded_audio
                    files = {"file": ("recorded_audio.wav", audio_bytes, "audio/wav")}
                    
                    # Show progress
                    with st.spinner(f"Initializing {st.session_state.current_task}..."):
                        # Create a progress bar
                        progress_bar = st.progress(0)
                        
                        try:
                            # Make request with timeout - send each parameter individually, not as a JSON string
                            progress_bar.progress(10, "Uploading file...")
                            time.sleep(0.5) # Small delay for visual feedback
                            
                            # Start making the request
                            operation_type = "translation" if st.session_state.current_task == "translate" else "transcription"
                            progress_bar.progress(30, f"Processing audio for {operation_type}...")
                            
                            # Determine which endpoint to use based on the task
                            endpoint = "translate" if st.session_state.current_task == "translate" else "transcribe"
                            
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
                            time.sleep(0.5) # Small delay for visual feedback
                            
                            # Check response
                            if response.status_code == 200:
                                st.session_state.transcription_result = response.json() # Store the result
                                operation_type = "translation" if st.session_state.current_task == "translate" else "transcription"
                                progress_bar.progress(100, f"{operation_type.capitalize()} complete!")
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
            if AUTH_ENABLED and not st.session_state.authenticated:
                st.error("Please sign in to use this feature")
            elif not groq_api_key_input:
                # Check for API key
                st.warning("Please enter your Groq API key to continue")
            else:
                # Generate button text based on the selected task
                button_text = "Translate" if st.session_state.current_task == "translate" else "Transcribe"
                if st.button(button_text, type="primary", use_container_width=True):
                    # Check if a model is selected
                    if not selected_model:
                        st.error("Please select a model from the dropdown menu in the sidebar.")
                    else:
                        # Create form data
                        form_data = {
                            "model": selected_model,
                            "response_format": "verbose_json",  # Always request verbose_json
                            "task": st.session_state.current_task,  # Use session state variable
                            "chunk_length": st.session_state.chunk_length,
                            "overlap": st.session_state.overlap,
                            "temperature": st.session_state.temperature
                        }
                        
                        # Add language if specified, but only if it's not None
                        if selected_language is not None:
                            form_data["language"] = selected_language

                        # Create multipart form data
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        
                        # Show progress
                        with st.spinner(f"Initializing {st.session_state.current_task}..."):
                            # Create a progress bar
                            progress_bar = st.progress(0)
                            
                            try:
                                # Make request with timeout - send each parameter individually, not as a JSON string
                                progress_bar.progress(10, "Uploading file...")
                                time.sleep(0.5) # Small delay for visual feedback
                                
                                # Start making the request
                                operation_type = "translation" if st.session_state.current_task == "translate" else "transcription"
                                progress_bar.progress(30, f"Processing audio for {operation_type}...")
                                
                                # Determine which endpoint to use based on the task
                                endpoint = "translate" if st.session_state.current_task == "translate" else "transcribe"
                                
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
                                time.sleep(0.5) # Small delay for visual feedback
                                
                                # Check response
                                if response.status_code == 200:
                                    st.session_state.transcription_result = response.json() # Store the result
                                    operation_type = "translation" if st.session_state.current_task == "translate" else "transcription"
                                    progress_bar.progress(100, f"{operation_type.capitalize()} complete!")
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
                                st.error("The transcription request timed out after 5 minutes. The audio file might be too large or the server might be busy.")
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
if st.session_state.authenticated or not AUTH_ENABLED:
    results_container = st.container(border=True)
    with results_container:
        # Determine operation type for labels
        operation_type = "Translation" if "current_task" in st.session_state and st.session_state.current_task == "translate" else "Transcription"
        
        # --- Radio buttons for view selection (outside the upload/transcribe logic) ---
        view_option = st.radio(
            f"{operation_type} Results:",
            ["Text", "Segments", "JSON"],
            horizontal=True
        )    
        # Add clear results button at the top
        if st.session_state.transcription_result is not None:
            if st.button(f"Clear {operation_type} Results", type="secondary", use_container_width=True):
                st.session_state.transcription_result = None
                st.rerun()
                
            # Show appropriate success message based on task
            operation_type = "Translation" if st.session_state.current_task == "translate" else "Transcription"
            st.success(f"{operation_type} completed successfully!")
            
            # Display results based on selected view option
            if view_option == "Text":
                # Show either Translation or Transcription result heading
                st.markdown(f"### {operation_type} Result")
                st.markdown(st.session_state.transcription_result.get("text", ""))
                
                # Add download button for text
                text_result = st.session_state.transcription_result.get("text", "")
                st.download_button(
                    label="Download Text",
                    data=text_result,
                    file_name="transcription.txt",
                    mime="text/plain"
                )
                
            elif view_option == "Segments":
                st.markdown("### Segments")
                segments = st.session_state.transcription_result.get("segments", [])
                
                if segments:
                    # Create a DataFrame for better display
                    segments_data = []
                    for segment in segments:
                        segments_data.append({
                            "Start": f"{segment.get('start', 0):.2f}s",
                            "End": f"{segment.get('end', 0):.2f}s",
                            "Text": segment.get("text", "")
                        })
                    
                    # Display as a DataFrame
                    df = pd.DataFrame(segments_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # Add download button for segments as CSV
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download Segments (CSV)",
                        data=csv,
                        file_name="segments.csv",
                        mime="text/csv"
                    )
                else:
                    st.info("No segments available in the transcription result.")
            
            elif view_option == "JSON":
                st.markdown("### Raw JSON Response")
                st.json(st.session_state.transcription_result)
                
                # Add download button for full JSON
                json_str = json.dumps(st.session_state.transcription_result, indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="transcription.json",
                    mime="application/json"
                )
        
# Display user info if authenticated
if AUTH_ENABLED and st.session_state.authenticated and st.session_state.user:
    # Add a user info section at the bottom of sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"Logged in as: **{st.session_state.user.get('email', 'User')}**")

# Add significant spacing before the Links expander
st.sidebar.markdown("<br>" * 5, unsafe_allow_html=True)

# Add social media links to sidebar in an expander
with st.sidebar.expander("Links", expanded=True):
    st.write("Connect with FlameheadLabs:")
    
    sac.buttons([
        sac.ButtonsItem(label='About FlameheadLabs', icon='house', href='http://flameheadlabs.tech/'),
        sac.ButtonsItem(label='GitHub', icon='github', href='https://github.com/Flamehead-Labs-Ug/flame-audio'),
        sac.ButtonsItem(label='Connect with us on Discord', icon='discord', href='https://discord.gg/fFjXkk5m'),
        sac.ButtonsItem(label='Follow on X', icon='twitter', href='https://x.com/flameheadlabsug'),
        sac.ButtonsItem(label='LinkedIn', icon='linkedin', href='https://www.linkedin.com/in/flamehead-labs-919910285'),
    ], align='center', size='sm')

# Add a professional footer
footer = """
<footer style="margin-top: 5rem; padding: 2.5rem 0; border-top: 1px solid rgba(0,0,0,0.05); width: 100%;">    
    <div style="max-width: 1200px; margin: 0 auto; padding: 0 1.5rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">        
        <div style="display: flex; align-items: center;">            
            <div style="font-size: 0.9rem; color: #6B7280; margin-right: 1rem;">
                &copy; 2025 FlameheadLabs
            </div>
            <div style="height: 16px; width: 1px; background-color: #E5E7EB; margin: 0 0.5rem;"></div>
            <a href="http://flameheadlabs.tech/privacy" style="font-size: 0.9rem; color: #6B7280; text-decoration: none; margin: 0 0.5rem;">Privacy Policy</a>
            <a href="http://flameheadlabs.tech/terms" style="font-size: 0.9rem; color: #6B7280; text-decoration: none; margin: 0 0.5rem;">Terms of Service</a>
        </div>
        <div style="background: linear-gradient(90deg, #FF4B4B 0%, #FF6B6B 100%); padding: 0.5rem 1rem; border-radius: 0.25rem; margin-top: 1rem; box-shadow: 0 4px 6px -1px rgba(255, 75, 75, 0.1), 0 2px 4px -1px rgba(255, 75, 75, 0.06);">
            <span style="font-weight: 600; color: white; font-size: 0.9rem;">Powered by FlameheadLabs AI</span>
        </div>
    </div>
</footer>
"""

st.markdown(footer, unsafe_allow_html=True)
