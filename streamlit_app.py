import streamlit as st

st.set_page_config(layout="wide")
import streamlit as st
import requests
from dotenv import load_dotenv
import os
import json
import pandas as pd  # Import pandas
import streamlit_antd_components as sac
from audio_recorder_streamlit import audio_recorder
import tempfile
import time
from authentication.auth_forms import auth_forms, handle_auth_callback, init_auth_session, load_session_data

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# FastAPI backend URL
BACKEND_URL = "http://localhost:8000"

# Configure page layout

# Initialize authentication session
init_auth_session()

# Check for authentication callback
handle_auth_callback()

# Auth Debug expander - Hidden per user request
# Uncomment the following block to show the Auth Debug panel
# with st.sidebar.expander("Auth Debug Info", expanded=False):
#     st.write(f"Authenticated: {st.session_state.get('authenticated', False)}")
#     if st.session_state.get('authenticated', False):
#         st.write(f"User: {st.session_state.get('user')}")
#     
#     # Show if session file exists
#     session_token = load_session_data()
#     st.write(f"Session file exists: {session_token is not None}")
#     
#     if st.button("Force Reload"):
#         st.rerun()

# Initialize transcription_result in session state
if 'transcription_result' not in st.session_state:
    st.session_state.transcription_result = None

# Authentication is now optional - users can access the app without logging in first
# The sign-in button is available in the sidebar for users who want to authenticate

# Sidebar configuration
# Add logo at the top of the sidebar
st.sidebar.image("logos/flame logo.jpg", width=250)

# Title after the logo
st.sidebar.title("Flame Speech To Text")

# Only show API key input and model selection if authenticated
if st.session_state.authenticated:
    # Groq API Key input
    groq_api_key_input = st.sidebar.text_input(
        "Groq API Key",
        value=groq_api_key if groq_api_key else "",
        type="password"
    )

    # Load available models
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def load_models():
        try:
            response = requests.get(f"{BACKEND_URL}/models", headers={"X-API-KEY": groq_api_key_input})
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
            models = load_models()
            if not models:
                st.warning("No audio models available. Please check your API key and connection.")

    # Initialize session state for model if not already set
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = None  # No default model
    
    if 'transcription_result' not in st.session_state:
        st.session_state.transcription_result = None  # Initialize result storage

    # Model selection
    if models:
            # Get the model info for the current selection
            selected_model_info = None
            for model in models:
                if model["id"] == st.session_state.selected_model:
                    selected_model_info = model
                    break

            # Model selection dropdown
            selected_model = st.sidebar.selectbox(
                "Model",
                options=[model["id"] for model in models],
                index=None,  # No default selection
                placeholder="Select a model",
                key="model_selector",
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
        
        # File upload section
        st.write("")
        uploaded_file = st.file_uploader(
            "Select File",
            type=["flac", "mp3", "mp4", "mpeg", "mpga", "m4a", "ogg", "wav", "webm"],
            help="40MB max. flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm supported"
        )

# Right column - Parameters
with col3:
    parameters_container = st.container(border=True)
    with parameters_container:
        st.subheader("PARAMETERS")
        
        # Task selection - transcribe or translate
        task = st.selectbox(
            "Task",
            options=["transcribe", "translate"],
            help="transcribe: Keep the original language, translate: Convert any language to English"
        )

        # Advanced settings expander
        with st.expander("Advanced Settings"):
            col1, col2 = st.columns(2)
            with col1:
                chunk_length = st.number_input(
                    "Chunk Length (seconds)", 
                    min_value=60, 
                    max_value=3600,
                    value=600,  # Default to 10 minutes for better processing
                    help="Length of each audio chunk in seconds. Smaller chunks process faster but may have more boundary issues."
                )
                
            with col2:
                overlap = st.number_input(
                    "Chunk Overlap (seconds)",
                    min_value=0,
                    max_value=60,
                    value=10,
                    help="Overlap between chunks in seconds to ensure smooth transitions."
                )
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                help="Controls randomness in output. Higher values make output more random."
            )


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
    if task == "transcribe":
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
    st.info("Note: The Groq API currently only supports transcribing in the original language or translating to English. Multi-language translation is not yet supported.")

# --- Radio buttons for view selection (outside the upload/transcribe logic) ---
view_option = st.radio(
    "Results:",
    ["Text", "Segments", "JSON"],
    horizontal=True
)

# Process audio when file is uploaded
if uploaded_file is not None:
    if not groq_api_key_input:
        st.warning("Please enter your Groq API key to continue")
    if st.button("Transcribe", type="primary", use_container_width=True):
        # Create form data
        form_data = {
            "model": selected_model,
            "response_format": "verbose_json",  # Always request verbose_json
            "task": task,
            "chunk_length": chunk_length,
            "overlap": overlap,
            "temperature": temperature
        }
        
        # Add language if specified, but only if it's not None
        if selected_language is not None:
            form_data["language"] = selected_language

        # Create multipart form data
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        
        # Show progress
        with st.spinner("Initializing transcription..."):
            # Create a progress bar
            progress_bar = st.progress(0)
            
            try:
                # Make request with timeout - send each parameter individually, not as a JSON string
                progress_bar.progress(10, "Uploading file...")
                time.sleep(0.5) # Small delay for visual feedback
                
                # Start making the request
                progress_bar.progress(30, "Processing audio...")
                
                response = requests.post(
                    f"{BACKEND_URL}/audio/transcribe",
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
                    progress_bar.progress(100, "Transcription complete!")
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

# Add transcribe button for recorded audio if audio was recorded
if 'recorded_audio' in st.session_state and st.session_state.recorded_audio is not None:
    if st.button("Transcribe Recording", type="primary", key="transcribe_recording", use_container_width=True):
        # Create form data
        form_data = {
            "model": selected_model,
            "response_format": "verbose_json",  # Always request verbose_json
            "task": task,
            "chunk_length": chunk_length,
            "overlap": overlap,
            "temperature": temperature
        }
        
        # Add language if specified, but only if it's not None
        if selected_language is not None:
            form_data["language"] = selected_language

        # Create multipart form data
        audio_bytes = st.session_state.recorded_audio
        files = {"file": ("recorded_audio.wav", audio_bytes, "audio/wav")}
        
        # Show progress
        with st.spinner("Initializing transcription..."):
            # Create a progress bar
            progress_bar = st.progress(0)
            
            try:
                # Make request with timeout - send each parameter individually, not as a JSON string
                progress_bar.progress(10, "Uploading file...")
                time.sleep(0.5) # Small delay for visual feedback
                
                # Start making the request
                progress_bar.progress(30, "Processing audio...")
                
                response = requests.post(
                    f"{BACKEND_URL}/audio/transcribe",
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
                    progress_bar.progress(100, "Transcription complete!")
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

# Display transcription results if available
results_container = st.container(border=True)
with results_container:
    # Add clear results button at the top
    if st.session_state.transcription_result is not None:
        if st.button("Clear Results", type="secondary", use_container_width=True):
            st.session_state.transcription_result = None
            st.rerun()
            
        st.success("Transcription completed successfully!")
        
        # Display results based on selected view option
        if view_option == "Text":
            st.markdown("### Transcription Result")
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
        
# Add sign in/out buttons in the sidebar
st.sidebar.markdown("---")
if not st.session_state.authenticated:
    # Use expander for login form instead of popover
    with st.sidebar.expander("Sign In", expanded=False):
        # Display authentication forms inside the expander
        auth_forms()
# Only show API key input and model selection if authenticated
if st.session_state.authenticated:
    if st.sidebar.button("Sign Out", use_container_width=True):
        # Clear session state and redirect to login page
        st.session_state.authenticated = False
        st.session_state.user = None
        if "_auth_token_cookie" in st.session_state:
            del st.session_state["_auth_token_cookie"]
        st.experimental_rerun()

# Display user info if authenticated
if st.session_state.authenticated and st.session_state.user:
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Signed in as:**")
    user_data = st.session_state.user
    # Handle both object and dictionary user data formats
    try:
        if isinstance(user_data, dict):
            email = user_data.get('email', '')
        else:
            email = user_data.email if hasattr(user_data, 'email') else ''
        
        if email:
            st.sidebar.markdown(f"*{email}*")
        else:
            st.sidebar.markdown("*No email available*")
    except Exception as e:
        st.sidebar.markdown("*Error displaying email*")
    st.sidebar.markdown("---")

# Add significant spacing before the Links expander
st.sidebar.markdown("<br>" * 5, unsafe_allow_html=True)

# Add social media links to sidebar in an expander
with st.sidebar.expander("Links", expanded=True):
    st.write("Connect with FlameheadLabs:")
    
    sac.buttons([
        sac.ButtonsItem(label='About FlameheadLabs', icon='house', href='http://flameheadlabs.tech/'),
        sac.ButtonsItem(label='GitHub', icon='github', href='https://github.com/Flamehead-Labs-Ug/flame-audio'),
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
