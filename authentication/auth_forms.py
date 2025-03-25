import os
import requests
import streamlit as st
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client
import uuid
import streamlit_antd_components as sac
import base64

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("auth_forms")

# Check if authentication is enabled
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() == "true"

# Get backend URL from environment variables
BACKEND_URL = os.getenv("BACKEND_URL")
if not BACKEND_URL:
    logger.error("BACKEND_URL not found in environment variables. Please set it in the .env file.")
    # Cannot use st.error here as this module might be imported before streamlit is ready

# Get Supabase credentials from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# Use an absolute path for the session file
# Store in user's home directory to ensure write permission
SESSION_FILE = os.path.join(os.path.expanduser("~"), ".flame_session_data")
logger.info(f"Using session file: {SESSION_FILE}")

def save_session_data(token):
    """Save token to a persistent file"""
    try:
        # Create a simple encoded token
        encoded = base64.b64encode(token.encode()).decode()
        with open(SESSION_FILE, "w") as f:
            f.write(encoded)
        logger.info(f"Successfully saved session data to {SESSION_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error saving session: {e}")
        return False

def load_session_data():
    """Load token from persistent file"""
    try:
        if os.path.exists(SESSION_FILE):
            logger.info(f"Session file found at {SESSION_FILE}")
            with open(SESSION_FILE, "r") as f:
                encoded = f.read().strip()
                if encoded:
                    token = base64.b64decode(encoded).decode()
                    logger.info("Successfully loaded session token")
                    return token
                else:
                    logger.warning("Session file exists but is empty")
        else:
            logger.info("No session file found")
    except Exception as e:
        logger.error(f"Error loading session: {e}")
    return None

def clear_session_data():
    """Remove the persistent session file"""
    try:
        if os.path.exists(SESSION_FILE):
            os.remove(SESSION_FILE)
            logger.info(f"Removed session file {SESSION_FILE}")
        else:
            logger.info("No session file to remove")
    except Exception as e:
        logger.error(f"Error clearing session: {e}")

def init_auth_session():
    """Initialize authentication session state variables"""
    # If authentication is disabled, set authenticated to True and bypass checks
    if not AUTH_ENABLED:
        logger.info("Authentication disabled via environment settings")
        st.session_state.authenticated = True
        st.session_state.user = {"email": "guest@example.com", "id": "guest"}
        return
        
    if "authenticated" not in st.session_state:
        logger.info("Initializing authentication session state")
        # Check for existing session token in persistent storage
        auth_token = load_session_data()
        if auth_token:
            logger.info("Found existing auth token, verifying...")
            try:
                # Verify token using FastAPI backend
                response = requests.get(
                    f"{BACKEND_URL}/auth/verify",
                    headers={"Authorization": f"Bearer {auth_token}"})
                logger.info(f"Token verification response: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info("Token verification successful")
                    st.session_state.authenticated = True
                    st.session_state.user = data
                    st.session_state["_auth_token_"] = auth_token
                    logger.info("Session restored successfully")
                else:
                    # Clear invalid session
                    logger.warning(f"Token verification failed: {response.text}")
                    st.session_state.authenticated = False
                    st.session_state.user = None
                    if "_auth_token_" in st.session_state:
                        del st.session_state["_auth_token_"]
                    clear_session_data()
            except Exception as e:
                # Log the error for debugging
                logger.error(f"Authentication error: {str(e)}")
                # Clear invalid session
                st.session_state.authenticated = False
                st.session_state.user = None
                if "_auth_token_" in st.session_state:
                    del st.session_state["_auth_token_"]
                clear_session_data()
        else:
            logger.info("No existing auth token found")
            st.session_state.authenticated = False
            st.session_state.user = None
    else:
        logger.info(f"Authentication state already initialized: {st.session_state.authenticated}")
    
    # Always regenerate auth_state if not in the middle of authentication flow
    # This ensures a fresh state for each authentication attempt
    if "auth_state" not in st.session_state or not st.query_params.get("state"):
        st.session_state.auth_state = str(uuid.uuid4())
        # Store the state in session state to persist across page refreshes
        st.session_state["_auth_state_"] = st.session_state.auth_state

def handle_auth_callback():
    """Handle authentication callback from Supabase"""
    # Get query parameters
    query_params = st.query_params
    
    # Verify state parameter to prevent CSRF attacks
    if "state" not in query_params:
        st.query_params.clear()
        return
        
    received_state = query_params["state"]
    if "auth_state" not in st.session_state or received_state != st.session_state.auth_state:
        st.query_params.clear()
        return
    
    # Process the authentication response
    if "access_token" in query_params:
        try:
            # Get the user from the session
            user = supabase.auth.get_user(query_params["access_token"])
            
            # Set the session state and store authentication persistently
            st.session_state.authenticated = True
            st.session_state.user = user
            st.session_state["_auth_user_"] = user  # Store user data persistently
            st.session_state["_auth_token_"] = query_params["access_token"]

            # Store the token in a persistent file
            save_session_data(query_params["access_token"])
            
            # Clear the state and URL parameters
            if "auth_state" in st.session_state:
                del st.session_state.auth_state
            st.query_params.clear()
            
            # Show success message
            st.success("Successfully logged in!")
            
        except Exception as e:
            st.error(f"Error authenticating: {str(e)}")
    
    # Handle authentication errors
    elif "error" in query_params and "error_description" in query_params:
        st.error(f"Authentication error: {query_params['error_description']}")
        if "auth_state" in st.session_state:
            del st.session_state.auth_state
        st.query_params.clear()

def auth_forms():
    """Display authentication forms"""
    # Create a compact container for the popover
    with st.container():
        
        
        # Create tabs for login and signup
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        
        with tab1:
            st.subheader("Login")
            
            # Email login form
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login", use_container_width=True)
                
                if submit:
                    if not email or not password:
                        st.error("Please enter both email and password.")
                        return

                    try:
                        # Sign in using FastAPI backend
                        response = requests.post(
                            f"{BACKEND_URL}/auth/login",
                            json={"email": email, "password": password}
                        )
                        
                        if response.status_code == 200:
                            data = response.json()
                            # Set session state and store session data
                            st.session_state.authenticated = True
                            st.session_state.user = data["user"]
                            st.session_state["_auth_token_"] = data["access_token"]
                            st.session_state["_auth_user_"] = data["user"]
                            st.session_state["_auth_state_"] = str(uuid.uuid4())
                            st.session_state["_request_headers_"] = {"Authorization": f"Bearer {data['access_token']}"}

                            # Store the token in persistent storage
                            save_session_data(data["access_token"])
                        
                            # Show success message and reload
                            st.success("Login successful!")
                            st.rerun()
                        
                        else:
                            error_message = response.json().get("detail", "Unknown error")
                            st.error(f"Login failed: {error_message}")
                        
                    except Exception as e:
                        error_message = str(e).lower()
                        if "invalid login credentials" in error_message:
                            st.error("Invalid email or password. Please check your credentials and try again.")
                        elif "email not confirmed" in error_message:
                            st.error("Please verify your email address before logging in.")
                        elif "too many requests" in error_message:
                            st.error("Too many login attempts. Please try again later.")
                        else:
                            st.error(f"Login failed: {str(e)}")
        
        with tab2:
                st.subheader("Sign Up")
                
                # Sign up form
                with st.form("signup_form"):
                    email = st.text_input("Email")
                    password = st.text_input("Password", type="password")
                    confirm_password = st.text_input("Confirm Password", type="password")
                    submit = st.form_submit_button("Sign Up", use_container_width=True)
                    
                    if submit:
                        if password != confirm_password:
                            st.error("Passwords do not match!")
                        else:
                            try:
                                # Sign up using FastAPI backend
                                response = requests.post(
                                    f"{BACKEND_URL}/auth/signup",
                                    json={"email": email, "password": password}
                                )
                                
                                if response.status_code == 200:
                                    data = response.json()
                                    st.success("Sign up successful! Please check your email to confirm your account.")
                                else:
                                    error_message = response.json().get("detail", "Unknown error")
                                    st.error(f"Sign up failed: {error_message}")
                                    
                            except Exception as e:
                                st.error(f"Sign up failed: {str(e)}")

def logout():
    """Log out the current user"""
    if st.session_state.authenticated:
        try:
            # Remove token from session state and clear persistent data
            if "_auth_token_" in st.session_state:
                # Call logout endpoint
                response = requests.post(
                    f"{BACKEND_URL}/auth/logout",
                    headers={"Authorization": f"Bearer {st.session_state['_auth_token_']}"}
                )
                
                # Clear session state
                del st.session_state["_auth_token_"]
            
            # Clear the persistent session data
            clear_session_data()
            
            # Reset session state
            st.session_state.authenticated = False
            st.session_state.user = None
            
            # Clear all authentication-related session state
            for key in list(st.session_state.keys()):
                if key.startswith("_auth_"):
                    del st.session_state[key]
            
            st.success("Successfully logged out!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error logging out: {str(e)}")
    else:
        st.warning("You are not currently logged in.")