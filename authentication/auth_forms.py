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
import extra_streamlit_components as stx

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

# Get frontend URL from environment variables
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:8501")
logger.info(f"Using frontend URL: {FRONTEND_URL}")

# Get Supabase credentials from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

# Initialize Supabase client with basic initialization to avoid compatibility issues
try:
    # First try with default initialization
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
except TypeError as e:
    # Fall back to a more explicit initialization without proxy
    logger.warning(f"Default Supabase initialization failed: {e}")
    # Create with explicit options dictionary, avoiding any imports of internal classes
    options = {
        "auto_refresh_token": True,
        "persist_session": True,
    }
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY, options)
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {e}")
    # In case of complete failure, create a placeholder to prevent crashes
    supabase = None
    logger.warning("Continuing with Supabase disabled")

# Cookie management
def get_cookie_manager():
    """Get or create a cookie manager"""
    if 'cookie_manager' not in st.session_state:
        st.session_state.cookie_manager = stx.CookieManager()
    return st.session_state.cookie_manager

# Session management
def save_session_data(token, user_data=None):
    """Save token to a cookie"""
    try:
        cookie_manager = get_cookie_manager()
        # Store token in cookie with 24 hour expiry
        cookie_manager.set("auth_token", token, expires_at=24*60*60)
        
        # Also store user ID if available
        if user_data and "id" in user_data:
            cookie_manager.set("user_id", user_data["id"], expires_at=24*60*60)
            
        logger.info(f"Session data saved to cookie")
        return True
    except Exception as e:
        logger.error(f"Error saving session: {e}")
        return False

def load_session_data():
    """Load token from cookie"""
    try:
        cookie_manager = get_cookie_manager()
        token = cookie_manager.get("auth_token")
        
        if token:
            logger.info("Session token loaded from cookie")
            return token
        else:
            logger.info("No session token found in cookie")
            return None
    except Exception as e:
        logger.error(f"Error loading session: {e}")
        return None

def clear_session_data():
    """Clear session data from cookies"""
    try:
        cookie_manager = get_cookie_manager()
        cookie_manager.delete("auth_token")
        cookie_manager.delete("user_id")
        logger.info("Session data cleared")
        return True
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        return False

def init_auth_session():
    """Initialize authentication session state variables using cookies"""
    # Initialize base session state for UI
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if "user" not in st.session_state:
        st.session_state.user = None
    
    # Try to load token from cookie
    auth_token = load_session_data()
    
    if auth_token:
        # Verify token with backend
        try:
            response = requests.get(
                f"{BACKEND_URL}/auth/verify",
                headers={"Authorization": f"Bearer {auth_token}"}
            )
            
            if response.status_code == 200:
                # Token is valid, set session state
                user_data = response.json()
                st.session_state.authenticated = True
                st.session_state.user = user_data
                logger.info("User authenticated from cookie")
                return
            else:
                # Token is invalid, clear it
                clear_session_data()
                logger.warning("Invalid session token in cookie")
        except Exception as e:
            logger.error(f"Error verifying token: {e}")
    
    # If we get here, either no token or invalid token
    st.session_state.authenticated = False
    st.session_state.user = None

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
            
            # Set the session state and store authentication in cookie
            st.session_state.authenticated = True
            st.session_state.user = user
            
            # Store the token in a cookie
            save_session_data(query_params["access_token"], user)
            
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
                            # Set session state and store session data in cookie
                            st.session_state.authenticated = True
                            st.session_state.user = data["user"]
                            
                            # Store the token in cookie
                            save_session_data(data["access_token"], data["user"])
                        
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
            # Remove token from cookie and clear session state
            auth_token = load_session_data()
            
            if auth_token:
                # Call logout endpoint
                response = requests.post(
                    f"{BACKEND_URL}/auth/logout",
                    headers={"Authorization": f"Bearer {auth_token}"}
                )
                
                # Clear cookies
                clear_session_data()
            
            # Clear session state
            st.session_state.authenticated = False
            st.session_state.user = None
            
            # Show success message and reload
            st.success("Successfully logged out!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Error logging out: {str(e)}")
    else:
        st.warning("You are not logged in.")