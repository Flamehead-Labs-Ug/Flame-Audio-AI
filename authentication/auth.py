from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import logging
from dotenv import load_dotenv
from supabase import create_client, Client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("auth")

# Load environment variables
load_dotenv()

# Get Supabase credentials from environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")

# Supabase client for authentication
def get_supabase_client() -> Client:
    """Get a configured Supabase client"""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_ANON_KEY")
    
    if not supabase_url or not supabase_key:
        logger.error("Supabase URL or key not found in environment variables")
        raise ValueError("Supabase configuration missing. Check your .env file.")
        
    try:
        logger.info(f"Initializing Supabase client with URL: {supabase_url}")
        return create_client(supabase_url, supabase_key)
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {str(e)}")
        raise

# Initialize Supabase client
try:
    supabase = get_supabase_client()
    logger.info("Supabase client initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Supabase client: {str(e)}")
    # Don't crash the app, but record the error
    supabase = None

# Security bearer token scheme
security = HTTPBearer()

# Create a user model for authenticated users
class User:
    def __init__(self, id: str, email: str, user_metadata: dict = None):
        self.id = id
        self.email = email
        self.user_metadata = user_metadata or {}

def verify_auth(token: str):
    """Verify authentication token"""
    if not supabase:
        raise HTTPException(status_code=503, detail="Authentication service unavailable")
        
    try:
        # Validate the JWT token
        user = supabase.auth.get_user(token)
        return user
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid authentication token: {str(e)}")

# Function to verify auth token and return user
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Verify JWT token and return a User object if token is valid
    """
    try:
        # Extract token
        token = credentials.credentials
        
        # Verify token with Supabase
        user = verify_auth(token)
        
        # Return user object
        return User(
            id=user.user.id,
            email=user.user.email,
            user_metadata=user.user.user_metadata
        )
    except Exception as e:
        # Invalid token
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )
