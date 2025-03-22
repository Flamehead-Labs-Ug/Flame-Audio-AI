from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from supabase import create_client, Client
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Create router
router = APIRouter(prefix="/auth", tags=["authentication"])

# Security bearer token scheme
security = HTTPBearer()

# Pydantic models for request validation
class LoginRequest(BaseModel):
    email: str
    password: str

class SignupRequest(BaseModel):
    email: str
    password: str

@router.post("/login")
async def login(request: LoginRequest):
    try:
        # Sign in with Supabase using the correct method
        auth_response = supabase.auth.sign_in_with_password({
            "email": request.email,
            "password": request.password
        })
        
        # Handle potential error responses
        if not auth_response:
            raise HTTPException(
                status_code=401,
                detail="Invalid credentials"
            )
        
        # Extract user and session directly from the response
        user = auth_response.user
        session = auth_response.session
        
        if not user or not session:
            raise HTTPException(
                status_code=401,
                detail="Authentication failed - invalid credentials"
            )
        
        # Return the authentication response
        return {
            "access_token": session.access_token,
            "token_type": "bearer",
            "expires_in": session.expires_in,
            "refresh_token": session.refresh_token,
            "user": {
                "id": user.id,
                "email": user.email,
                "app_metadata": user.app_metadata,
                "user_metadata": user.user_metadata
            }
        }
    except Exception as e:
        # Log the error for debugging
        print(f"Login error: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail=f"Authentication failed: {str(e)}"
        )

@router.post("/signup")
async def signup(request: SignupRequest):
    try:
        # Sign up with Supabase
        response = await supabase.auth.sign_up({
            "email": request.email,
            "password": request.password
        })
        
        if not response or not hasattr(response, 'data'):
            raise HTTPException(
                status_code=400,
                detail="Invalid response from authentication server"
            )
            
        auth_data = response.data
        
        if not hasattr(auth_data, 'user'):
            raise HTTPException(
                status_code=400,
                detail="User data not found in response"
            )
            
        return {
            "message": "Sign up successful! Please check your email to confirm your account.",
            "user": auth_data.user
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

@router.post("/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        # Sign out with Supabase
        supabase.auth.sign_out(credentials.credentials)
        return {"message": "Logged out successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

@router.get("/verify")
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        # Verify token with Supabase
        print(f"Verifying token: {credentials.credentials[:10]}...")
        response = supabase.auth.get_user(credentials.credentials)
        
        if not response or not response.user:
            print("Invalid or expired token: No user found in response")
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token"
            )
        
        # Return user object - ensure it has expected format
        print(f"Token validation successful for user: {response.user.email}")
        return response.user
    except Exception as e:
        print(f"Token verification error: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail=f"Invalid or expired token: {str(e)}"
        )