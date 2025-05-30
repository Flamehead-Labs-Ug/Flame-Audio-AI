from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Header, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Union, Dict, Any
import os
import aiofiles
import tempfile
from pathlib import Path
from audio_processor import transcribe_audio_in_chunks
from dotenv import load_dotenv
import logging
from supabase import create_client, Client
import httpx
from authentication.auth_routes import router as auth_router, verify_token
from database.routes import router as db_router
from database.vector_store import get_vector_store
from authentication.auth import get_current_user
from embedding import generate_embedding
from database.chat_routes import router as chat_router
from database.vector_store_routes import router as vector_store_router
from database.mcp_routes import router as mcp_router
from database.pg_connector import get_pg_db
from deep_remove_key import deep_remove_key

# MCP integration has been moved to a separate service
# See the mcp_service directory for details
#
# To use the MCP service:
# 1. Ensure this main backend is running on port 8000
# 2. Start the MCP service using: mcp_service\run_mcp_service.bat
# 3. The MCP service URL is set by the MCP_URL environment variable (default: http://localhost:8001)

# Initialize security scheme
security = HTTPBearer()

# Load environment variables from .env file
load_dotenv()

# Check if authentication is enabled
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() == "true"

# Initialize FastAPI app
app = FastAPI(
    title="Flame Audio API",
    description="API for audio transcription and translation using Groq API",
    version="1.0.0",
    openapi_url="/api/openapi.json",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_ANON_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Include routers
app.include_router(auth_router, prefix="/api")
# Register database routes at /api, so /agents is available at /api/agents for MCP compatibility
app.include_router(db_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(vector_store_router, prefix="/api")
app.include_router(mcp_router, prefix="/api")

# MCP routes have been moved to the separate MCP service
# See the mcp_service directory for details
# The MCP service provides the following tools:
# - transcribe_audio: Transcribe audio from a URL
# - search_documents: Search documents using vector search
# - get_document_by_id: Get a document by its ID
# - get_models: Get available audio models
# - get_languages: Get supported languages for transcription

from fastapi import APIRouter
import httpx

@app.post("/langgraph/tools/call")
async def langgraph_tools_call(request: Request):
    """
    Proxy endpoint for LangGraph tool invocation. Accepts JSON with 'tool_name' and 'args',
    forwards to MCP service, and returns the result.
    """
    try:
        payload = await request.json()
        tool_name = payload.get("tool_name")
        args = payload.get("args", {})
        if not tool_name:
            raise HTTPException(status_code=400, detail="Missing 'tool_name' in request body")
        mcp_url = os.environ.get("MCP_URL", "http://localhost:8001") + "/tools/call"
        mcp_payload = {"tool_name": tool_name, "args": args}
        async with httpx.AsyncClient() as client:
            mcp_response = await client.post(mcp_url, json=mcp_payload)
            if mcp_response.status_code == 200:
                return JSONResponse(content=mcp_response.json())
            else:
                return JSONResponse(status_code=mcp_response.status_code, content={"error": mcp_response.text})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ---

import logging
for route in app.routes:
    logging.info(f"Registered route: {route.path} [{route.methods}]")

# Add /api/chat_models endpoint for MCP tool compatibility
from database.chat_routes import get_chat_models, validate_chat_model
from fastapi import Depends, Request
from authentication.auth import get_current_user
from database.routes import get_user_agents
from langchain_groq import ChatGroq
from typing import List, Dict

@app.get("/api/chat_models")
async def api_chat_models(current_user = Depends(get_current_user)):
    return await get_chat_models(current_user)

# Alias: /api/agents forwards to the same logic as /api/db/agents for MCP compatibility
@app.get("/api/agents")
async def api_agents(current_user = Depends(get_current_user)):
    return await get_user_agents(current_user)

# Add a chat completions endpoint for OpenAI-compatible API
@app.post("/api/chat/completions")
async def chat_completions(request: Request, current_user = Depends(get_current_user)):
    """Process chat completions requests in OpenAI-compatible format"""
    try:
        # Parse the request body
        body = await request.json()
        model = body.get("model", "llama3-70b-8192")
        messages = body.get("messages", [])
        temperature = float(body.get("temperature", 0.7))
        max_tokens = int(body.get("max_tokens", 1024))
        top_p = float(body.get("top_p", 0.9))

        # Clean parameters: remove 'proxies' if present anywhere
        cleaned_body = deep_remove_key(body, "proxies")
        logging.info(f"Cleaned chat completion parameters: {cleaned_body}")

        # Validate model name dynamically using chat_routes utility
        await validate_chat_model(cleaned_body.get("model", model))

        # Get Groq API key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(
                status_code=400,
                detail="GROQ_API_KEY not found in environment variables"
            )

        # Create Groq client
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=cleaned_body.get("model", model),
            temperature=float(cleaned_body.get("temperature", temperature)),
            max_tokens=int(cleaned_body.get("max_tokens", max_tokens)),
            top_p=float(cleaned_body.get("top_p", top_p))
        )

        # Format messages for Groq
        formatted_messages = []
        for msg in cleaned_body.get("messages", []):
            if msg.get("role") and msg.get("content"):
                if msg["role"] == "system":
                    from langchain_core.messages import SystemMessage
                    formatted_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    from langchain_core.messages import HumanMessage
                    formatted_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    from langchain_core.messages import AIMessage
                    formatted_messages.append(AIMessage(content=msg["content"]))

        # Call the model
        response = llm.invoke(formatted_messages)

        # Format the response in OpenAI-compatible format
        return {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": response.content
                    },
                    "index": 0,
                    "finish_reason": "stop"
                }
            ],
            "model": model,
            "object": "chat.completion"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat completions request: {str(e)}"
        )

# Initialize database connections
@app.on_event("startup")
async def startup_db_client():
    try:
        # Initialize PostgreSQL connection
        pg_db = get_pg_db()
        logging.info("PostgreSQL database connection initialized")

        # Initialize Vector Store connection
        vector_store = get_vector_store()
        logging.info("Vector Store connection initialized")
    except Exception as e:
        logging.error(f"Error initializing database connections: {e}")

@app.on_event("shutdown")
async def shutdown_db_client():
    try:
        # Close PostgreSQL connection
        pg_db = get_pg_db()
        pg_db.disconnect()
        logging.info("PostgreSQL database connection closed")
    except Exception as e:
        logging.error(f"Error closing database connections: {e}")

# Get Groq API key from environment
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")

# Base URL for Groq API
BASE_URL = "https://api.groq.com/openai/v1"

# Import supported languages from configuration
from languages import SUPPORTED_LANGUAGES

@app.get("/api/languages")
async def get_supported_languages():
    """Get list of supported languages for transcription"""
    return {"languages": SUPPORTED_LANGUAGES}

class AudioModel(BaseModel):
    model: str
    response_format: str = Field(
        default="text",
        description="Output format: 'json' for basic response, 'verbose_json' for timestamps, 'text' for text-only"
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Controls randomness in translation output (0.0 to 1.0)"
    )
    language: Optional[str] = Field(
        default=None,
        description="Language code for transcription (not used for translation)"
    )
    task: str = Field(
        default="transcribe",
        description="Task to perform: 'transcribe' or 'translate'"
    )
    chunk_length: int = Field(
        default=1800,
        description="Length of each audio chunk in seconds"
    )
    overlap: int = Field(
        default=10,
        description="Overlap between chunks in seconds"
    )

class TranscriptionResponse(BaseModel):
    text: str
    segments: Optional[List[Dict[str, Any]]] = None

async def validate_audio_file(file: UploadFile):
    """Validate audio file type and size."""
    allowed_types = {
        'audio/mpeg', 'audio/mp3', 'audio/wav', 'audio/wave',
        'audio/x-wav', 'audio/flac', 'audio/ogg', 'audio/webm',
        'video/mp4', 'video/mpeg', 'video/webm'
    }

    # Get content type from the file
    content_type = file.content_type
    if not content_type:
        # Try to guess content type from file extension
        ext_to_type = {
            '.mp3': 'audio/mpeg',
            '.wav': 'audio/wav',
            '.flac': 'audio/flac',
            '.ogg': 'audio/ogg',
            '.m4a': 'audio/mp4',
            '.webm': 'audio/webm',
            '.mp4': 'video/mp4',
            '.mpeg': 'video/mpeg',
            '.mpg': 'video/mpeg'
        }
        ext = Path(file.filename).suffix.lower()
        content_type = ext_to_type.get(ext)

    if not content_type or content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {content_type}. Must be one of: {', '.join(allowed_types)}"
        )

    # Save file temporarily to check size
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    try:
        content = await file.read()
        await file.seek(0)  # Reset file pointer

        # Check file size (25MB limit)
        if len(content) > 25 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="File size too large. Maximum size is 25MB."
            )

        return True

    finally:
        temp_file.close()
        os.unlink(temp_file.name)

@app.post("/api/audio/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form(...),
    response_format: str = Form("text"),
    task: str = Form("transcribe"),
    language: Optional[str] = Form(None),
    temperature: float = Form(0.0),
    chunk_length: int = Form(600),  # Default to 10 minutes for better chunking
    overlap: int = Form(10),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Verify token first
    try:
        user = await verify_token(credentials)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token"
            )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Authentication failed: {str(e)}"
        )
    # Validate language code if provided
    if language and not any(lang["code"] == language for lang in SUPPORTED_LANGUAGES):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language code: {language}"
        )
    """
    Transcribe audio using the Groq API.
    """
    logger = logging.getLogger(__name__)

    # Configure logging
    logging.basicConfig(level=logging.DEBUG)

    # Log request details
    logger.info(f"Received transcription request for file: {file.filename}")
    logger.info(f"Model: {model}, Task: {task}, Language: {language}, Response format: {response_format}")
    logger.info(f"Chunk length: {chunk_length}, Overlap: {overlap}, Temperature: {temperature}")

    try:
        # Create a temporary file to store the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            # Write the file content to the temporary file
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

            logger.info(f"Saved uploaded file to temporary path: {temp_path}")
            logger.info(f"File size: {len(content) / 1024:.2f} KB")

        try:
            # Process the audio file
            logger.info("Starting audio processing...")
            result = transcribe_audio_in_chunks(
                audio_path=Path(temp_path),
                chunk_length=chunk_length,
                overlap=overlap,
                model=model,
                temperature=temperature,
                task=task  # Pass the task parameter to correctly select the API endpoint
            )
            logger.info("Audio processing completed successfully")

            # Format the response based on the requested format
            if response_format == "text":
                logger.info("Returning text response")
                return {"text": result.get("text", "")}
            else:
                # Return the full result for other formats
                logger.info("Returning full JSON response")
                return result

        except Exception as e:
            # Log the error
            logger.error(f"Error processing audio: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing audio: {type(e).__name__}: {str(e)}"
            )
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_path)
                logger.info(f"Removed temporary file: {temp_path}")
            except Exception as e:
                logger.warning(f"Error removing temporary file: {str(e)}")

    except Exception as e:
        # Log the error
        logger.error(f"Error handling file upload: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error handling file upload: {type(e).__name__}: {str(e)}"
        )

@app.post("/api/audio/translate", response_model=TranscriptionResponse)
async def translate_audio(
    file: UploadFile = File(...),
    model: str = Form(...),
    response_format: str = Form("text"),
    temperature: float = Form(0.0),
    chunk_length: int = Form(600),  # Default to 10 minutes for better chunking
    overlap: int = Form(10),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Verify token first
    try:
        user = await verify_token(credentials)
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired token"
            )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Authentication failed: {str(e)}"
        )

    """
    Translate audio to English text using the Groq API.
    """
    logger = logging.getLogger(__name__)

    # Configure logging
    logging.basicConfig(level=logging.DEBUG)

    # Log request details
    logger.info(f"Received translation request for file: {file.filename}")
    logger.info(f"Model: {model}, Response format: {response_format}")
    logger.info(f"Chunk length: {chunk_length}, Overlap: {overlap}, Temperature: {temperature}")

    try:
        # Create a temporary file to store the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            # Write the file content to the temporary file
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name

            logger.info(f"Saved uploaded file to temporary path: {temp_path}")
            logger.info(f"File size: {len(content) / 1024:.2f} KB")

        try:
            # Process the audio file
            logger.info("Starting audio translation...")
            result = transcribe_audio_in_chunks(
                audio_path=Path(temp_path),
                chunk_length=chunk_length,
                overlap=overlap,
                model=model,
                temperature=temperature,
                task="translate"  # Always use translate for this endpoint
            )
            logger.info("Audio translation completed successfully")

            # Format the response based on the requested format
            if response_format == "text":
                logger.info("Returning text response")
                return {"text": result.get("text", "")}
            else:
                # Return the full result for other formats
                logger.info("Returning full JSON response")
                return result

        except Exception as e:
            # Log the error
            logger.error(f"Error processing audio: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing audio: {type(e).__name__}: {str(e)}"
            )
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_path)
                logger.info(f"Removed temporary file: {temp_path}")
            except Exception as e:
                logger.warning(f"Error removing temporary file: {str(e)}")

    except Exception as e:
        # Log the error
        logger.error(f"Error handling file upload: {type(e).__name__}: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error handling file upload: {type(e).__name__}: {str(e)}"
        )

import httpx

# Models endpoint
@app.get("/api/models")
async def get_models():
    """Get available audio models from Groq API."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/models",
                headers={"Authorization": f"Bearer {groq_api_key}"}
            )
            response.raise_for_status()
            models_data = response.json()

            # Filter for Whisper models (speech-to-text)
            speech_models = [
                {
                    "id": model["id"],
                    "name": model["id"].replace("-", " ").title(),
                    "description": model.get("description", "Speech-to-text model")
                }
                for model in models_data["data"]
                if "whisper" in model["id"].lower()
            ]

            return speech_models
    except httpx.HTTPError as e:
        raise HTTPException(status_code=e.response.status_code if hasattr(e, 'response') else 500,
                          detail=f"Error fetching models from Groq API: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Task options endpoint
@app.get("/api/tasks")
async def get_tasks():
    """
    Returns available task options for audio processing.
    """
    try:
        # Return available task options
        task_options = ["transcribe", "translate"]
        return {"tasks": task_options}
    except Exception as e:
        logging.error(f"Error getting task options: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/embed")
async def embed_text(text_data: dict, current_user = Depends(get_current_user)):
    """Generate an embedding vector for the given text"""
    if "text" not in text_data:
        raise HTTPException(status_code=400, detail="Text field is required")

    try:
        embedding = generate_embedding(text_data["text"])
        return {
            "embedding": embedding,
            "dimensions": len(embedding)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

# Run the application
@app.get("/api/auth/verify")
async def verify_auth(user = Depends(verify_token)):
    """Verify if the user is authenticated"""
    return {"authenticated": True, "user": user.dict()}

@app.post("/api/auth/logout")
async def logout(user = Depends(verify_token)):
    """Log out the current user"""
    try:
        # The actual logout happens on the client side by removing the token
        # Here we just verify the token is valid and return success
        return {"success": True, "message": "Logged out successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Logout failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
