import requests
from requests.exceptions import RequestException
from mcp.server.fastmcp import FastMCP
from mcp.shared.exceptions import McpError
from mcp.types import ErrorData, INTERNAL_ERROR, INVALID_PARAMS
import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request as StarletteRequest
from starlette.routing import Route, Mount
from mcp.server.sse import SseServerTransport
import inspect
import sys
import json
import re
from typing import Dict, Any, Optional, List, Union

# Create an MCP server instance for DB tools
mcp = FastMCP("Flame Audio MCP DB Tool Server")

# Helper function to convert Pydantic models to dictionaries
def convert_pydantic_to_dict(obj: Any) -> Any:
    """Convert Pydantic models to dictionaries recursively."""
    if hasattr(obj, "model_dump"):
        # Pydantic v2
        return obj.model_dump()
    elif hasattr(obj, "dict"):
        # Pydantic v1
        return obj.dict()
    elif isinstance(obj, dict):
        return {k: convert_pydantic_to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_pydantic_to_dict(item) for item in obj]
    else:
        return obj

# Utility: Deep remove key from nested dict/list

def deep_remove_key(obj, key):
    """
    Recursively remove all instances of `key` from nested dictionaries and lists.
    Args:
        obj: The dictionary or list to clean.
        key: The key to remove.
    """
    if isinstance(obj, dict):
        obj.pop(key, None)
        for v in obj.values():
            deep_remove_key(v, key)
    elif isinstance(obj, list):
        for item in obj:
            deep_remove_key(item, key)


# MCP REGISTRATION TEST TOOL

# USER CRUD TOOLS

# FILE UPLOAD MCP TOOLS
@mcp.tool()
def initiate_file_upload(token: str, file_type: str, file_name: str) -> str:
    """
    Initiate a file upload job and receive an upload URL or token.
    Args:
        token: Bearer token for authentication (from authenticate_user)
        file_type: 'audio' or 'document'
        file_name: Name of the file to be uploaded
    Returns:
        JSON string with upload URL/token or error message.
    """
    try:
        if not token or not file_type or not file_name:
            raise ValueError("Authentication token, file_type, and file_name are required.")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = f"http://localhost:8000/api/upload/initiate"
        data = {"file_type": file_type, "file_name": file_name}
        response = requests.post(url, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to initiate upload. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def check_upload_status(token: str, upload_id: str) -> str:
    """
    Check the status of a file upload job.
    Args:
        token: Bearer token for authentication (from authenticate_user)
        upload_id: The unique ID for the upload job
    Returns:
        JSON string with upload status or error message.
    """
    try:
        if not token or not upload_id:
            raise ValueError("Authentication token and upload_id are required.")
        headers = {"Authorization": f"Bearer {token}"}
        url = f"http://localhost:8000/api/upload/status/{upload_id}"
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        elif response.status_code == 404:
            raise McpError(ErrorData(INVALID_PARAMS, "Upload job not found."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to check upload status. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def create_user(user_data: dict) -> str:
    """
    Create a new user account.
    Args:
        user_data: Dictionary with user fields (e.g., email, password, name)
    Returns:
        JSON string of created user info or error message.
    """
    try:
        if not user_data:
            raise ValueError("user_data is required.")
        headers = {"Content-Type": "application/json"}
        url = "http://localhost:8000/api/users"
        response = requests.post(url, headers=headers, json=user_data, timeout=10)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 409:
            raise McpError(ErrorData(INVALID_PARAMS, "User already exists."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to create user. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def update_user(token: str, user_id: str, user_data: dict) -> str:
    """
    Update an existing user account.
    Args:
        token: Bearer token for authentication (from authenticate_user)
        user_id: The user's unique ID
        user_data: Dictionary with updated user fields
    Returns:
        JSON string of updated user info or error message.
    """
    try:
        if not token or not user_id or not user_data:
            raise ValueError("Authentication token, user_id, and user_data are required.")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = f"http://localhost:8000/api/users/{user_id}"
        response = requests.put(url, headers=headers, json=user_data, timeout=10)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        elif response.status_code == 404:
            raise McpError(ErrorData(INVALID_PARAMS, "User not found."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to update user. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def delete_user(token: str, user_id: str) -> str:
    """
    Delete a user account.
    Args:
        token: Bearer token for authentication (from authenticate_user)
        user_id: The user's unique ID
    Returns:
        JSON string of deletion result or error message.
    """
    try:
        if not token or not user_id:
            raise ValueError("Authentication token and user_id are required.")
        headers = {"Authorization": f"Bearer {token}"}
        url = f"http://localhost:8000/api/users/{user_id}"
        response = requests.delete(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        elif response.status_code == 404:
            raise McpError(ErrorData(INVALID_PARAMS, "User not found."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to delete user. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e


@mcp.tool()
def get_user_documents(token: str, agent_id: str = None) -> list:
    """
    Retrieve all user documents from the database API, optionally filtered by agent.
    Args:
        token: Bearer token for authentication (from authenticate_user)
        agent_id: (optional) Filter documents by agent ID
    Returns:
        List of document objects or empty list if none found.
    """
    try:
        if not token:
            raise ValueError("Authentication token is required.")
        headers = {"Authorization": f"Bearer {token}"}
        url = "http://localhost:8000/api/db/documents"
        params = {}
        if agent_id:
            params["agent_id"] = agent_id
        print(f"DEBUG: Making request to {url} with headers: {headers} and params: {params}")
        response = requests.get(url, headers=headers, params=params, timeout=10)
        print(f"DEBUG: Response status code: {response.status_code}")
        print(f"DEBUG: Response text: {response.text[:100]}...")
        if response.status_code == 200:
            # Parse the JSON response and return it as a Python object
            import json
            result = json.loads(response.text)
            print(f"DEBUG: get_user_documents result: {result}")
            return result
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to retrieve documents. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def create_session(token: str, user_id: str, expiry_days: int = 7) -> str:
    """
    Create a new user session.
    Args:
        token: Bearer token for authentication (from authenticate_user)
        user_id: The user's ID
        expiry_days: Number of days until session expiry (default 7)
    Returns:
        JSON string of session info or error message.
    """
    try:
        if not token or not user_id:
            raise ValueError("Authentication token and user_id are required.")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = "http://localhost:8000/api/db/sessions"
        data = {"user_id": user_id, "expiry_days": expiry_days}
        response = requests.post(url, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to create session. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def create_audio_job(token: str, job_data: dict) -> str:
    """
    Create a new audio job.
    Args:
        token: Bearer token for authentication (from authenticate_user)
        job_data: Dictionary with job fields (see AudioJobCreate)
    Returns:
        JSON string of job info or error message.
    """
    try:
        if not token or not job_data:
            raise ValueError("Authentication token and job_data are required.")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = "http://localhost:8000/api/db/jobs"
        response = requests.post(url, headers=headers, json=job_data, timeout=10)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to create audio job. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def save_transcription(token: str, transcription_data: dict) -> str:
    """
    Save a new transcription segment.
    Args:
        token: Bearer token for authentication (from authenticate_user)
        transcription_data: Dictionary with transcription fields (see TranscriptionCreate)
    Returns:
        JSON string of transcription info or error message.
    """
    try:
        if not token or not transcription_data:
            raise ValueError("Authentication token and transcription_data are required.")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = "http://localhost:8000/api/db/transcriptions"
        response = requests.post(url, headers=headers, json=transcription_data, timeout=10)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to save transcription. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def save_translation(token: str, translation_data: dict) -> str:
    """
    Save a translation segment.
    Args:
        token: Bearer token for authentication (from authenticate_user)
        translation_data: Dictionary with translation fields (see TranslationCreate)
    Returns:
        JSON string of translation info or error message.
    """
    try:
        if not token or not translation_data:
            raise ValueError("Authentication token and translation_data are required.")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = "http://localhost:8000/api/db/translations"
        response = requests.post(url, headers=headers, json=translation_data, timeout=10)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to save translation. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def save_document(token: str, document_data: dict) -> str:
    """
    Save a document to the database.
    Args:
        token: Bearer token for authentication (from authenticate_user)
        document_data: Dictionary with document fields (see DocumentData)
    Returns:
        JSON string of document info or error message.
    """
    try:
        if not token or not document_data:
            raise ValueError("Authentication token and document_data are required.")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = "http://localhost:8000/api/db/documents"
        response = requests.post(url, headers=headers, json=document_data, timeout=10)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to save document. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def store_vector(token: str, vector_data: dict) -> str:
    """
    Store an embedding vector for a transcription chunk.
    Args:
        token: Bearer token for authentication (from authenticate_user)
        vector_data: Dictionary with vector fields
    Returns:
        JSON string of vector info or error message.
    """
    try:
        if not token or not vector_data:
            raise ValueError("Authentication token and vector_data are required.")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = "http://localhost:8000/api/db/vectors"
        response = requests.post(url, headers=headers, json=vector_data, timeout=10)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to store vector. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def search_transcriptions(token: str, search_data: dict) -> str:
    """
    Search for similar transcriptions using vector similarity.
    Args:
        token: Bearer token for authentication (from authenticate_user)
        search_data: Dictionary with search fields (see SearchQuery)
    Returns:
        JSON string of search results or error message.
    """
    try:
        if not token or not search_data:
            raise ValueError("Authentication token and search_data are required.")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = "http://localhost:8000/api/db/search"
        response = requests.post(url, headers=headers, json=search_data, timeout=10)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to search transcriptions. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def get_user_agents(token: str) -> dict:
    """
    Get all agents created by the current user.
    Args:
        token: Bearer token for authentication (from authenticate_user)
    Returns:
        List of agent objects or empty list if none found.
    """
    try:
        print(f"DEBUG: get_user_agents called with token: {token[:10]}...")
        if not token:
            raise ValueError("Authentication token is required.")
        headers = {"Authorization": f"Bearer {token}"}
        url = "http://localhost:8000/api/db/agents"
        print(f"DEBUG: Making request to {url} with headers: {headers}")
        response = requests.get(url, headers=headers, timeout=10)
        print(f"DEBUG: Response status code: {response.status_code}")
        print(f"DEBUG: Response text: {response.text[:100]}...")
        if response.status_code == 200:
            # Parse the JSON response and return it as a Python object
            import json
            result = json.loads(response.text)
            print(f"DEBUG: get_user_agents result: {result}")
            return result
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to retrieve agents. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def get_qdrant_diagnostics(token: str, agent_id: str = None) -> str:
    """
    Get diagnostic information about Qdrant collection.
    Args:
        token: Bearer token for authentication (from authenticate_user)
        agent_id: (optional) Filter diagnostics by agent ID
    Returns:
        JSON string of diagnostics info or error message.
    """
    try:
        if not token:
            raise ValueError("Authentication token is required.")
        headers = {"Authorization": f"Bearer {token}"}
        url = "http://localhost:8000/api/db/diagnostics/qdrant"
        params = {}
        if agent_id:
            params["agent_id"] = agent_id
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to retrieve diagnostics. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def get_chat_models(token: str) -> dict:
    """
    List available chat models from the backend chat API.
    Args:
        token: Bearer token for authentication (from authenticate_user)
    Returns:
        List of chat model objects or empty list if none found.
    """
    try:
        print(f"DEBUG: get_chat_models called with token: {token[:10]}...")
        if not token:
            raise ValueError("Authentication token is required.")
        headers = {"Authorization": f"Bearer {token}"}
        url = "http://localhost:8000/api/chat/models"
        print(f"DEBUG: Making request to {url} with headers: {headers}")
        response = requests.get(url, headers=headers, timeout=10)
        print(f"DEBUG: Response status code: {response.status_code}")
        print(f"DEBUG: Response text: {response.text[:100]}...")
        if response.status_code == 200:
            # Parse the JSON response and return it as a Python object
            import json
            result = json.loads(response.text)
            print(f"DEBUG: get_chat_models result: {result}")
            return result
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to retrieve chat models. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def create_chat_session(token: str, session_data: dict) -> str:
    """
    Create a new chat session.
    Args:
        token: Bearer token for authentication (from authenticate_user)
        session_data: Dictionary with chat session fields (see ChatSessionCreate)
    Returns:
        JSON string of session info or error message.
    """
    try:
        print(f"DEBUG: create_chat_session called with token: {token[:10]}... and session_data: {session_data}")
        if not token or not session_data:
            raise ValueError("Authentication token and session_data are required.")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = "http://localhost:8000/api/chat/sessions"
        print(f"DEBUG: Making request to {url} with headers: {headers} and data: {session_data}")
        response = requests.post(url, headers=headers, json=session_data, timeout=10)
        print(f"DEBUG: Response status code: {response.status_code}")
        print(f"DEBUG: Response text: {response.text[:100]}...")
        if response.status_code == 200:
            return response.text
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to create chat session. HTTP status code: {response.status_code}, response: {response.text}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def get_user_chat_sessions(token: str, agent_id: str = None) -> list:
    """
    List user chat sessions, optionally filtered by agent.
    Args:
        token: Bearer token for authentication (from authenticate_user)
        agent_id: (optional) Filter sessions by agent ID
    Returns:
        List of chat session objects or empty list if none found.
    """
    try:
        if not token:
            raise ValueError("Authentication token is required.")
        headers = {"Authorization": f"Bearer {token}"}
        url = "http://localhost:8000/api/chat/sessions"
        params = {}
        if agent_id:
            params["agent_id"] = agent_id
        print(f"DEBUG: Making request to {url} with headers: {headers} and params: {params}")
        response = requests.get(url, headers=headers, params=params, timeout=10)
        print(f"DEBUG: Response status code: {response.status_code}")
        print(f"DEBUG: Response text: {response.text[:100]}...")
        if response.status_code == 200:
            # Parse the JSON response and return it as a Python object
            import json
            result = json.loads(response.text)
            print(f"DEBUG: get_user_chat_sessions result: {result}")
            return result
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to retrieve chat sessions. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def get_chat_session(token: str, session_id: str) -> dict:
    """
    Get a specific chat session and its messages.
    Args:
        token: Bearer token for authentication (from authenticate_user)
        session_id: The chat session ID
    Returns:
        Dictionary with chat session info or empty dict if not found.
    """
    try:
        if not token or not session_id:
            raise ValueError("Authentication token and session_id are required.")
        headers = {"Authorization": f"Bearer {token}"}
        url = f"http://localhost:8000/api/chat/sessions/{session_id}"
        print(f"DEBUG: Making request to {url} with headers: {headers}")
        response = requests.get(url, headers=headers, timeout=10)
        print(f"DEBUG: Response status code: {response.status_code}")
        print(f"DEBUG: Response text: {response.text[:100]}...")
        if response.status_code == 200:
            # Parse the JSON response and return it as a Python object
            import json
            result = json.loads(response.text)
            print(f"DEBUG: get_chat_session result: {result}")
            return result
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        elif response.status_code == 404:
            raise McpError(ErrorData(INVALID_PARAMS, "Chat session not found."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to retrieve chat session. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def delete_chat_session(token: str, session_id: str) -> dict:
    """
    Delete a chat session and all its messages.
    Args:
        token: Bearer token for authentication (from authenticate_user)
        session_id: The chat session ID
    Returns:
        Dictionary with deletion result or empty dict if not found.
    """
    try:
        if not token or not session_id:
            raise ValueError("Authentication token and session_id are required.")
        headers = {"Authorization": f"Bearer {token}"}
        url = f"http://localhost:8000/api/chat/sessions/{session_id}"
        print(f"DEBUG: Making request to {url} with headers: {headers}")
        response = requests.delete(url, headers=headers, timeout=10)
        print(f"DEBUG: Response status code: {response.status_code}")
        print(f"DEBUG: Response text: {response.text[:100]}...")
        if response.status_code == 200:
            # Parse the JSON response and return it as a Python object
            import json
            result = json.loads(response.text)
            print(f"DEBUG: delete_chat_session result: {result}")
            return result
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        elif response.status_code == 404:
            raise McpError(ErrorData(INVALID_PARAMS, "Chat session not found."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to delete chat session. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def chat_with_agent(
    token: str,
    question: str,
    retrieved_chunks: list = None,
    system_prompt: str = None,
    agent_id: str = None,
    document_id: str = None,
    parameters: dict = None,
    metadata: dict = None,
    session_id: str = None,
) -> dict:
    """
    Chat with an agent using structured RAG input. Supports all RAG parameters directly (question, retrieved_chunks, system_prompt, agent_id, document_id, parameters, metadata, session_id).
    Args:
        token: Bearer token for authentication (from authenticate_user)
        question: User question
        retrieved_chunks: List of dicts with chunk info
        system_prompt: Optional system prompt
        agent_id: Optional agent id
        document_id: Optional document id
        parameters: Optional chat parameters
        metadata: Optional extra metadata
        session_id: Optional chat session id
    Returns:
        Dictionary with chat response or empty dict if error.
    """
    try:
        print(f"DEBUG: chat_with_agent called with token: {token[:10]}... and session_id: {session_id}")
        print(f"DEBUG: agent_id: {agent_id}, document_id: {document_id}")
        print(f"DEBUG: question: {question}")

        if not token or not question:
            raise ValueError("Authentication token and question are required.")

        chat_request = {
            "message": question,
            "retrieved_chunks": retrieved_chunks,
            "system_prompt": system_prompt,
            "agent_id": agent_id,
            "document_id": document_id,
            "parameters": parameters or {},
            "metadata": metadata or {},
        }

        print(f"DEBUG: Created chat_request: {chat_request}")
        if session_id:
            chat_request["session_id"] = session_id
            print(f"DEBUG: Added session_id {session_id} to chat_request")

        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = "http://localhost:8000/api/chat/"

        # Recursively remove all problematic keys from parameters
        if "parameters" in request_data:
            for key in ["proxies", "http_client", "http_async_client", "client", "async_client"]:
                deep_remove_key(request_data["parameters"], key)

        # If session_id is provided, check if it exists first
        if "session_id" in request_data and request_data["session_id"]:
            session_id_val = request_data["session_id"]
            session_url = f"http://localhost:8000/api/chat/sessions/{session_id_val}"
            try:
                print(f"DEBUG: Checking if session {session_id_val} exists at {session_url}")
                session_response = requests.get(session_url, headers=headers, timeout=10)
                print(f"DEBUG: Session check response: {session_response.status_code}")
                if session_response.status_code != 200:
                    print(f"DEBUG: Session {session_id_val} not found, removing from request")
                    del request_data["session_id"]
                else:
                    print(f"DEBUG: Session {session_id_val} exists, keeping in request")
            except Exception as e:
                print(f"DEBUG: Error checking session: {str(e)}")
                del request_data["session_id"]

        print(f"DEBUG: Sending chat request to {url} with data: {request_data}")
        response = requests.post(url, headers=headers, json=request_data, timeout=30)
        print(f"DEBUG: Chat response status code: {response.status_code}")
        print(f"DEBUG: Chat response text: {response.text[:100]}...")

        if response.status_code == 200:
            result = json.loads(response.text)
            print(f"DEBUG: Raw result from backend: {result}")

            # Extract source documents if present
            metadata = result.get("metadata", {})
            source_documents = result.get("source_documents", [])

            # If source_documents is in the top level, add it to metadata
            if source_documents and "source_documents" not in metadata:
                metadata["source_documents"] = source_documents
                print(f"DEBUG: Added source_documents to metadata: {len(source_documents)} documents")

            # Process source documents to ensure they have proper metadata
            if "source_documents" in metadata:
                for doc in metadata["source_documents"]:
                    # Make sure each document has a metadata field
                    if "metadata" not in doc:
                        doc["metadata"] = {}

                    # Check if we have nested metadata (common in Qdrant responses)
                    nested_metadata = doc["metadata"].get("metadata", {})

                    # Extract segment information from the document content if available
                    content = doc.get("content", "")
                    if content and "Segment" in content:
                        try:
                            # Try to extract segment number from content
                            segment_match = re.search(r"Segment\s+(\d+)", content)
                            if segment_match:
                                segment_num = segment_match.group(1)
                                # Add segment to both levels of metadata for compatibility
                                doc["metadata"]["segment"] = segment_num
                                if nested_metadata:
                                    doc["metadata"]["metadata"]["segment"] = segment_num
                                print(f"DEBUG: Extracted segment {segment_num} from content")
                        except Exception as e:
                            print(f"DEBUG: Error extracting segment info: {str(e)}")

                    # If we have chunk_index in nested metadata but not segment, use chunk_index as segment
                    if nested_metadata and "chunk_index" in nested_metadata and "segment" not in nested_metadata:
                        chunk_index = nested_metadata.get("chunk_index")
                        if chunk_index is not None:
                            # Add segment to both levels of metadata for compatibility
                            doc["metadata"]["segment"] = str(chunk_index + 1)  # Make 1-based for display
                            nested_metadata["segment"] = str(chunk_index + 1)  # Make 1-based for display
                            print(f"DEBUG: Using chunk_index {chunk_index} as segment {chunk_index + 1}")

                    # If we still don't have a segment, set it to 1 by default
                    if "segment" not in doc["metadata"]:
                        doc["metadata"]["segment"] = "1"
                        if nested_metadata:
                            nested_metadata["segment"] = "1"
                        print(f"DEBUG: Setting default segment to 1")

            formatted_result = {
                "success": True,
                "response": result.get("response", ""),
                "session_id": result.get("session_id", ""),
                "messages": result.get("messages", []),
                "metadata": metadata
            }
            print(f"DEBUG: Formatted result: {formatted_result}")
            return formatted_result
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to chat with agent. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def debug_vector_store(token: str) -> str:
    """
    Debug vector store connection for chat API.
    Args:
        token: Bearer token for authentication (from authenticate_user)
    Returns:
        JSON string of debug info or error message.
    """
    try:
        if not token:
            raise ValueError("Authentication token is required.")
        headers = {"Authorization": f"Bearer {token}"}
        url = "http://localhost:8000/api/chat/debug/vector_store"
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to debug vector store. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def get_vectorstore_settings(token: str) -> str:
    """
    Get current vector store settings for the authenticated user.
    Args:
        token: Bearer token for authentication (from authenticate_user)
    Returns:
        JSON string of vector store settings or error message.
    """
    try:
        if not token:
            raise ValueError("Authentication token is required.")
        headers = {"Authorization": f"Bearer {token}"}
        url = "http://localhost:8000/api/vectorstore/settings"
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to retrieve vector store settings. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def save_vectorstore_settings(token: str, settings: dict) -> str:
    """
    Save or update vector store settings for the authenticated user.
    Args:
        token: Bearer token for authentication (from authenticate_user)
        settings: Dictionary with vector store settings (see VectorStoreSettings)
    Returns:
        JSON string of saved settings or error message.
    """
    try:
        if not token or not settings:
            raise ValueError("Authentication token and settings are required.")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = "http://localhost:8000/api/vectorstore/settings"
        response = requests.post(url, headers=headers, json=settings, timeout=10)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to save vector store settings. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def get_embedding_models(token: str) -> str:
    """
    List available embedding models for the vector store.
    Args:
        token: Bearer token for authentication (from authenticate_user)
    Returns:
        JSON string of embedding models or error message.
    """
    try:
        if not token:
            raise ValueError("Authentication token is required.")
        headers = {"Authorization": f"Bearer {token}"}
        url = "http://localhost:8000/api/vectorstore/models"
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to retrieve embedding models. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def get_vector_store_collections(token: str) -> str:
    """
    Get information about all vector store collections for the user.
    Args:
        token: Bearer token for authentication (from authenticate_user)
    Returns:
        JSON string of collections info or error message.
    """
    try:
        if not token:
            raise ValueError("Authentication token is required.")
        headers = {"Authorization": f"Bearer {token}"}
        url = "http://localhost:8000/api/vectorstore/collections"
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to retrieve vector store collections. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def index_document(token: str, index_request: dict) -> str:
    """
    Index a document directly to the vector store.
    Args:
        token: Bearer token for authentication (from authenticate_user)
        index_request: Dictionary with document index fields (see DocumentIndexRequest)
    Returns:
        JSON string of indexing result or error message.
    """
    try:
        if not token or not index_request:
            raise ValueError("Authentication token and index_request are required.")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = "http://localhost:8000/api/vectorstore/index"
        response = requests.post(url, headers=headers, json=index_request, timeout=30)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to index document. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def get_available_chat_models(token: str) -> dict:
    """
    List available chat models (Groq LLMs) for the chat interface.
    Args:
        token: Bearer token for authentication (from authenticate_user)
    Returns:
        List of chat model objects or empty list if none found.
    """
    try:
        if not token:
            raise ValueError("Authentication token is required.")
        headers = {"Authorization": f"Bearer {token}"}
        url = "http://localhost:8000/api/chat_models"
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            # Parse the JSON response and return it as a Python object
            import json
            return json.loads(response.text)
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to retrieve chat models. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def get_multi_agents(token: str) -> dict:
    """
    List all user agents for multi-agent chat.
    Args:
        token: Bearer token for authentication (from authenticate_user)
    Returns:
        List of agent objects or empty list if none found.
    """
    try:
        if not token:
            raise ValueError("Authentication token is required.")
        headers = {"Authorization": f"Bearer {token}"}
        url = "http://localhost:8000/api/agents"
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            # Parse the JSON response and return it as a Python object
            import json
            return json.loads(response.text)
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to retrieve agents. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def create_agent(token: str, agent_data: dict) -> str:
    """
    Create a new agent for the user.
    Args:
        token: Bearer token for authentication (from authenticate_user)
        agent_data: Dictionary with agent fields (e.g., name, system_message)
    Returns:
        JSON string of created agent info or error message.
    """
    try:
        if not token or not agent_data:
            raise ValueError("Authentication token and agent_data are required.")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = "http://localhost:8000/api/agents"
        response = requests.post(url, headers=headers, json=agent_data, timeout=10)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to create agent. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def update_agent(token: str, agent_id: str, agent_data: dict) -> str:
    """
    Update an existing agent for the user.
    Args:
        token: Bearer token for authentication (from authenticate_user)
        agent_id: The agent's unique ID
        agent_data: Dictionary with updated agent fields
    Returns:
        JSON string of updated agent info or error message.
    """
    try:
        if not token or not agent_id or not agent_data:
            raise ValueError("Authentication token, agent_id, and agent_data are required.")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = f"http://localhost:8000/api/agents/{agent_id}"
        response = requests.put(url, headers=headers, json=agent_data, timeout=10)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        elif response.status_code == 404:
            raise McpError(ErrorData(INVALID_PARAMS, "Agent not found."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to update agent. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def delete_agent(token: str, agent_id: str) -> str:
    """
    Delete an agent for the user.
    Args:
        token: Bearer token for authentication (from authenticate_user)
        agent_id: The agent's unique ID
    Returns:
        JSON string of deletion result or error message.
    """
    try:
        if not token or not agent_id:
            raise ValueError("Authentication token and agent_id are required.")
        headers = {"Authorization": f"Bearer {token}"}
        url = f"http://localhost:8000/api/agents/{agent_id}"
        response = requests.delete(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        elif response.status_code == 404:
            raise McpError(ErrorData(INVALID_PARAMS, "Agent not found."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to delete agent. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def chat_completions(token: str, completion_request: dict) -> str:
    """
    OpenAI-compatible endpoint for chat completions using langchain_groq.ChatGroq.
    Args:
        token: Bearer token for authentication (from authenticate_user)
        completion_request: Dictionary with OpenAI chat completion fields (model, messages, etc.)
    Returns:
        JSON string of chat completion response or error message.
    """
    try:
        if not token or not completion_request:
            raise ValueError("Authentication token and completion_request are required.")
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        url = "http://localhost:8000/api/chat/completions"
        response = requests.post(url, headers=headers, json=completion_request, timeout=30)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to complete chat. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def get_audio_models(token: str) -> str:
    """
    List available Groq audio models (filters for Whisper models).
    Args:
        token: Bearer token for authentication (from authenticate_user)
    Returns:
        JSON string of audio models or error message.
    """
    try:
        if not token:
            raise ValueError("Authentication token is required.")
        headers = {"Authorization": f"Bearer {token}"}
        url = "http://localhost:8000/api/models"
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to retrieve audio models. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def get_audio_tasks(token: str) -> str:
    """
    List available audio tasks (transcribe, translate).
    Args:
        token: Bearer token for authentication (from authenticate_user)
    Returns:
        JSON string of audio tasks or error message.
    """
    try:
        if not token:
            raise ValueError("Authentication token is required.")
        headers = {"Authorization": f"Bearer {token}"}
        url = "http://localhost:8000/api/tasks"
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to retrieve audio tasks. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e

@mcp.tool()
def get_supported_languages(token: str) -> str:
    """
    List supported transcription languages for audio models.
    Args:
        token: Bearer token for authentication (from authenticate_user)
    Returns:
        JSON string of supported languages or error message.
    """
    try:
        if not token:
            raise ValueError("Authentication token is required.")
        headers = {"Authorization": f"Bearer {token}"}
        url = "http://localhost:8000/api/languages"
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 401:
            raise McpError(ErrorData(INVALID_PARAMS, "Invalid or expired token."))
        else:
            raise McpError(
                ErrorData(
                    INTERNAL_ERROR,
                    f"Failed to retrieve supported languages. HTTP status code: {response.status_code}"
                )
            )
    except ValueError as e:
        raise McpError(ErrorData(INVALID_PARAMS, str(e))) from e
    except RequestException as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Request error: {str(e)}")) from e
    except Exception as e:
        raise McpError(ErrorData(INTERNAL_ERROR, f"Unexpected error: {str(e)}")) from e


# Set up the SSE transport for MCP communication.
sse = SseServerTransport("/messages/")

async def handle_sse(request: StarletteRequest) -> None:
    _server = mcp._mcp_server
    async with sse.connect_sse(
        request.scope,
        request.receive,
        request._send,
    ) as (reader, writer):
        await _server.run(reader, writer, _server.create_initialization_options())

# --- FLAME MCP COMPATIBILITY ENDPOINTS ---
from starlette.responses import JSONResponse
from starlette.responses import PlainTextResponse
from starlette.endpoints import HTTPEndpoint

async def health(request):
    return JSONResponse({"status": "ok", "service": "Flame MCP DB Tool Server"})

async def langgraph_health(request):
    return JSONResponse({"status": "ok", "service": "Flame MCP DB Tool Server", "langgraph": True})

async def langgraph_tools(request):
    # Return a list of available MCP tools (names and docstrings)
    tools = []
    tools_list = await mcp.list_tools()
    for tool in tools_list:
        tools.append({
            "name": getattr(tool, 'name', str(tool)),
            "description": getattr(tool, 'description', getattr(tool, '__doc__', ''))
        })
    print(f"Registered tools (from list_tools): {tools}")  # Debug print
    return JSONResponse({"tools": tools})

async def langgraph_sessions(request):
    # Integrate with backend session/auth API
    try:
        if request.method != "POST":
            return JSONResponse({"error": "Method not allowed"}, status_code=405)
        data = await request.json()
        # Extract bearer token from Authorization header
        auth_header = request.headers.get("authorization")
        if not auth_header or not auth_header.lower().startswith("bearer "):
            return JSONResponse({"error": "Missing or invalid Authorization header (Bearer token required)"}, status_code=401)
        token = auth_header.split(" ", 1)[1]
        # Forward user_id and expiry_days if present
        user_id = data.get("user_id")
        expiry_days = data.get("expiry_days", 7)
        if not user_id:
            return JSONResponse({"error": "user_id is required"}, status_code=400)
        backend_url = "http://localhost:8000/api/db/sessions"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {"user_id": user_id, "expiry_days": expiry_days}
        try:
            response = requests.post(backend_url, headers=headers, json=payload, timeout=10)
        except RequestException as e:
            return JSONResponse({"error": f"Backend request error: {str(e)}"}, status_code=502)
        if response.status_code == 200:
            return JSONResponse(response.json())
        elif response.status_code == 401:
            return JSONResponse({"error": "Invalid or expired token."}, status_code=401)
        else:
            return JSONResponse({"error": f"Backend returned status {response.status_code}: {response.text}"}, status_code=502)
    except Exception as e:
        return JSONResponse({"error": f"Unexpected error: {str(e)}"}, status_code=500)

# All @mcp.tool() functions are defined above this point to ensure proper registration.

# DEBUG: Print MCP attributes and registered tools before app creation
print("DEBUG: mcp attributes:", dir(mcp))
print("DEBUG: mcp._tools:", getattr(mcp, "_tools", None))

# Create the Starlette app with all endpoints:
async def langgraph_tool_invoke(request):
    tool_name = request.path_params["tool_name"]
    tools_list = await mcp.list_tools()
    tool_obj = next((t for t in tools_list if getattr(t, 'name', None) == tool_name), None)
    if not tool_obj:
        return JSONResponse({"error": "Tool not found"}, status_code=404)
    data = await request.json()
    try:
        # If the tool is async, await it; else call directly
        if callable(tool_obj):
            if hasattr(tool_obj, "__call__") and hasattr(tool_obj, "__code__") and tool_obj.__code__.co_flags & 0x80:
                # Async function
                result = await tool_obj(**data)
            else:
                result = tool_obj(**data)
        else:
            result = None
        return JSONResponse({"result": result})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

async def langgraph_tools_call(request):
    data = await request.json()
    # Support both formats: {"tool": "...", "args": {...}} and {"tool_name": "...", "args": {...}}
    tool_name = data.get("tool") or data.get("tool_name")
    args = data.get("args", {}) or data.get("tool_args", {})

    if not tool_name:
        return JSONResponse({"error": "Missing tool name in request"}, status_code=400)

    print(f"DEBUG: Tool call received for {tool_name} with args: {args}")

    # Special handling for chat_with_agent to bypass Pydantic model issues
    if tool_name == "chat_with_agent":
        try:
            token = args.get("token")
            chat_request = args.get("chat_request", {})

            if not token or not chat_request:
                return JSONResponse({"error": "Missing token or chat_request"}, status_code=400)

            # Convert chat_request to a proper dictionary
            if hasattr(chat_request, "model_dump"):
                chat_request = chat_request.model_dump()
            elif hasattr(chat_request, "dict"):
                chat_request = chat_request.dict()

            # Use the main chat endpoint
            url = "http://localhost:8000/api/chat/"
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

            print(f"DEBUG: Making direct request to {url} with chat_request: {chat_request}")
            response = requests.post(url, headers=headers, json=chat_request, timeout=30)
            print(f"DEBUG: Response status code: {response.status_code}")

            if response.status_code == 200:
                result = json.loads(response.text)
                formatted_result = {
                    "success": True,
                    "response": result.get("response", ""),
                    "session_id": result.get("session_id", ""),
                    "messages": result.get("messages", []),
                    "metadata": result.get("metadata", {})
                }
                return JSONResponse({"result": formatted_result})
            else:
                return JSONResponse({"error": f"Chat API error: {response.text}"}, status_code=response.status_code)
        except Exception as e:
            print(f"DEBUG: Direct chat_with_agent error: {str(e)}")
            return JSONResponse({"error": f"Error in chat_with_agent: {str(e)}"}, status_code=500)

    # For other tools, use the standard approach
    # Get the actual function object from the MCP server
    tools_list = await mcp.list_tools()
    tool_obj = next((t for t in tools_list if getattr(t, 'name', None) == tool_name), None)

    # If the tool is not callable, try to get the actual function from the module
    if tool_obj and not callable(tool_obj):
        print(f"DEBUG: Tool {tool_name} is not callable, trying to get the actual function")
        import inspect
        import sys
        current_module = sys.modules[__name__]
        for name, obj in inspect.getmembers(current_module):
            if callable(obj) and hasattr(obj, '__name__') and obj.__name__ == tool_name:
                print(f"DEBUG: Found callable function {name} for tool {tool_name}")
                tool_obj = obj
                break

    if not tool_obj:
        print(f"DEBUG: Tool not found: {tool_name}")
        print(f"DEBUG: Available tools: {[getattr(t, 'name', str(t)) for t in tools_list]}")
        return JSONResponse({"error": f"Tool not found: {tool_name}"}, status_code=404)

    # Process args to handle Pydantic models
    # Convert any Pydantic models in args to dictionaries
    args = convert_pydantic_to_dict(args)

    try:
        print(f"DEBUG: Calling tool {tool_name} with args: {args}")
        print(f"DEBUG: Tool object: {tool_obj}")
        if callable(tool_obj):
            if hasattr(tool_obj, "__call__") and hasattr(tool_obj, "__code__") and tool_obj.__code__.co_flags & 0x80:
                # Async function
                print(f"DEBUG: Calling async tool {tool_name}")
                result = await tool_obj(**args)
            else:
                print(f"DEBUG: Calling sync tool {tool_name}")
                result = tool_obj(**args)
        else:
            print(f"DEBUG: Tool {tool_name} is not callable")
            result = None

        print(f"DEBUG: Tool call result type: {type(result)}")
        print(f"DEBUG: Tool call result: {result[:100] if isinstance(result, str) else result}")
        # Convert result to JSON-serializable format if needed
        result = convert_pydantic_to_dict(result)
        print(f"DEBUG: Returning result: {result}")
        return JSONResponse({"result": result})
    except Exception as e:
        print(f"DEBUG: Tool call error: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)

app = Starlette(
    debug=True,
    routes=[
        Route("/health", endpoint=health),
        Route("/langgraph/health", endpoint=langgraph_health),
        Route("/langgraph/tools", endpoint=langgraph_tools),
        Route("/langgraph/sessions", endpoint=langgraph_sessions, methods=["POST"]),
        Route("/langgraph/tool/{tool_name}", endpoint=langgraph_tool_invoke, methods=["POST"]),
        Route("/langgraph/tools/call", endpoint=langgraph_tools_call, methods=["POST"]),
        Route("/sse", endpoint=handle_sse),
        Mount("/messages/", app=sse.handle_post_message),
    ],
)

print("DEBUG: Starlette routes:", [route.path for route in app.routes])



if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)
