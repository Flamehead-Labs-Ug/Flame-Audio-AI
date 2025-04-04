import os
import logging
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from authentication.auth import get_current_user, User
from .pg_connector import get_pg_db
from embedding import list_embedding_models
from .vector_store_langchain import get_vector_store, FlameVectorStore

# Set up logging
logger = logging.getLogger(__name__)

# Initialize router
router = APIRouter(prefix="/vectorstore", tags=["vectorstore"])

# Models for request/response
class VectorStoreSettings(BaseModel):
    embedding_model: str = "all-MiniLM-L6-v2"  # Default embedding model
    dimension: int = 384  # Default dimension for embeddings
    similarity_threshold: float = 0.2  # Default similarity threshold
    chunk_size: int = 1000  # Default chunk size for text
    chunk_overlap: int = 200  # Default chunk overlap
    match_count: int = 10  # Default number of results to return
    enabled: bool = True  # Whether vector search is enabled
    metadata: Dict[str, Any] = Field(default_factory=dict)

class VectorStoreResponse(BaseModel):
    settings: VectorStoreSettings
    available_models: List[Dict[str, Any]]

# Get vector store settings for a user
@router.get("/settings")
async def get_vector_store_settings(current_user: User = Depends(get_current_user)) -> VectorStoreResponse:
    """Get vector store settings for the current user"""
    try:
        # Connect to database
        db = get_pg_db()
        
        # Query user settings
        query = """
        SELECT 
            settings 
        FROM 
            user_data.vector_store_settings 
        WHERE 
            user_id = %(user_id)s
        """
        
        result = db.execute_query(query, {"user_id": current_user.id})
        
        # Default settings
        settings = VectorStoreSettings()
        
        # If user has settings, use those
        if result and len(result) > 0 and result[0].get("settings"):
            user_settings = result[0]["settings"]
            settings = VectorStoreSettings(**user_settings)
        
        # Get available embedding models
        available_models = list_embedding_models()
        
        return VectorStoreResponse(
            settings=settings,
            available_models=available_models
        )
    except Exception as e:
        logger.error(f"Error getting vector store settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get vector store settings: {str(e)}"
        )

# Save vector store settings for a user
@router.post("/settings")
async def save_vector_store_settings(settings: VectorStoreSettings, current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """Save vector store settings for the current user"""
    try:
        # Connect to database
        db = get_pg_db()
        
        # Upsert settings
        query = """
        INSERT INTO user_data.vector_store_settings
            (user_id, settings)
        VALUES
            (%(user_id)s, %(settings)s::jsonb)
        ON CONFLICT (user_id) 
        DO UPDATE SET 
            settings = %(settings)s::jsonb,
            updated_at = NOW()
        RETURNING id
        """
        
        result = db.execute_query(
            query, 
            {
                "user_id": current_user.id,
                "settings": settings.model_dump()
            }
        )
        
        if not result or len(result) == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save vector store settings"
            )
        
        return {
            "id": str(result[0]["id"]) if result and len(result) > 0 else None,
            "message": "Vector store settings saved successfully"
        }
    except Exception as e:
        logger.error(f"Error saving vector store settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save vector store settings: {str(e)}"
        )

# Get available embedding models
@router.get("/models")
async def get_embedding_models(current_user: Optional[User] = Depends(get_current_user)) -> List[Dict[str, Any]]:
    """Get list of available embedding models"""
    try:
        # Get available embedding models
        available_models = list_embedding_models()
        return available_models
    except Exception as e:
        logger.error(f"Error getting embedding models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get embedding models: {str(e)}"
        )
