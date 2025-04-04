"""API routes for vector store operations"""

import os
import logging
from typing import Dict, List, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from authentication.auth import get_current_user, User
from .pg_connector import get_pg_db
from embedding import list_embedding_models
from .vector_store import get_vector_store

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


# Response models
class VectorStoreSettings(BaseModel):
    """Vector store settings model"""
    embedding_model: str = Field(..., description="HuggingFace embedding model to use")
    similarity_threshold: float = Field(..., description="Similarity threshold for search")
    chunk_size: int = Field(..., description="Size of text chunks")
    chunk_overlap: int = Field(..., description="Overlap between text chunks")
    match_count: int = Field(..., description="Number of matches to return")
    enabled: bool = Field(..., description="Whether vector store is enabled")


class EmbeddingModel(BaseModel):
    """Embedding model information"""
    name: str
    dimensions: int
    language: str = "en"  # Default to English


@router.get("/vectorstore/settings", response_model=VectorStoreSettings)
async def get_vectorstore_settings(current_user: User = Depends(get_current_user)):
    """Get current vector store settings for the user"""
    try:
        # Initialize vector store to load settings
        vector_store = get_vector_store()
        
        # Get settings, or use defaults if none are available
        settings = vector_store.settings if hasattr(vector_store, 'settings') else {}
        
        # Use default values if settings are None or keys don't exist
        default_settings = {
            "embedding_model": "all-MiniLM-L6-v2",
            "similarity_threshold": 0.2,
            "chunk_size": 1500,
            "chunk_overlap": 200,
            "match_count": 10,
            "enabled": True
        }
        
        # Merge with defaults
        for key, default_value in default_settings.items():
            if settings is None or key not in settings:
                if settings is None:
                    settings = {}
                settings[key] = default_value
        
        # Return settings
        return VectorStoreSettings(
            embedding_model=settings.get("embedding_model", default_settings["embedding_model"]),
            similarity_threshold=settings.get("similarity_threshold", default_settings["similarity_threshold"]),
            chunk_size=settings.get("chunk_size", default_settings["chunk_size"]),
            chunk_overlap=settings.get("chunk_overlap", default_settings["chunk_overlap"]),
            match_count=settings.get("match_count", default_settings["match_count"]),
            enabled=settings.get("enabled", default_settings["enabled"])
        )
    except Exception as e:
        logger.error(f"Error getting vector store settings: {e}")
        
        # Return default settings on error
        default_settings = {
            "embedding_model": "all-MiniLM-L6-v2",
            "similarity_threshold": 0.2,
            "chunk_size": 1500,
            "chunk_overlap": 200,
            "match_count": 10,
            "enabled": True
        }
        
        return VectorStoreSettings(**default_settings)


@router.post("/vectorstore/settings", response_model=VectorStoreSettings)
async def save_vectorstore_settings(settings: VectorStoreSettings, current_user: User = Depends(get_current_user)):
    """Save vector store settings for the user"""
    try:
        # Connect to database
        db = get_pg_db()
        
        # Convert settings to dict
        settings_dict = settings.dict()
        
        # Save settings to database
        query = """
        INSERT INTO user_data.vector_store_settings 
            (user_id, settings) 
        VALUES 
            (%s, %s)
        ON CONFLICT (user_id) DO UPDATE 
        SET 
            settings = EXCLUDED.settings,
            updated_at = NOW()
        RETURNING id
        """
        
        result = db.execute_query(
            query, 
            (current_user.id, settings_dict)
        )
        
        if not result or len(result) == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save vector store settings"
            )
            
        # Reinitialize vector store with new settings
        vector_store = get_vector_store(embedding_model=settings.embedding_model)
        
        # Return saved settings
        return settings
    except Exception as e:
        logger.error(f"Error saving vector store settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving vector store settings: {str(e)}"
        )


@router.get("/vectorstore/models", response_model=List[EmbeddingModel])
async def get_embedding_models():
    """Get list of available embedding models"""
    try:
        models = list_embedding_models()
        return models
    except Exception as e:
        logger.error(f"Error listing embedding models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing embedding models: {str(e)}"
        )


@router.get("/vectorstore/collections", response_model=Dict[str, Any])
async def get_vector_store_collections(current_user: User = Depends(get_current_user)):
    """Get information about all vector store collections"""
    try:
        # Get vector store instance
        vector_store = get_vector_store()
        
        # Get collections info
        collections_info = vector_store.get_collections_info()
        
        return collections_info
    except Exception as e:
        logger.error(f"Error getting vector store collections: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Error getting vector store collections: {str(e)}"
        )


class DocumentIndexRequest(BaseModel):
    """Request model for indexing a document"""
    document_name: str
    document_content: Dict[str, Any]
    user_id: str
    agent_id: Optional[str] = None
    vector_store_settings: Dict[str, Any]


class DocumentIndexResponse(BaseModel):
    """Response model for indexed document"""
    indexed_count: int
    collection_name: str
    points_count: int
    indexed_vectors_count: int
    document_name: str


@router.post("/vectorstore/index", response_model=DocumentIndexResponse)
async def index_document(request: DocumentIndexRequest, current_user: User = Depends(get_current_user)):
    """Index a document directly to the vector store"""
    try:
        # Validate user permissions
        if current_user.id != request.user_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to index documents for this user"
            )

        # Extract vector store settings
        embedding_model = request.vector_store_settings.get("embedding_model", "all-MiniLM-L6-v2")
        enabled = request.vector_store_settings.get("enabled", True)
        
        # If vector search is disabled, return early with empty results
        if not enabled:
            return DocumentIndexResponse(
                indexed_count=0,
                collection_name="transcription_embeddings",
                points_count=0,
                indexed_vectors_count=0,
                document_name=request.document_name
            )
        
        # Initialize vector store with requested embedding model
        vector_store = get_vector_store(embedding_model=embedding_model)
        
        # Process document content - extract segments
        segments = request.document_content.get("segments", [])
        
        # Index each segment
        indexed_count = 0
        for i, segment in enumerate(segments):
            # Extract segment text and metadata
            text = segment.get("text", "")
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            
            # Create metadata
            metadata = {
                "start_time": start,
                "end_time": end,
                "segment_index": i,
                "agent_id": request.agent_id
            }
            
            # Store in vector store if text is not empty
            if text.strip():
                vector_store.store_embedding(
                    user_id=request.user_id,
                    document_name=request.document_name,
                    chunk_index=i,
                    content=text,
                    metadata=metadata,
                    embedding_model=embedding_model
                )
                indexed_count += 1
        
        # Get collection information to return stats
        collections_info = vector_store.get_collections_info()
        primary_collection = collections_info.get("primary_collection", "transcription_embeddings")
        
        # Find collection details
        collection_details = {}
        for collection in collections_info.get("collections", []):
            if collection.get("name") == primary_collection:
                collection_details = collection
                break
        
        # Return indexing results
        return DocumentIndexResponse(
            indexed_count=indexed_count,
            collection_name=primary_collection,
            points_count=collection_details.get("points_count", 0),
            indexed_vectors_count=collection_details.get("indexed_vectors_count", 0),
            document_name=request.document_name
        )
    except Exception as e:
        logger.error(f"Error indexing document: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
            detail=f"Error indexing document: {str(e)}"
        )
