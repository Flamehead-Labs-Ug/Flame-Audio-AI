from datetime import datetime
import logging
import json
import uuid
from uuid import UUID
from typing import Dict, List, Any, Optional

# Set up logger
logger = logging.getLogger(__name__)

from fastapi import APIRouter, HTTPException, status, Depends, BackgroundTasks
from .models import (
    SessionCreate, Session,
    AudioJobCreate, AudioJob, 
    TranscriptionCreate, Transcription,
    TranslationCreate, Translation,
    SearchQuery, SearchResult,
    DocumentData
)
from .pg_connector import get_pg_db
from .vector_store import get_vector_store, sanitize_collection_name
from authentication.auth import get_current_user, User
from embedding import generate_embedding

router = APIRouter(prefix="/db", tags=["database"])

# Session Routes
@router.post("/sessions", response_model=Dict[str, Any])
async def create_session(session_data: SessionCreate, current_user = Depends(get_current_user)):
    """Create a new user session"""
    # Verify the user is creating their own session
    if session_data.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Can only create sessions for your own user account"
        )
    
    db = get_pg_db()
    try:
        result = db.call_function(
            "user_data.create_session", 
            {
                "p_user_id": session_data.user_id,
                "p_expires_interval": f"{session_data.expiry_days} days"
            }
        )
        return {"session_id": result.get("create_session")}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create session: {str(e)}"
        )

# Audio Job Routes
@router.post("/jobs", response_model=Dict[str, Any])
async def create_audio_job(job_data: dict, current_user = Depends(get_current_user)):
    """Create a new audio job"""
    try:
        db = get_pg_db()
        
        # Get the SQL query for creating an audio job
        query = """
        SELECT user_data.create_audio_job(
            %(user_id)s, -- user_id
            %(session_id)s, -- session_id
            %(agent_id)s, -- agent_id
            %(filename)s, -- file_name
            %(task_type)s, -- task_type
            %(original_language)s, -- original_language
            %(file_type)s, -- file_type
            %(file_size_bytes)s, -- file_size_bytes
            %(duration_seconds)s, -- duration_seconds
            %(settings)s -- settings
        ) as id
        """
        
        # Get values from request
        user_id = current_user.id
        session_id = job_data.get("session_id")  # Can be null
        agent_id = job_data.get("agent_id")  # Can be null
        filename = job_data.get("filename")
        task_type = job_data.get("task_type")
        original_language = job_data.get("original_language")
        file_type = job_data.get("file_type", "mp3")
        file_size_bytes = job_data.get("file_size_bytes", 0)
        duration_seconds = job_data.get("duration_seconds", 0.0)
        
        # Convert settings dict to JSON string format
        settings_json = json.dumps(job_data.get("settings", {}))
        
        # Execute the query
        result = db.execute_query(
            query, 
            {
                "user_id": user_id,
                "session_id": session_id,
                "agent_id": agent_id,
                "filename": filename,
                "task_type": task_type,
                "original_language": original_language,
                "file_type": file_type,
                "file_size_bytes": file_size_bytes,
                "duration_seconds": duration_seconds,
                "settings": settings_json
            }
        )
        
        return {"id": result[0]["id"]}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create audio job: {str(e)}"
        )

@router.get("/jobs", response_model=List[Dict[str, Any]])
async def get_user_jobs(current_user = Depends(get_current_user)):
    """Get all audio jobs for the current user"""
    db = get_pg_db()
    try:
        query = """
        SELECT * FROM user_data.audio_jobs
        WHERE user_id = %(user_id)s
        ORDER BY created_at DESC
        """
        results = db.execute_query(query, {"user_id": current_user.id})
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve jobs: {str(e)}"
        )

# Transcription Routes
@router.post("/transcriptions", response_model=Dict[str, Any])
async def save_transcription(transcription_data: TranscriptionCreate, current_user = Depends(get_current_user)):
    """Save a new transcription segment"""
    try:
        # Connect to the database
        db = get_pg_db()
        
        # Validate job ownership
        job_id = transcription_data.job_id
        if not is_job_owner(db, job_id, current_user.id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to add transcriptions to this job"
            )
        
        # Use explicit type casting in a parameterized query
        query = """
        INSERT INTO user_data.transcriptions
            (job_id, chunk_index, start_time, end_time, text, language)
        VALUES
            (%(job_id)s::uuid, %(chunk_index)s, %(start_time)s, %(end_time)s, %(text)s, %(language)s)
        RETURNING id as transcription_id
        """
        
        # Execute the query with validated parameters
        result = db.execute_query(
            query, 
            {
                "job_id": str(job_id),
                "chunk_index": transcription_data.chunk_index,
                "start_time": transcription_data.start_time,
                "end_time": transcription_data.end_time,
                "text": transcription_data.text,
                "language": transcription_data.language or "auto"
            }
        )
        
        # Check if we got a valid result
        if not result or len(result) == 0 or "transcription_id" not in result[0]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save transcription"
            )
        
        # Save the transcription ID for return
        transcription_id = result[0]["transcription_id"]
        
        # Now attempt to generate and store embedding for vector search
        try:
            # Generate and store embedding for vector search
            vector_store = get_vector_store()
            embedding = generate_embedding(transcription_data.text)
            
            metadata = {
                "start_time": transcription_data.start_time,
                "end_time": transcription_data.end_time,
                "language": transcription_data.language or "auto",
                "chunk_index": transcription_data.chunk_index
            }
            
            vector_store.store_embedding(
                user_id=current_user.id,
                job_id=job_id,
                chunk_index=transcription_data.chunk_index,
                content=transcription_data.text,
                embedding=embedding,
                metadata=metadata
            )
            vector_saved = True
        except Exception as ve:
            # Log error but don't fail the request if vector storage fails
            logger.warning(f"Warning: Failed to store vector embedding: {str(ve)}")
            vector_saved = False
        
        # Return success response with status information
        return {
            "id": transcription_id,
            "vector_embedding_saved": vector_saved
        }
    except Exception as e:
        logger.error(f"Error saving transcription: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save transcription: {str(e)}"
        )

@router.get("/transcriptions/{job_id}", response_model=List[Dict[str, Any]])
async def get_job_transcriptions(job_id: str, current_user = Depends(get_current_user)):
    """Get all transcriptions for a specific job"""
    # Verify user owns the job
    db = get_pg_db()
    job_query = """
    SELECT user_id FROM user_data.audio_jobs
    WHERE id = %(job_id)s
    """
    job_result = db.execute_query(job_query, {"job_id": job_id})
    
    if not job_result or job_result[0].get("user_id") != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Can only view transcriptions for your own jobs"
        )
    
    try:
        query = """
        SELECT * FROM user_data.transcriptions
        WHERE job_id = %(job_id)s
        ORDER BY chunk_index ASC
        """
        results = db.execute_query(query, {"job_id": job_id})
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve transcriptions: {str(e)}"
        )

# Translation Routes
@router.post("/translations", response_model=Dict[str, Any])
async def save_translation(translation_data: dict, current_user = Depends(get_current_user)):
    """Save a translation"""
    # Verify user owns the job
    db = get_pg_db()
    job_query = """
    SELECT user_id FROM user_data.audio_jobs
    WHERE id = %(job_id)s
    """
    job_result = db.execute_query(job_query, {"job_id": translation_data.get("job_id")})
    
    if not job_result or job_result[0].get("user_id") != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Can only add translations to your own jobs"
        )
    
    try:
        # Use explicit type casting
        query = """
        SELECT * FROM user_data.save_translation(
            %(job_id)s::uuid,               -- p_job_id
            %(transcription_id)s::uuid,      -- p_transcription_id
            %(chunk_index)s::integer,        -- p_chunk_index
            %(start_time)s::float,           -- p_start_time
            %(end_time)s::float,             -- p_end_time
            %(original_text)s::text,         -- p_original_text
            %(translated_text)s::text,       -- p_translated_text
            %(source_language)s::text,       -- p_source_language
            %(target_language)s::text        -- p_target_language
        ) as id
        """
        
        # Set default languages if not provided
        if "source_language" not in translation_data:
            translation_data["source_language"] = translation_data.get("original_language", "auto")
        if "target_language" not in translation_data:
            translation_data["target_language"] = "en"
        
        # Execute the query
        result = db.execute_query(
            query,
            {
                "job_id": translation_data.get("job_id"),
                "transcription_id": translation_data.get("transcription_id"),
                "chunk_index": translation_data.get("chunk_index", 0),
                "start_time": translation_data.get("start_time", 0.0),
                "end_time": translation_data.get("end_time", 0.0),
                "original_text": translation_data.get("original_text", ""),
                "translated_text": translation_data.get("translated_text", ""),
                "source_language": translation_data.get("source_language", "auto"),
                "target_language": translation_data.get("target_language", "en")
            }
        )
        
        return {"id": result[0]["save_translation"]}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save translation: {str(e)}"
        )

@router.get("/translations/{job_id}", response_model=List[Dict[str, Any]])
async def get_job_translations(job_id: str, current_user = Depends(get_current_user)):
    """Get all translations for a specific job"""
    # Verify user owns the job
    db = get_pg_db()
    job_query = """
    SELECT user_id FROM user_data.audio_jobs
    WHERE id = %(job_id)s
    """
    job_result = db.execute_query(job_query, {"job_id": job_id})
    
    if not job_result or job_result[0].get("user_id") != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Can only view translations for your own jobs"
        )
    
    try:
        query = """
        SELECT * FROM user_data.translations
        WHERE job_id = %(job_id)s
        ORDER BY chunk_index ASC
        """
        results = db.execute_query(query, {"job_id": job_id})
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve translations: {str(e)}"
        )

# Vector Search Routes
@router.post("/vectors", response_model=Dict[str, Any])
async def store_vector(vector_data: dict, current_user = Depends(get_current_user)):
    """Store an embedding vector for a transcription chunk"""
    # Verify the user is storing their own data
    if vector_data.get("user_id") != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Can only store vectors for your own transcriptions"
        )
    
    try:
        vector_store = get_vector_store()
        result = vector_store.store_embedding(
            user_id=vector_data.get("user_id"),
            job_id=vector_data.get("job_id"),
            chunk_index=vector_data.get("chunk_index"),
            content=vector_data.get("content"),
            embedding=vector_data.get("embedding"),
            metadata=vector_store.get("metadata")
        )
        
        # Handle the new result format which might be a status object
        if isinstance(result, list) and len(result) > 0 and "id" in result[0]:
            # Original format with id in first item of list
            return {"vector_id": result[0]["id"], "status": "success"}
        elif isinstance(result, dict):
            if "id" in result:
                # Direct object with id
                return {"vector_id": result["id"], "status": "success"}
            elif "status" in result:
                # Status object from our fallback handling
                return {"status": result["status"], "message": result.get("message", "")}
            
        # Return a default response if no recognizable format
        return {"status": "unknown", "message": "Vector storage completed but with unknown result format"}
    except Exception as e:
        # Don't raise an exception, just return the error info
        return {"status": "error", "message": str(e)}

@router.post("/search", response_model=List[Dict[str, Any]])
async def search_transcriptions(search_data: SearchQuery, current_user = Depends(get_current_user)):
    """Search for similar transcriptions using vector similarity"""
    try:
        # Generate embedding for search query
        embedding = generate_embedding(search_data.query)
        
        # Search for similar content
        vector_store = get_vector_store()
        results = vector_store.search_similar(
            query_embedding=embedding,
            match_count=search_data.match_count,
            similarity_threshold=search_data.similarity_threshold,
            filter_metadata=search_data.filter_metadata
        )
        
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search transcriptions: {str(e)}"
        )

# Consolidate document saving into a single endpoint
@router.post("/save_document", response_model=Dict[str, Any])
async def save_document(document: DocumentData, background_tasks: BackgroundTasks, current_user: User = Depends(get_current_user)):
    """Save a document to the database with backend processing for segments and embeddings"""
    # Validate document and get parameters
    job_id = str(uuid.uuid4())
    document_name = document.filename
    
    # Sanitize document name for Qdrant collection
    qdrant_collection_name = sanitize_collection_name(document_name)
    
    try:
        # Connect to the database
        db = get_pg_db()
        
        # Check if a document with the same name already exists for this user
        check_query = """
        SELECT id FROM user_data.audio_jobs 
        WHERE user_id = %s AND file_name = %s
        """
        
        existing_doc = db.execute_query(check_query, (current_user.id, document_name))
        
        # If document exists, use its ID instead of creating a new record
        if existing_doc and len(existing_doc) > 0:
            job_id = str(existing_doc[0]["id"])
            logger.info(f"Found existing document with name '{document_name}'. Updating existing record with ID {job_id}")
            
            # Update the existing record with new settings
            update_query = """
            UPDATE user_data.audio_jobs 
            SET task_type = %s, original_language = %s, settings = %s, agent_id = %s
            WHERE id = %s
            """
            
            # Insert the job record first
            settings = {
                "task_type": document.task_type,
                "model": document.model,
                "original_language": document.original_language,
            }
            
            # Add description if present
            if hasattr(document, 'description') and document.description:
                settings["description"] = document.description
            
            # Get agent_id from the document or use default if not present
            agent_id = None
            if hasattr(document, 'agent_id') and document.agent_id:
                # Handle the special case where agent_id is 'all'
                if document.agent_id == 'all':
                    agent_id = None
                else:
                    agent_id = document.agent_id
            
            # Execute update query
            db.execute_query(
                update_query,
                (document.task_type, document.original_language, json.dumps(settings), agent_id, job_id),
                fetch=False
            )
        else:
            # Insert the job record first
            settings = {
                "task_type": document.task_type,
                "model": document.model,
                "original_language": document.original_language,
            }
            
            # Add description if present
            if hasattr(document, 'description') and document.description:
                settings["description"] = document.description
            
            # Get agent_id from the document or use default if not present
            agent_id = None
            if hasattr(document, 'agent_id') and document.agent_id:
                # Handle the special case where agent_id is 'all'
                if document.agent_id == 'all':
                    agent_id = None
                else:
                    agent_id = document.agent_id
            
            # Insert job with qdrant collection name and agent_id
            query = """
            INSERT INTO user_data.audio_jobs 
                (id, user_id, file_name, task_type, original_language, settings, qdrant_collection_name, agent_id, file_type)
            VALUES 
                (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """
            
            # Get file_type from document, using audio as default if not provided
            file_type = getattr(document, 'file_type', 'audio')
            
            result = db.execute_query(
                query, 
                (job_id, current_user.id, document_name, document.task_type, document.original_language, json.dumps(settings), qdrant_collection_name, agent_id, file_type)
            )
        
        # Define the background processing function to handle segments and embeddings
        def process_document_segments(job_id, segments, user_id, original_language, embedding_settings, agent_id=None):
            try:
                # Get database connection in this thread
                process_db = get_pg_db()
                
                # Get the document name and Qdrant collection from the database
                job_query = """
                SELECT file_name, qdrant_collection_name, agent_id
                FROM user_data.audio_jobs 
                WHERE id = %s AND user_id = %s
                """
                job_result = process_db.execute_query(job_query, (job_id, user_id))
                
                if not job_result:
                    logger.error(f"Failed to find job record for processing: {job_id}")
                    return
                    
                document_name = job_result[0]["file_name"]
                qdrant_collection = job_result[0]["qdrant_collection_name"]
                
                # Use the agent_id from the database if it wasn't explicitly provided
                if agent_id is None and "agent_id" in job_result[0] and job_result[0]["agent_id"]:
                    agent_id = job_result[0]["agent_id"]
                    logger.info(f"Using agent_id from database: {agent_id}")
                
                # Get vector store instance
                vector_store = get_vector_store()
                
                # Set up metadata for all segments
                base_metadata = {
                    "job_id": job_id,
                    "language": original_language,
                    "user_id": user_id
                }
                
                # Use the embedding model specified in settings
                embedding_model = embedding_settings.get("model", "all-MiniLM-L6-v2")
                chunk_size = embedding_settings.get("chunk_size", 1000)
                chunk_overlap = embedding_settings.get("chunk_overlap", 200)
                
                logger.info(f"Using embedding model: {embedding_model}, chunk size: {chunk_size}, overlap: {chunk_overlap}")
                
                # Counter for completed segments
                completed_segments = 0
                total_segments = len(segments)
                
                # Insert each segment and create embeddings
                for index, segment in enumerate(segments):
                    try:
                        # Extract segment data
                        segment_text = segment.get("text", "")
                        start_time = segment.get("start", 0)
                        end_time = segment.get("end", 0)
                        
                        # Skip empty segments
                        if not segment_text.strip():
                            continue
                        
                        # Create segment metadata
                        segment_metadata = base_metadata.copy()
                        segment_metadata.update({
                            "start_time": start_time,
                            "end_time": end_time,
                            "chunk_index": index,
                            "embedding_model": embedding_model,
                            "agent_id": agent_id
                        })
                        
                        # Store embeddings with document name instead of job_id
                        result = vector_store.store_embedding(
                            user_id=user_id,
                            document_name=document_name,
                            chunk_index=index,
                            content=segment_text,
                            embedding=None,  # Auto-generate
                            metadata=segment_metadata,
                            embedding_model=embedding_model
                        )
                        
                        # Insert the transcription segment
                        segment_query = """
                        INSERT INTO user_data.transcriptions 
                            (job_id, chunk_index, start_time, end_time, text, user_id)
                        VALUES 
                            (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (job_id, chunk_index)
                        DO UPDATE SET text = EXCLUDED.text
                        """
                        
                        process_db.execute_query(
                            segment_query,
                            (job_id, index, start_time, end_time, segment_text, user_id)
                        )
                        
                        # Update progress
                        completed_segments += 1
                        
                    except Exception as segment_error:
                        logger.error(f"Error processing segment {index}: {str(segment_error)}")
                
                # Update job status to completed
                status = "completed" if completed_segments == total_segments else "partial"
                completion_query = """
                UPDATE user_data.audio_jobs 
                SET settings = jsonb_set(settings, '{status}', %s) 
                WHERE id = %s
                """
                
                process_db.execute_query(completion_query, (f'"{status}"', job_id))
                logger.info(f"Document processing completed with status: {status}")
                
            except Exception as process_error:
                logger.error(f"Error in process_document_segments: {str(process_error)}")
                # Try to update job status to failed
                try:
                    fail_db = get_pg_db()
                    fail_db.execute_query(
                        "UPDATE user_data.audio_jobs SET settings = jsonb_set(settings, '{status}', '\"failed\"') WHERE id = %s",
                        (job_id,)
                    )
                except Exception as update_error:
                    logger.error(f"Failed to update job status: {str(update_error)}")
        
        # Add the segment processing to background tasks
        background_tasks.add_task(
            process_document_segments, 
            job_id, 
            document.segments, 
            current_user.id, 
            document.original_language,
            document.embedding_settings,
            agent_id
        )
        
        # Return immediate response with job ID
        return {
            "success": True,
            "message": "Document processing started",
            "job_id": job_id,
            "total_segments": len(document.segments),
            "status": "processing"
        }
    
    except Exception as e:
        logger.error(f"Error in save_document: {str(e)}")
        if hasattr(e, "status_code") and hasattr(e, "detail"):
            # Re-raise HTTPExceptions
            raise
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save document: {str(e)}"
            )

@router.get("/document_status/{job_id}", response_model=Dict[str, Any])
async def get_document_status(job_id: str, current_user = Depends(get_current_user)):
    """Get the status of a document processing job"""
    try:
        db = get_pg_db()
        query = """
        SELECT 
            aj.id, 
            aj.file_name, 
            aj.settings,
            aj.created_at,
            (SELECT COUNT(*) FROM user_data.transcriptions WHERE job_id = aj.id) as transcription_count
        FROM 
            user_data.audio_jobs aj
        WHERE 
            aj.id = %s AND
            aj.user_id = %s
        """
        
        try:
            results = db.execute_query(query, (job_id, current_user.id))
            
            if not results or len(results) == 0:
                # Return a simple status response if job not found
                return {
                    "id": job_id,
                    "status": "processing",
                    "progress": 0,
                    "error": None
                }
        except Exception as query_error:
            logger.warning(f"Error retrieving job: {str(query_error)}")
            # Return a simple status response if query fails
            return {
                "id": job_id,
                "status": "processing",
                "progress": 0,
                "error": None
            }
        
        job_info = results[0]
        settings = job_info.get("settings", {})
        
        # Parse settings if it's a string
        if isinstance(settings, str):
            try:
                settings = json.loads(settings)
            except:
                settings = {}
        
        # Extract status information from settings
        job_status = settings.get("status", "processing")
        progress = settings.get("progress", 0)
        error = settings.get("error")
        
        # Get vector store information if the job is complete
        collection_info = None
        if job_status == "completed" or job_status == "complete":
            try:
                # Get the vector store
                vector_store = get_vector_store()
                
                # Only proceed if vector store client is properly initialized
                if vector_store and hasattr(vector_store, 'client') and vector_store.client:
                    # Check if collection exists
                    collection_name = f"job_{job_id}"
                    collections = vector_store.client.get_collections().collections
                    collection_names = [c.name for c in collections]
                    
                    if collection_name in collection_names:
                        # Get collection info
                        collection_details = vector_store.client.get_collection(collection_name=collection_name)
                        
                        # Count points in collection
                        count_result = vector_store.client.count(collection_name=collection_name)
                        
                        collection_info = {
                            "name": collection_name,
                            "vector_size": collection_details.config.params.vectors.size,
                            "vector_distance": collection_details.config.params.vectors.distance,
                            "points_count": count_result.count
                        }
                else:
                    logger.warning("Vector store client not properly initialized, skipping collection info")
            except Exception as e:
                logger.warning(f"Error getting collection info: {e}")
        
        # Create response
        response = {
            "id": job_info.get("id"),
            "status": job_status,
            "created_at": job_info.get("created_at"),
            "completed_at": settings.get("completed_at"),
            "error": error,
            "progress": progress,
            "total_steps": settings.get("total_steps"),
            "current_step": settings.get("current_step"),
            "metadata": settings,
            "transcription_count": job_info.get("transcription_count") or 0,
            "vector_store": collection_info
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting document status: {str(e)}"
        )

# Agent management endpoints
@router.get("/agents", response_model=List[Dict[str, Any]])
async def get_user_agents(current_user = Depends(get_current_user)):
    """Get all agents created by the current user"""
    try:
        db = get_pg_db()
        query = """
        SELECT 
            id, 
            name,
            system_message,
            settings,
            created_at,
            updated_at
        FROM 
            user_data.user_agents 
        WHERE 
            user_id = %(user_id)s
        ORDER BY 
            updated_at DESC
        """
        
        # Debug info
        print(f"Current user ID: {current_user.id}")
        
        # Use named parameters
        result = db.execute_query(query, {"user_id": str(current_user.id)})
        
        # Convert the raw database results to properly formatted dictionaries
        formatted_result = []
        for agent in result:
            formatted_agent = {
                "id": str(agent["id"]) if agent.get("id") else None,
                "name": agent.get("name", ""),
                "system_message": agent.get("system_message", ""),
                "settings": agent.get("settings", {}),
                "created_at": agent.get("created_at", None),
                "updated_at": agent.get("updated_at", None)
            }
            formatted_result.append(formatted_agent)
            
        return formatted_result
    except Exception as e:
        # Print debug information
        print(f"Error in get_user_agents: {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            traceback.print_tb(e.__traceback__)
            
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve agents: {str(e)}"
        )

@router.get("/agents/{agent_id}", response_model=Dict[str, Any])
async def get_agent(agent_id: str, current_user = Depends(get_current_user)):
    """Get a specific agent by ID"""
    try:
        db = get_pg_db()
        query = """
        SELECT 
            id, 
            name,
            system_message,
            settings,
            created_at,
            updated_at
        FROM 
            user_data.user_agents 
        WHERE 
            id = %(agent_id)s AND user_id = %(user_id)s
        """
        
        # Use named parameters
        result = db.execute_query(query, {"agent_id": str(agent_id), "user_id": str(current_user.id)})
        if not result or len(result) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found"
            )
        
        return result[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve agent: {str(e)}"
        )

@router.post("/agents", response_model=Dict[str, Any])
async def create_or_update_agent(agent_data: dict, current_user = Depends(get_current_user)):
    """Create a new agent or update an existing one"""
    try:
        db = get_pg_db()
        
        # Check if agent with this name already exists for this user
        if not agent_data.get("id"):
            check_query = """
            SELECT id FROM user_data.user_agents 
            WHERE user_id = %(user_id)s AND name = %(name)s
            """
            check_result = db.execute_query(check_query, {"user_id": str(current_user.id), "name": agent_data.get("name")})
            
            if check_result and len(check_result) > 0:
                # Agent exists, update it
                agent_id = check_result[0]["id"]
                update_query = """
                UPDATE user_data.user_agents 
                SET 
                    system_message = %(system_message)s,
                    settings = %(settings)s
                WHERE id = %(agent_id)s AND user_id = %(user_id)s
                RETURNING id, name, system_message, settings, created_at, updated_at
                """
                
                params = {
                    "system_message": agent_data.get("system_message"),
                    "settings": json.dumps(agent_data.get("settings", {})),
                    "agent_id": str(agent_id),
                    "user_id": str(current_user.id)
                }
                
                result = db.execute_query(update_query, params)
                return {"id": agent_id, "action": "updated", "agent": result[0]}
            else:
                # Create new agent
                insert_query = """
                INSERT INTO user_data.user_agents 
                    (user_id, name, system_message, settings) 
                VALUES 
                    (%(user_id)s, %(name)s, %(system_message)s, %(settings)s)
                RETURNING id, name, system_message, settings, created_at, updated_at
                """
                
                params = {
                    "user_id": str(current_user.id),
                    "name": agent_data.get("name"),
                    "system_message": agent_data.get("system_message"),
                    "settings": json.dumps(agent_data.get("settings", {}))
                }
                
                result = db.execute_query(insert_query, params)
                return {"id": result[0]["id"], "action": "created", "agent": result[0]}
        else:
            # Update existing agent by ID
            agent_id = agent_data.get("id")
            
            # Verify ownership
            check_query = "SELECT user_id FROM user_data.user_agents WHERE id = %(agent_id)s"
            check_result = db.execute_query(check_query, {"agent_id": str(agent_id)})
            
            if not check_result or len(check_result) == 0:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Agent not found"
                )
            
            if check_result[0]["user_id"] != current_user.id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have permission to update this agent"
                )
            
            update_query = """
            UPDATE user_data.user_agents 
            SET 
                name = %(name)s,
                system_message = %(system_message)s,
                settings = %(settings)s
            WHERE id = %(agent_id)s AND user_id = %(user_id)s
            RETURNING id, name, system_message, settings, created_at, updated_at
            """
            
            params = {
                "name": agent_data.get("name"),
                "system_message": agent_data.get("system_message"),
                "settings": json.dumps(agent_data.get("settings", {})),
                "agent_id": str(agent_id),
                "user_id": str(current_user.id)
            }
            
            result = db.execute_query(update_query, params)
            return {"id": agent_id, "action": "updated", "agent": result[0]}
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save agent: {str(e)}"
        )

@router.delete("/agents/{agent_id}", response_model=Dict[str, Any])
async def delete_agent(agent_id: str, current_user = Depends(get_current_user)):
    """Delete an agent"""
    try:
        db = get_pg_db()
        
        # Verify the user owns this agent
        check_query = "SELECT user_id FROM user_data.user_agents WHERE id = %(agent_id)s"
        check_result = db.execute_query(check_query, {"agent_id": str(agent_id)})
        
        if not check_result or len(check_result) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found"
            )
        
        if check_result[0]["user_id"] != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to delete this agent"
            )
        
        # Delete the agent
        delete_query = "DELETE FROM user_data.user_agents WHERE id = %(agent_id)s"
        db.execute_query(delete_query, {"agent_id": str(agent_id)})
        
        return {"status": "success", "message": "Agent deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete agent: {str(e)}"
        )

# Update documents endpoint to filter by agent
@router.get("/documents", response_model=List[Dict[str, Any]])
async def get_user_documents(agent_id: str = None, current_user = Depends(get_current_user)):
    """Get all documents saved by the current user, optionally filtered by agent"""
    try:
        db = get_pg_db()
        # Updated query to check for indexed documents using qdrant_collection_name
        query = """
        SELECT 
            aj.id, 
            aj.file_name as document_name, 
            aj.file_type,
            aj.task_type as task,
            aj.original_language as language,
            aj.agent_id,
            CASE 
                WHEN aj.qdrant_collection_name IS NOT NULL THEN 'Indexed' 
                ELSE 'Not Indexed' 
            END as vector_status,
            aj.created_at,
            aj.settings->'description' as description
        FROM 
            user_data.audio_jobs aj
        WHERE 
            aj.user_id = %(user_id)s
        """
        
        params = {"user_id": str(current_user.id)}
        
        # Add agent filter if specified and not 'all'
        if agent_id and agent_id.lower() != 'all':
            query += " AND (aj.agent_id = %(agent_id)s OR aj.agent_id IS NULL)"
            params["agent_id"] = str(agent_id)
            
        query += """
        ORDER BY 
            aj.created_at DESC
        LIMIT 100
        """
        
        # Log the query and parameters for debugging
        logger.info(f"Fetching documents with params: {params}")
        
        # Execute query without timeout parameter
        result = db.execute_query(query, params)
        return result
    except Exception as e:
        logger.error(f"Error in get_user_documents: {str(e)}")
        if hasattr(e, '__traceback__'):
            import traceback
            trace_str = ''.join(traceback.format_tb(e.__traceback__))
            logger.error(f"Traceback: {trace_str}")
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve documents: {str(e)}"
        )

@router.delete("/documents/{document_id}", response_model=Dict[str, Any])
async def delete_document(document_id: str, current_user = Depends(get_current_user)):
    """Delete a document and all its associated data"""
    try:
        # Verify the user owns this document
        db = get_pg_db()
        query = """
        SELECT user_id, file_name, qdrant_collection_name 
        FROM user_data.audio_jobs 
        WHERE id = %(document_id)s
        """
        
        result = db.execute_query(query, {"document_id": document_id})
        if not result or len(result) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        if result[0]['user_id'] != current_user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to delete this document"
            )
        
        # Get document information for cleaning up
        document_name = result[0].get('file_name')
        qdrant_collection = result[0].get('qdrant_collection_name')
        
        # Clean up Qdrant collection if it exists
        if qdrant_collection:
            try:
                # Get vector store instance
                vector_store = get_vector_store()
                
                # Delete Qdrant collection
                vector_store.delete_collection(qdrant_collection)
                logger.info(f"Deleted Qdrant collection: {qdrant_collection}")
            except Exception as qdrant_error:
                logger.error(f"Error deleting Qdrant collection: {str(qdrant_error)}")
                # Continue with deletion even if Qdrant cleanup fails
        
        # Delete associated data first (due to foreign key constraints)
        # 1. Delete vector embeddings
        vector_query = """
        DELETE FROM vectors.transcription_embeddings WHERE job_id = %(document_id)s
        """
        db.execute_query(vector_query, {"document_id": document_id})
        
        # 2. Delete transcriptions
        transcription_query = """
        DELETE FROM user_data.transcriptions WHERE job_id = %(document_id)s
        """
        db.execute_query(transcription_query, {"document_id": document_id})
        
        # 3. Finally delete the job
        job_query = """
        DELETE FROM user_data.audio_jobs WHERE id = %(document_id)s
        """
        db.execute_query(job_query, {"document_id": document_id})
        
        return {"status": "success", "message": "Document and all associated data successfully deleted"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )

@router.post("/refresh_document_index_status", response_model=Dict[str, Any])
async def refresh_document_index_status(agent_id: str = None, current_user = Depends(get_current_user)):
    """Refresh the Qdrant indexing status for all documents"""
    try:
        db = get_pg_db()
        vector_store = get_vector_store()
        
        # First, get all documents for the user
        query = """
        SELECT 
            id, 
            file_name as document_name,
            qdrant_collection_name
        FROM 
            user_data.audio_jobs
        WHERE 
            user_id = %(user_id)s
        """
        
        params = {"user_id": str(current_user.id)}
        
        # Add agent filter if specified
        if agent_id:
            query += " AND agent_id = %(agent_id)s"
            params["agent_id"] = str(agent_id)
            
        documents = db.execute_query(query, params)
        
        updated_count = 0
        
        # Check each document and update its Qdrant status
        for doc in documents:
            doc_id = doc['id']
            document_name = doc['document_name']
            
            # Skip if already has a collection name
            if doc.get('qdrant_collection_name'):
                continue
                
            # Create proper collection name
            collection_name = sanitize_collection_name(document_name)
            
            # Check if collection exists in Qdrant
            collection_exists = vector_store.ensure_collection_exists(collection_name)
            
            if collection_exists:
                # Update the document's qdrant_collection_name
                update_query = """
                UPDATE user_data.audio_jobs
                SET qdrant_collection_name = %(collection_name)s
                WHERE id = %(doc_id)s
                """
                
                db.execute_query(update_query, {
                    "collection_name": collection_name,
                    "doc_id": doc_id
                })
                
                updated_count += 1
        
        return {
            "status": "success", 
            "message": f"Refreshed index status for {updated_count} documents"
        }
        
    except Exception as e:
        logger.error(f"Error refreshing document index status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to refresh document index status: {str(e)}"
        )

@router.post("/associate_documents_with_agent", response_model=Dict[str, Any])
async def associate_documents_with_agent(agent_id: str, document_ids: List[str] = None, all_documents: bool = False, current_user = Depends(get_current_user)):
    """Associate existing documents with an agent"""
    try:
        if not agent_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Agent ID is required"
            )
            
        # Handle the special case where agent_id is 'all'
        if agent_id == 'all':
            # Setting to NULL removes agent association
            agent_id = None
        
        # Connect to the database
        db = get_pg_db()
        
        # Verify agent exists and belongs to user (skip check if agent_id is None)
        if agent_id:
            agent_query = """
            SELECT id 
            FROM user_data.user_agents 
            WHERE id = %(agent_id)s AND user_id = %(user_id)s
            """
            
            agent_result = db.execute_query(agent_query, {
                "agent_id": agent_id,
                "user_id": str(current_user.id)
            })
            
            if not agent_result or len(agent_result) == 0:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Agent not found or you do not have permission to access it"
                )
        
        # Build the update query
        base_query = """
        UPDATE user_data.audio_jobs
        SET agent_id = %(agent_id)s
        WHERE user_id = %(user_id)s
        """
        
        params = {
            "agent_id": agent_id,
            "user_id": str(current_user.id)
        }
        
        # Handle specific document IDs if provided
        if document_ids and len(document_ids) > 0:
            # Update specific documents
            doc_ids_str = ",".join([f"'{doc_id}'" for doc_id in document_ids])
            base_query += f" AND id IN ({doc_ids_str})"
            
        elif not all_documents:
            # If neither specific IDs nor all_documents flag is provided
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either document_ids or all_documents flag must be provided"
            )
        
        # Execute the update
        db.execute_query(base_query, params)
        
        # Count the number of updated documents
        count_query = """
        SELECT COUNT(*) as updated_count
        FROM user_data.audio_jobs
        WHERE user_id = %(user_id)s AND agent_id = %(agent_id)s
        """
        
        count_result = db.execute_query(count_query, params)
        updated_count = count_result[0]['updated_count'] if count_result else 0
        
        return {
            "status": "success",
            "message": f"Successfully associated {updated_count} documents with the agent"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error associating documents with agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to associate documents with agent: {str(e)}"
        )

# Add diagnostics endpoint
@router.get("/diagnostics/qdrant", response_model=Dict[str, Any])
async def get_qdrant_diagnostics(agent_id: str = None, current_user = Depends(get_current_user)):
    """Get diagnostic information about Qdrant collection"""
    try:
        # Get vector store
        vs = get_vector_store()
        
        # Get all vectors
        vectors = vs.get_all_vectors(limit=100)
        
        # Get basic stats
        stats = {
            "total_vectors": len(vectors),
            "agents": []
        }
        
        # Analyze agent IDs
        agent_ids = {}
        for vector in vectors:
            a_id = vector["payload"].get("agent_id")
            if a_id:
                if a_id not in agent_ids:
                    agent_ids[a_id] = 0
                agent_ids[a_id] += 1
                
        # Format agent stats
        for a_id, count in agent_ids.items():
            stats["agents"].append({
                "agent_id": a_id,
                "vector_count": count
            })
            
        # Get sample vectors (limited to prevent overwhelming response)
        sample_vectors = []
        for i, vector in enumerate(vectors):
            if i >= 5:  # Just show a few samples
                break
                
            sample_vectors.append({
                "id": vector["id"],
                "content": vector["payload"].get("content", ""),
                "agent_id": vector["payload"].get("agent_id"),
                "document_name": vector["payload"].get("document_name"),
                "embedding_model": vector["payload"].get("embedding_model")
            })
            
        return {
            "status": "success",
            "stats": stats,
            "sample_vectors": sample_vectors
        }
    except Exception as e:
        logger.error(f"Error in get_qdrant_diagnostics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting Qdrant diagnostics: {str(e)}"
        )
