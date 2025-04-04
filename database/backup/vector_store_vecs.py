import os
import logging
from typing import Dict, Any, List, Optional

import vecs
from vecs.adapter import Adapter, TextEmbedding
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.schema.retriever import BaseRetriever
import urllib.parse

load_dotenv()

logger = logging.getLogger(__name__)

class CustomRetriever(BaseRetriever):
    """Simple retriever that works with our VectorStore"""
    
    def __init__(self, vector_store, search_kwargs=None):
        """Initialize with vector store and search parameters"""
        self.vector_store = vector_store
        self.search_kwargs = search_kwargs or {}
        super().__init__()
        
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get documents relevant to the query"""
        from embedding import generate_embedding
        
        # Extract search parameters
        user_id = self.search_kwargs.get("user_id")
        job_id = self.search_kwargs.get("job_id")
        k = self.search_kwargs.get("k", 5)
        
        # Search for similar documents
        results = self.vector_store.search_by_text(
            query_text=query,
            user_id=user_id,
            job_id=job_id,
            limit=k
        )
        
        # Convert to LangChain documents
        documents = []
        for result in results:
            documents.append(
                Document(
                    page_content=result.get("content", ""),
                    metadata={
                        "job_id": result.get("job_id"),
                        "chunk_index": result.get("chunk_index", 0),
                        "similarity": result.get("similarity", 0),
                        **result.get("metadata", {})
                    }
                )
            )
        
        return documents

class VectorStore:
    """Manage vector storage and retrieval using vecs"""
    
    def __init__(self, vector_store_url, service_role_key=None):
        """
        Initialize the vector store with the given URL.
        
        Args:
            vector_store_url (str): URL to connect to the vector store
            service_role_key (str, optional): Service role key for authentication
        """
        self.url = vector_store_url
        self.service_role_key = service_role_key
        self.vx = None
        self.transcriptions = None
        
        try:
            # Try to build a PostgreSQL connection string for Supabase pooler
            # Read credentials from environment variables
            pg_host = os.getenv("PG_HOST", "aws-0-eu-central-1.pooler.supabase.com")
            pg_port = os.getenv("PG_PORT", "6543")
            pg_database = os.getenv("PG_DATABASE", "postgres")
            pg_user = os.getenv("PG_USER")
            pg_password = os.getenv("PG_PASSWORD")
            
            # Form the PostgreSQL connection string for Supabase pooler
            if pg_host and pg_user and pg_password:
                logger.info(f"Using Supabase PostgreSQL Pooler at {pg_host}:{pg_port}")
                postgres_url = f"postgresql://{pg_user}:{pg_password}@{pg_host}:{pg_port}/{pg_database}"
                
                try:
                    # Create vecs client using PostgreSQL pooler
                    self.vx = vecs.create_client(postgres_url)
                    
                    # Get or create collection for transcriptions with TextEmbedding adapter
                    self.transcriptions = self.vx.get_or_create_collection(
                        name="transcription_embeddings",
                        dimension=384,
                        adapter=Adapter([TextEmbedding(model='all-MiniLM-L6-v2')])
                    )
                    
                    # Create index if it doesn't exist
                    self.transcriptions.create_index()
                    logger.info("Successfully connected to Supabase Vector Store using vecs")
                    return
                except Exception as pooler_error:
                    logger.error(f"Error connecting to Supabase PostgreSQL Pooler: {pooler_error}")
                    # Will fall back to original vector store implementation
            
            # If we couldn't connect with PostgreSQL pooler, fall back to original implementation
            logger.warning("Could not connect using PostgreSQL pooler. Falling back to REST API implementation.")
            raise ValueError("Vector store connection failed - using fallback implementation")
            
        except Exception as e:
            logger.error(f"Error connecting to Vector Store: {e}")
            raise
    
    def store_embedding(self, 
                       user_id: str,
                       job_id: str,
                       chunk_index: int,
                       content: str,
                       embedding: List[float] = None,  # Optional since TextEmbedding adapter will create embeddings
                       metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Store content and its embedding in the vector store"""
        try:
            # Generate a unique ID for this embedding
            record_id = f"{job_id}_{chunk_index}"
            
            # Prepare metadata
            record_metadata = {
                "user_id": user_id,
                "job_id": job_id,
                "chunk_index": chunk_index,
                **(metadata or {})
            }
            
            # Store in vecs
            if embedding is not None:
                # If embedding is provided, store it directly
                self.transcriptions.upsert([
                    (record_id, embedding, record_metadata)
                ])
            else:
                # If no embedding provided, let the TextEmbedding adapter create it
                self.transcriptions.upsert([
                    (record_id, content, record_metadata)
                ])
            
            return {"id": record_id}
        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
            raise
    
    def search_similar(self,
                      query_embedding: List[float] = None,
                      query_text: str = None,
                      match_count: int = 5,
                      similarity_threshold: float = 0.5,
                      filter_metadata: Dict[str, Any] = None,
                      post_filter_job_id: str = None) -> List[Dict[str, Any]]:
        """Search for similar content using either embedding or text"""
        try:
            # Extract user ID and job ID from filter metadata
            user_id = None
            job_id = None
            
            if filter_metadata:
                # Handle both direct filters and composite ($and) filters
                if '$and' in filter_metadata:
                    # Extract from composite filter structure
                    for condition in filter_metadata['$and']:
                        if 'user_id' in condition:
                            user_id = condition['user_id'].get('$eq')
                        if 'job_id' in condition:
                            job_id = condition['job_id'].get('$eq')
                else:
                    # Direct filter extraction
                    if 'user_id' in filter_metadata:
                        user_id = filter_metadata.get('user_id')
                        # Handle both simple and complex formats
                        if isinstance(user_id, dict) and '$eq' in user_id:
                            user_id = user_id['$eq']
                    if 'job_id' in filter_metadata:
                        job_id = filter_metadata.get('job_id')
                        # Handle both simple and complex formats
                        if isinstance(job_id, dict) and '$eq' in job_id:
                            job_id = job_id['$eq']
            
            if not user_id:
                logger.error("No user_id provided for vector search")
                return []
            
            # For vecs, use the most basic filter structure possible
            # From vecs documentation: {"year": {"$eq": 2012}}
            # Each filter must be a single equality check with $eq operator
            if user_id:
                filters = {"user_id": {"$eq": user_id}}
                
                # Log search parameters
                if job_id:
                    logger.info(f"Searching with basic user_id filter: {filters}, will post-filter by job_id: {job_id}")
                else:
                    logger.info(f"Searching with basic user_id filter: {filters}")
            else:
                logger.error("No user_id provided for vector search")
                return []
            
            # Perform search based on what was provided
            if query_embedding is not None:
                results = self.transcriptions.query(
                    data=query_embedding,
                    limit=match_count * 2 if job_id else match_count,  # Get more results if we need to post-filter
                    filters=filters,
                    include_metadata=True,
                    include_value=True
                )
            elif query_text is not None:
                results = self.transcriptions.query(
                    data=query_text,  # TextEmbedding adapter will convert this to an embedding
                    limit=match_count * 2 if job_id else match_count,  # Get more results if we need to post-filter
                    filters=filters,
                    include_metadata=True,
                    include_value=True
                )
            else:
                logger.error("Either query_embedding or query_text must be provided")
                return []
            
            # Format and filter results
            formatted_results = []
            for record_id, similarity, metadata in results:
                # If job_id filtering is required, apply it here
                if post_filter_job_id and metadata.get("job_id") != post_filter_job_id:
                    continue
                    
                # Extract job_id and chunk_index from metadata
                result = {
                    "id": record_id,
                    "job_id": metadata.get("job_id"),
                    "chunk_index": metadata.get("chunk_index", 0),
                    "similarity": similarity,
                    "metadata": metadata
                }
                
                # Retrieve the content for this record
                # This requires an additional query since vecs doesn't return the vector content directly
                content_record = self.get_content_by_id(record_id)
                if content_record:
                    result["content"] = content_record.get("content", "")
                else:
                    result["content"] = ""
                
                formatted_results.append(result)
            
            logger.info(f"Found {len(formatted_results)} results")
            return formatted_results
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            logger.error(f"Filter metadata: {filter_metadata}")
            return []  # Return empty list instead of raising to avoid breaking chat flow
    
    def get_content_by_id(self, record_id: str) -> Dict[str, Any]:
        """Retrieve content for a specific record by ID"""
        try:
            # In vecs, we need to query the original table to get content
            # This is a simple PostgreSQL query with proper parameter handling
            query = """
            SELECT id, content, metadata 
            FROM vectors.transcription_embeddings 
            WHERE id = %s
            """
            
            # Execute using the vecs client's underlying connection
            with self.vx._pool.connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, (record_id,))  # Use parameterized query to prevent SQL injection
                    result = cursor.fetchone()
                    
                    if result:
                        logger.info(f"Retrieved content for record {record_id}")
                        return {"id": result[0], "content": result[1], "metadata": result[2]}
                    else:
                        logger.warning(f"No content found for record {record_id}")
                        
                        # Fallback: try querying by just the first part of the ID (before any dash)
                        if '-' in record_id:
                            base_id = record_id.split('-')[0]
                            fallback_query = """
                            SELECT id, content, metadata 
                            FROM vectors.transcription_embeddings 
                            WHERE id LIKE %s || '%%'
                            LIMIT 1
                            """
                            
                            cursor.execute(fallback_query, (base_id,))
                            fallback_result = cursor.fetchone()
                            
                            if fallback_result:
                                logger.info(f"Retrieved content using fallback ID match for {record_id}")
                                return {"id": fallback_result[0], "content": fallback_result[1], "metadata": fallback_result[2]}
            
            # If no results were found using either method
            logger.warning(f"Could not find content for {record_id} in the database")
            # Try a more direct approach - query the transcriptions table directly by job_id
            # This is a fallback to ensure we get something
            fallback_content_query = """
            SELECT id, chunk_index, content FROM vectors.transcription_chunks 
            WHERE id LIKE %s || '%%' OR job_id = %s
            LIMIT 1
            """
            
            with self.vx._pool.connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(fallback_content_query, (record_id, record_id))
                    content_result = cursor.fetchone()
                    
                    if content_result:
                        logger.info(f"Retrieved content from raw transcription table for {record_id}")
                        return {"id": content_result[0], "content": content_result[2], "metadata": {"chunk_index": content_result[1]}}
            
            return None
        except Exception as e:
            logger.error(f"Error getting content for record {record_id}: {e}")
            return None
    
    def search_by_text(self, query_text: str, user_id: str = None, job_id: str = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Search by text query (simpler interface for retriever)"""
        # We can leverage our improved search_similar_transcriptions method
        # which now has fallback to direct database search
        return self.search_similar_transcriptions(
            query_text=query_text,
            user_id=user_id,
            job_id=job_id,
            limit=limit
        )
    
    def search_similar_transcriptions(self, query_text, user_id=None, job_id=None, limit=10):
        """
        Search for similar transcriptions using vector similarity search.
        Includes filters for user_id and job_id if provided.
        
        Args:
            query_text (str): The query text to search for
            user_id (str, optional): Filter by user ID
            job_id (str, optional): Filter by job ID
            limit (int, optional): Maximum number of results. Defaults to 10.
            
        Returns:
            list: List of documents with content and metadata
        """
        logger.info(f"Searching for similar transcriptions to: '{query_text}'")
        
        # For vecs compatibility, only pass user_id as a direct filter
        # We'll handle job_id filter post-query
        filter_metadata = None
        if user_id:
            filter_metadata = {"user_id": {"$eq": user_id}}
            
        # Calculate how many results to fetch based on whether we'll post-filter
        fetch_limit = limit * 3 if job_id else limit
        logger.info(f"Using simple filter: {filter_metadata}, post-filtering for job_id: {job_id}, fetch_limit: {fetch_limit}")
        
        # Get similar documents using vecs similarity search
        search_results = self.search_similar(
            query_text=query_text,
            match_count=fetch_limit,
            similarity_threshold=0.2,  # Lower the threshold to get more results
            filter_metadata=filter_metadata,
            post_filter_job_id=job_id  # Pass job_id separately for post-filtering
        )
        
        if search_results:
            logger.info(f"Found {len(search_results)} results using vector search")
            # Post-filter for job_id if needed
            if job_id:
                filtered_results = []
                for result in search_results:
                    if result.get("job_id") == job_id:
                        filtered_results.append(result)
                return filtered_results
            else:    
                return search_results
        else:
            logger.warning(f"No vector search results, trying direct database search for: '{query_text}'")
            # Fallback: Try a direct text search in the PostgreSQL database
            try:
                # Query the PostgreSQL transcriptions table directly
                fallback_query = """
                SELECT t.id, t.job_id, t.chunk_index, t.text as content, 
                       t.start_time, t.end_time, t.language
                FROM user_data.transcriptions t
                WHERE 
                    t.user_id = %s
                    AND (%s IS NULL OR t.job_id = %s)
                    AND t.text ILIKE %s
                ORDER BY t.job_id, t.chunk_index
                LIMIT %s
                """
                
                with self.vx._pool.connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            fallback_query, 
                            (user_id, job_id, job_id, f'%{query_text}%', limit)
                        )
                        results = cursor.fetchall()
                        
                        if results:
                            # Convert results to the expected format
                            logger.info(f"Found {len(results)} results using direct database search")
                            formatted_results = []
                            for row in results:
                                formatted_results.append({
                                    "id": row[0],
                                    "job_id": row[1],
                                    "chunk_index": row[2],
                                    "content": row[3],
                                    "similarity": 0.95,  # Add an artificial high similarity score
                                    "metadata": {
                                        "job_id": row[1],
                                        "chunk_index": row[2],
                                        "start_time": row[4],
                                        "end_time": row[5],
                                        "language": row[6],
                                        "similarity": 0.95,
                                        "direct_match": True  # Indicate this was a direct match
                                    }
                                })
                            return formatted_results
            except Exception as e:
                logger.error(f"Error in fallback direct database search: {e}")
            
            # If we got here, both search methods failed
            logger.warning(f"No similar transcriptions found for query: '{query_text}'")
            return []
    
    def as_retriever(self, search_kwargs: Dict[str, Any] = None) -> BaseRetriever:
        """Return a LangChain compatible retriever"""
        return CustomRetriever(self, search_kwargs)
    
    def close(self):
        """Close the connection to the vector store"""
        if self.vx:
            self.vx.disconnect()
            self.vx = None

# Singleton instance
_vector_store = None

def get_vector_store(vector_store_url, service_role_key=None) -> VectorStore:
    """Get the global vector store instance"""
    global _vector_store
    if _vector_store is None:
        try:
            # Try to create a vecs-based vector store
            _vector_store = VectorStore(vector_store_url, service_role_key)
        except Exception as e:
            # If vecs fails, fall back to the original implementation
            logger.warning(f"Failed to initialize vecs-based vector store: {e}")
            logger.warning("Falling back to original vector store implementation")
            from .vector_store import get_vector_store as get_original_vector_store
            from .vector_store import VectorStore as OriginalVectorStore
            from embedding import generate_embedding
            
            # Get the original vector store instance
            original_store = get_original_vector_store()
            
            # Add compatibility methods if they don't exist
            if not hasattr(original_store, 'search_by_text'):
                def search_by_text(self, query_text, user_id=None, job_id=None, limit=10):
                    """Compatibility method for original vector store"""
                    logger.info(f"Using compatibility search_by_text method for query: {query_text}")
                    
                    # Generate embedding for text query
                    query_embedding = generate_embedding(query_text)
                    
                    # For compatibility with original vector store, use only user_id as the filter
                    filter_metadata = {}
                    if user_id:
                        filter_metadata["user_id"] = user_id
                    
                    # Calculate how many results to fetch based on whether we'll post-filter
                    fetch_limit = limit * 3 if job_id else limit
                    logger.info(f"Compatibility search with filter: {filter_metadata}, post-filtering for job_id: {job_id}")
                    
                    # Get search results from original vector store implementation
                    results = self.search_similar(
                        query_embedding=query_embedding,
                        match_count=fetch_limit,
                        similarity_threshold=0.2,  # Lower threshold for better matches
                        filter_metadata=filter_metadata
                    )
                    
                    # Post-filter for job_id if needed
                    if job_id and results:
                        filtered_results = []
                        for result in results:
                            if result.get("job_id") == job_id or result.get("metadata", {}).get("job_id") == job_id:
                                filtered_results.append(result)
                        return filtered_results
                    else:
                        return results
                
                # Add the method to the instance
                import types
                original_store.search_by_text = types.MethodType(search_by_text, original_store)
            
            if not hasattr(original_store, 'search_similar_transcriptions'):
                def search_similar_transcriptions(self, query_text, user_id=None, job_id=None, limit=10):
                    """Compatibility method for original vector store"""
                    logger.info(f"Using compatibility search_similar_transcriptions method for query: {query_text}")
                    
                    # Generate embedding for text query
                    query_embedding = generate_embedding(query_text)
                    
                    # For compatibility with original vector store, use only user_id as the filter
                    filter_metadata = {}
                    if user_id:
                        filter_metadata["user_id"] = user_id
                    
                    # Calculate how many results to fetch based on whether we'll post-filter
                    fetch_limit = limit * 3 if job_id else limit
                    logger.info(f"Compatibility search with filter: {filter_metadata}, post-filtering for job_id: {job_id}")
                    
                    # Get search results from original vector store implementation
                    results = self.search_similar(
                        query_embedding=query_embedding,
                        match_count=fetch_limit,
                        similarity_threshold=0.2,  # Lower threshold for better matches
                        filter_metadata=filter_metadata
                    )
                    
                    # Post-filter for job_id if needed
                    if job_id and results:
                        filtered_results = []
                        for result in results:
                            if result.get("job_id") == job_id or result.get("metadata", {}).get("job_id") == job_id:
                                filtered_results.append(result)
                        return filtered_results
                    else:
                        return results
                
                # Add the method to the instance
                import types
                original_store.search_similar_transcriptions = types.MethodType(search_similar_transcriptions, original_store)
            
            return original_store
    return _vector_store
