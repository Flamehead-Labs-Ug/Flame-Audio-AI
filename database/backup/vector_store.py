import os
import logging
from typing import Dict, Any, List, Optional

import numpy as np
from supabase import create_client, Client
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain.schema.retriever import BaseRetriever

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
        
        # Generate embedding for the query
        query_embedding = generate_embedding(query)
        
        # Extract search parameters
        user_id = self.search_kwargs.get("user_id")
        job_id = self.search_kwargs.get("job_id")
        k = self.search_kwargs.get("k", 5)
        
        # Prepare filter metadata
        filter_metadata = {}
        if user_id:
            filter_metadata["user_id"] = user_id
        if job_id:
            filter_metadata["job_id"] = job_id
            
        # Search for similar content
        results = self.vector_store.search_similar(
            query_embedding=query_embedding,
            match_count=k,
            similarity_threshold=1 - 0.5, # Higher threshold is passed to the function
            filter_metadata=filter_metadata
        )
        
        # Convert to LangChain Document format
        documents = []
        for result in results:
            documents.append(
                Document(
                    page_content=result.get("content", ""),
                    metadata={
                        "job_id": result.get("job_id"),
                        "chunk_index": result.get("chunk_index"),
                        "similarity": result.get("similarity"),
                        "metadata": result.get("metadata", {})
                    }
                )
            )
            
        return documents

class VectorStore:
    """Connection handler for Supabase Vector Store"""
    
    def __init__(self):
        self.supabase: Client = None
        self.connect()
    
    def connect(self):
        """Connect to Supabase using environment variables"""
        try:
            url = os.getenv("VECTOR_STORE_URL")
            key = os.getenv("VECTOR_STORE_SERVICE_KEY")
            self.supabase = create_client(url, key)
            logger.info("Successfully connected to Supabase Vector Store")
        except Exception as e:
            logger.error(f"Error connecting to Supabase Vector Store: {e}")
            raise
    
    def store_embedding(self, 
                       user_id: str,
                       job_id: str,
                       chunk_index: int,
                       content: str,
                       embedding: List[float],
                       metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Store an embedding in the vector store"""
        try:
            # First try a direct insert into the pgvector table
            try:
                # Try inserting directly into the table
                result = self.supabase.table('vectors.transcription_embeddings').insert({
                    'user_id': user_id,
                    'job_id': job_id,
                    'chunk_index': chunk_index,
                    'content': content,
                    'embedding': embedding,
                    'metadata': metadata
                }).execute()
                
                return result.data
            except Exception as table_error:
                logger.warning(f"Direct table insert failed: {table_error}, trying public function")
                
                # If direct table access fails, try using the public function
                result = self.supabase.rpc(
                    'store_embedding',
                    {
                        'p_user_id': user_id,
                        'p_job_id': job_id,
                        'p_chunk_index': chunk_index,
                        'p_content': content,
                        'p_embedding': embedding,
                        'p_metadata': metadata
                    }
                ).execute()
                
                return result.data
        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
            # Don't raise the exception, just log it and return error info
            return {"status": "error", "message": str(e)}
    
    def search_similar(self,
                      query_embedding: List[float],
                      match_count: int = 5,
                      similarity_threshold: float = 0.5,
                      filter_metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar content using direct SQL query"""
        try:
            # Extract user ID and job ID from filter metadata
            user_id = filter_metadata.get('user_id') if filter_metadata else None
            job_id = filter_metadata.get('job_id') if filter_metadata else None
            
            if not user_id:
                logger.error("No user_id provided for vector search")
                return []
            
            # Add debugging logs
            logger.info(f"Searching for embeddings with user_id={user_id}, job_id={job_id}")
            logger.info(f"Using similarity threshold: {similarity_threshold}")
            
            # Use a parameterized query instead of string formatting for safety
            function_params = {
                "p_user_id": user_id,
                "p_query_embedding": query_embedding,
                "p_match_threshold": 1 - similarity_threshold, # Higher threshold is passed to the function
                "p_match_count": match_count
            }
            
            if job_id:
                # If job_id is provided, include it in a filter
                function_params["p_job_id"] = job_id
            
            # Try to use match_transcriptions function first
            try:
                logger.info("Trying to call match_transcriptions function")
                result = self.supabase.rpc(
                    "match_transcriptions",
                    {
                        "query_embedding": query_embedding,
                        "match_count": match_count,
                        "similarity_threshold": similarity_threshold,
                        "filter": {"transcription_id": job_id} if job_id else {}
                    }
                ).execute()
                
                if result.data:
                    logger.info(f"Found {len(result.data)} results using match_transcriptions")
                    return result.data
            except Exception as e:
                logger.warning(f"match_transcriptions failed: {e}, trying fallback")
            
            # Fallback to direct table query using .from_ method
            try:
                logger.info("Trying direct postgrest query")
                query = self.supabase.from_("vectors.transcription_embeddings")\
                    .select("*")\
                    .filter("user_id", "eq", user_id)
                
                if job_id:
                    # If job_id is provided, add the filter directly to the query
                    # The .or_ method doesn't exist in newer versions of supabase-py
                    # Use individual filters instead
                    query = query.filter("job_id", "eq", job_id)
                
                result = query.execute()
                
                if result.data:
                    logger.info(f"Found {len(result.data)} results using direct query")
                    return result.data
                    
            except Exception as e:
                logger.warning(f"Direct table query failed: {e}")
            
            # If nothing worked, return empty list
            logger.warning(f"No matching documents found for {job_id}")
            return []
                
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            logger.error(f"Filter metadata: {filter_metadata}")
            return []  # Return empty list instead of raising to avoid breaking chat flow

    def debug_vector_store(self):
        """Debug function to check vector store connection and structure"""
        try:
            # First, test a simple query to list tables
            tables_result = self.supabase.table('information_schema.tables')\
                .select('table_schema, table_name')\
                .eq('table_schema', 'vectors')\
                .execute()
            
            logger.info(f"Vector tables: {tables_result.data}")
            
            # Try to get columns for the transcription_embeddings table
            if any(t.get('table_name') == 'transcription_embeddings' for t in tables_result.data):
                columns_result = self.supabase.table('information_schema.columns')\
                    .select('column_name, data_type')\
                    .eq('table_schema', 'vectors')\
                    .eq('table_name', 'transcription_embeddings')\
                    .execute()
                
                logger.info(f"Transcription embeddings columns: {columns_result.data}")
            
            # Try to list available functions
            functions_result = self.supabase.table('information_schema.routines')\
                .select('routine_schema, routine_name')\
                .execute()
            
            logger.info(f"Available functions: {[f'{f.get('routine_schema')}.{f.get('routine_name')}' for f in functions_result.data]}")
            
            return {
                'tables': tables_result.data,
                'columns': columns_result.data if 'columns_result' in locals() else None,
                'functions': functions_result.data
            }
        except Exception as e:
            logger.error(f"Error debugging vector store: {e}")
            raise

    def as_retriever(self, search_kwargs: Dict[str, Any] = None) -> BaseRetriever:
        """Return a LangChain compatible retriever"""
        return CustomRetriever(self, search_kwargs)

# Singleton instance
_vector_store = None

def get_vector_store():
    """Get singleton vector store connection"""
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
