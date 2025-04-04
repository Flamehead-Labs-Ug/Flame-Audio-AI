"""Vector store implementation using LangChain vector stores"""

import os
import logging
from typing import List, Dict, Any, Optional
import json

# LangChain imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_qdrant import Qdrant

# Local imports
from embedding import generate_embedding, list_embedding_models
from .pg_connector import get_pg_db

# Configure logging
logger = logging.getLogger(__name__)

# Global instance for singleton pattern
_vector_store = None


class FlameVectorStore:
    """Manages vector storage and retrieval using LangChain vector stores"""
    
    def __init__(self, supabase_url=None, supabase_key=None, embedding_model="all-MiniLM-L6-v2"):
        """
        Initialize the vector store with connection to Qdrant.
        
        Args:
            supabase_url (str, optional): Not used, kept for backward compatibility
            supabase_key (str, optional): Not used, kept for backward compatibility
            embedding_model (str, optional): HuggingFace embedding model to use
        """
        # Always use Qdrant as the vector store type
        self.vector_store_type = "qdrant"
        
        # Initialize Qdrant credentials
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.qdrant_port = os.getenv("QDRANT_PORT", "6333")
        
        # Validate Qdrant credentials
        if not self.qdrant_url or not self.qdrant_api_key:
            logger.warning("Qdrant URL or API key not provided. Using local embeddings only.")
            self.vectorstore = None
            return
            
        logger.info(f"Using Qdrant at {self.qdrant_url}")
        
        # Load user's vector store settings if available
        self.settings = {
            "embedding_model": embedding_model,
            "dimension": 384,  # Default dimension for MiniLM-L6-v2
            "similarity_threshold": 0.2,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "match_count": 10,
            "enabled": True
        }
        
        # Try to load user settings from database
        try:
            self._load_settings()
        except Exception as e:
            logger.warning(f"Could not load vector store settings, using defaults: {e}")
        
        # Initialize HuggingFace embeddings with user's preferred model
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.settings["embedding_model"],
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            logger.info(f"Initialized embeddings model: {self.settings['embedding_model']}")
        except Exception as e:
            logger.error(f"Error initializing embeddings model: {e}")
            # Fallback to simple embedding function
            self.embeddings = None
            return
        
        # Create the vector store
        try:
            from qdrant_client import QdrantClient
            
            # Check if we're using Qdrant Cloud or local
            is_cloud = "cloud.qdrant.io" in self.qdrant_url or ".gcp." in self.qdrant_url
            
            if is_cloud:
                # For cloud deployments, don't use port - it's part of the URL
                logger.info(f"Detected Qdrant Cloud URL. Connecting to {self.qdrant_url}")
                client = QdrantClient(
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key,
                )
            else:
                # For local or non-standard deployments, use port
                try:
                    port = int(self.qdrant_port)
                except (ValueError, TypeError):
                    port = 6333
                
                logger.info(f"Connecting to Qdrant at {self.qdrant_url}:{port}")
                client = QdrantClient(
                    url=self.qdrant_url,
                    port=port,
                    api_key=self.qdrant_api_key if self.qdrant_api_key else None,
                )
            
            # Test connection
            collections = client.get_collections()
            logger.info(f"Successfully connected to Qdrant. Found collections: {collections}")
            
            # Now create the LangChain vectorstore
            self.vectorstore = Qdrant(
                client=client,
                collection_name="transcription_embeddings",
                embedding=self.embeddings
            )
            logger.info("Successfully initialized Qdrant vector store")
        except ImportError as e:
            logger.error(f"Qdrant client library not installed: {e}")
            self.vectorstore = None
        except Exception as e:
            logger.error(f"Error initializing Qdrant vector store: {e}")
            self.vectorstore = None
    
    def _load_settings(self):
        """Load vector store settings from the database"""
        try:
            # Connect to database
            db = get_pg_db()
            
            # Query settings for current user
            # Note: This relies on a service role connection that can read any user's settings
            query = """
            SELECT 
                settings 
            FROM 
                user_data.vector_store_settings 
            ORDER BY updated_at DESC
            LIMIT 1
            """
            
            result = db.execute_query(query)
            
            # If user has settings, use those
            if result and len(result) > 0 and result[0].get("settings"):
                user_settings = result[0]["settings"]
                # Update our default settings with user preferences
                self.settings.update(user_settings)
                logger.info(f"Loaded user vector store settings: {self.settings}")
        except Exception as e:
            logger.warning(f"Could not load vector store settings, using defaults: {e}")
    
    def store_embedding(self, 
                        user_id: str,
                        job_id: str,
                        chunk_index: int,
                        content: str,
                        embedding: List[float] = None,  # Optional, embeddings are auto-generated
                        metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Store content and its embedding in the vector store"""
        try:
            # Generate a unique ID for this embedding
            record_id = f"{job_id}_{chunk_index}"
            collection_name = f"job_{job_id}"
            
            # Make sure metadata exists
            record_metadata = metadata or {}
            
            # Ensure these fields are in metadata for compatibility
            record_metadata.update({
                "user_id": user_id,
                "job_id": job_id,
                "chunk_index": chunk_index,
                "agent_id": metadata.get("agent_id") if metadata else None,  # Support for multi-agent system
            })
            
            # Extract agent_id for table column
            agent_id = record_metadata.get("agent_id")
            
            # If no embedding provided, generate one
            if embedding is None:
                try:
                    embedding = generate_embedding(content)
                    logger.info(f"Generated embedding with dimensions: {len(embedding)}")
                except Exception as e:
                    logger.error(f"Error generating embedding: {e}")
                    embedding = None
            
            # Check if vectorstore is available
            if self.vectorstore is None:
                logger.warning("Vector store not available. Skipping embedding storage.")
                return {"id": record_id, "collection": collection_name, "status": "skipped"}
            
            # Create a Document object as expected by LangChain
            document = Document(
                page_content=content,
                metadata=record_metadata
            )
            
            # Ensure the collection exists
            try:
                # Get client from the vectorstore
                client = self.vectorstore._client
                
                # Check if collection exists
                collections = client.get_collections().collections
                collection_names = [c.name for c in collections]
                
                if collection_name not in collection_names:
                    # Create collection with the right settings
                    logger.info(f"Creating collection: {collection_name}")
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config={
                            "size": self.embeddings.embed_query("test").shape[0],
                            "distance": "Cosine"
                        }
                    )
            except Exception as e:
                logger.warning(f"Error checking/creating collection: {e}")
                # Continue anyway, as the add_documents might still work
            
            # Add the document to Qdrant
            try:
                self.vectorstore.add_documents(
                    [document],
                    collection_name=collection_name,
                    ids=[record_id]
                )
                
                logger.info(f"Stored document in Qdrant collection: {collection_name}, ID: {record_id}")
                return {"id": record_id, "collection": collection_name, "status": "success"}
            except Exception as e:
                logger.error(f"Error adding document to Qdrant: {e}")
                # Try alternative approach as fallback
                try:
                    # Create a new Qdrant instance specifically for this collection
                    from qdrant_client import QdrantClient
                    from langchain_qdrant import Qdrant
                    
                    # Recreate client with same parameters
                    if hasattr(self.vectorstore, "_client"):
                        client = self.vectorstore._client
                        # Use the same client to create a new Qdrant vectorstore
                        temp_store = Qdrant(
                            client=client,
                            collection_name=collection_name,
                            embedding=self.embeddings
                        )
                        
                        # Try adding document with the new instance
                        temp_store.add_documents(
                            [document],
                            ids=[record_id]
                        )
                        
                        logger.info(f"Stored document in Qdrant using fallback method, collection: {collection_name}")
                        return {"id": record_id, "collection": collection_name, "status": "success"}
                    else:
                        raise ValueError("Vectorstore client not accessible")
                except Exception as inner_e:
                    logger.error(f"Error using fallback method: {inner_e}")
                    return {"id": record_id, "collection": collection_name, "status": "error", "error": str(inner_e)}
            
        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
            return {"id": None, "collection": None, "status": "error", "error": str(e)}
    
    def search_transcriptions(self, query_text: str, user_id: str = None, job_id: str = None, agent_id: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar transcriptions/documents using the vector store"""
        try:
            logger.info(f"Searching for similar transcriptions to: '{query_text}'")
            similarity_threshold = self.settings.get("similarity_threshold", 0.2)
            
            # Create filter dict for vector store
            filter_dict = {}
            if user_id:
                filter_dict["user_id"] = user_id
            if job_id:
                filter_dict["job_id"] = job_id
            if agent_id:
                filter_dict["agent_id"] = agent_id
                
            logger.info(f"Using filter: {filter_dict}, limit: {limit}")
            
            # Check if vector store is available
            if self.vectorstore is None:
                logger.warning("Vector store not available. Cannot search for transcriptions.")
                return []
                
            # Search within the specific job collection
            collection_name = f"job_{job_id}" if job_id else "transcription_embeddings"
            
            # Try with the specified collection 
            try:
                # Use similarity search with scores
                docs_with_scores = self.vectorstore.similarity_search_with_relevance_scores(
                    query=query_text,
                    k=limit,
                    filter=filter_dict if filter_dict else None,
                    score_threshold=similarity_threshold,
                    collection_name=collection_name
                )
                
                logger.info(f"Found {len(docs_with_scores)} results with similarity threshold {similarity_threshold}")
                
                # If no results, try with a lower threshold
                if not docs_with_scores and similarity_threshold > 0.05:
                    lower_threshold = max(0.05, similarity_threshold / 2)
                    logger.info(f"No results found, retrying with lower threshold: {lower_threshold}")
                    
                    docs_with_scores = self.vectorstore.similarity_search_with_relevance_scores(
                        query=query_text,
                        k=limit,
                        filter=filter_dict if filter_dict else None,
                        score_threshold=lower_threshold,
                        collection_name=collection_name
                    )
                    
                    logger.info(f"Found {len(docs_with_scores)} results with lowered threshold {lower_threshold}")
            except Exception as e:
                # If collection doesn't exist, try with default collection
                logger.warning(f"Error searching collection {collection_name}: {e}")
                
                if collection_name != "transcription_embeddings":
                    try:
                        logger.info(f"Trying default collection: transcription_embeddings")
                        docs_with_scores = self.vectorstore.similarity_search_with_relevance_scores(
                            query=query_text,
                            k=limit,
                            filter=filter_dict if filter_dict else None,
                            score_threshold=similarity_threshold,
                            collection_name="transcription_embeddings"
                        )
                        logger.info(f"Found {len(docs_with_scores)} results in default collection")
                    except Exception as inner_e:
                        logger.error(f"Error searching default collection: {inner_e}")
                        return []
                else:
                    # If we're already using the default collection and it failed, return empty
                    return []
            
            # Return formatted results
            results = []
            for doc, score in docs_with_scores:
                result = {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)  # Convert numpy types to Python float for JSON serialization
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching transcriptions: {e}")
            return []
    
    def search_by_text(self, query_text: str, user_id: str = None, job_id: str = None, limit: int = None, agent_id: str = None) -> List[Dict[str, Any]]:
        """Search by text query (simpler interface for retriever)"""
        return self.search_transcriptions(
            query_text=query_text,
            user_id=user_id,
            job_id=job_id,
            limit=limit or self.settings.get("match_count", 10),
            agent_id=agent_id
        )
    
    def as_retriever(self, search_kwargs: Dict[str, Any] = None) -> VectorStoreRetriever:
        """Return a LangChain compatible retriever"""
        search_kwargs = search_kwargs or {}
        return self.vectorstore.as_retriever(**search_kwargs)


# Function to get the singleton instance of the vector store
def get_vector_store(supabase_url=None, service_role_key=None) -> FlameVectorStore:
    """Get the singleton instance of the vector store"""
    global _vector_store
    
    if _vector_store is None:
        try:
            _vector_store = FlameVectorStore()
            logger.info("Initialized FlameVectorStore with Qdrant backend")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    return _vector_store
