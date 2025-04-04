"""Vector store implementation using Qdrant and sentence transformers"""

import os
import logging
import json
import uuid
from typing import List, Dict, Any, Optional
import re
import numpy as np

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models as qmodels
import httpx
from huggingface_hub import InferenceClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global cache for embedding models to avoid reloading
MODEL_CACHE = {}

# Global cache for HuggingFace Inference clients
HF_CLIENT_CACHE = {}

# Create a zero vector with 384 dimensions for PostgreSQL placeholder
ZERO_VECTOR_384 = "[" + ",".join(["0.0"] * 384) + "]"

# Utility function for creating safe collection names
def sanitize_collection_name(name: str) -> str:
    """Convert a document name to a safe Qdrant collection name."""
    if not name:
        return ""
    # Replace spaces with underscores and remove invalid characters
    sanitized = re.sub(r'[^\w\-_]', '', name.lower().replace(' ', '_'))
    # Ensure the name isn't too long for Qdrant (max 255 chars)
    return sanitized[:200] if len(sanitized) > 200 else sanitized

# Local imports
from .pg_connector import get_pg_db

# Global instance for singleton pattern
_vector_store = None

class FlameVectorStore:
    """Manages vector storage and retrieval using Qdrant and sentence transformers"""
    
    def __init__(self, embedding_model="all-MiniLM-L6-v2"):
        """Initialize the vector store"""
        # Get settings from environment
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        self.qdrant_port = os.getenv("QDRANT_PORT", "6333")
        
        # Initialize client as None
        self.client = None
        
        # Set embedding model
        self.embedding_model = embedding_model
        
        # Initialize with default settings
        self.settings = {
            "chunk_size": 1500,
            "chunk_overlap": 200,
            "similarity_threshold": 0.2,
            "match_count": 10,
            "embedding_model": embedding_model
        }
        
        # Load settings from database
        try:
            user_settings = self._load_user_settings()
            if user_settings:
                self.settings = user_settings
        except Exception as e:
            logger.warning(f"Error loading user settings, using defaults: {e}")
        
        # Initialize the Qdrant client
        try:
            self._initialize_client()
            
            # Ensure the standard collection exists
            self.ensure_collection_exists("transcription_embeddings")
            
            # Check if collection is empty but documents exist in PostgreSQL
            self._check_and_restore_collection("transcription_embeddings")
        except Exception as e:
            logger.error(f"Error initializing Qdrant client: {e}")
            # We'll continue without a client - operations that need it will be skipped
        
        logger.info(f"Vector store initialized with model {self.embedding_model}")

    def _initialize_client(self):
        """Initialize the Qdrant client with error handling"""
        try:
            # Check if we have URL or need to use local
            if self.qdrant_url:
                # Cloud Qdrant
                logger.info(f"Connecting to Qdrant Cloud at {self.qdrant_url}")
                self.client = QdrantClient(
                    url=self.qdrant_url,
                    api_key=self.qdrant_api_key,
                    timeout=60
                )
            else:
                # Local Qdrant
                port = int(self.qdrant_port)
                logger.info(f"Connecting to local Qdrant at port {port}")
                self.client = QdrantClient(
                    host="localhost",
                    port=port,
                    timeout=60
                )
            
            # Test connection
            self.client.get_collections()
            logger.info("Successfully connected to Qdrant")
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant client: {e}")
            self.client = None
            raise

    def _load_user_settings(self):
        """Load vector store settings from the database"""
        # Initialize with default settings
        default_settings = {
            "chunk_size": 1500,
            "chunk_overlap": 200,
            "similarity_threshold": 0.2,
            "match_count": 10,
            "embedding_model": self.embedding_model
        }
        
        try:
            # Connect to database
            db = get_pg_db()
            
            # Query settings for current user
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
                default_settings.update(user_settings)
                logger.info(f"Loaded user vector store settings: {default_settings}")
            else:
                logger.info(f"No user vector store settings found, using defaults: {default_settings}")
                
            return default_settings
        except Exception as e:
            logger.warning(f"Could not load vector store settings, using defaults: {e}")
            return default_settings

    def generate_embedding(self, text: str, embedding_model: str = None) -> List[float]:
        """Generate embedding for a text using HuggingFace InferenceClient"""
        if not text:
            logger.warning("Cannot generate embedding for empty text")
            return None
            
        try:
            # Use the specified model or fall back to the default
            model_name = embedding_model or self.settings.get("embedding_model", self.embedding_model)
            
            # Fix model name format if it's not in the correct format
            if model_name == "MiniLM L6":
                model_name = "all-MiniLM-L6-v2"
            elif model_name == "MPNet":
                model_name = "all-mpnet-base-v2"
            elif model_name == "Multilingual":
                model_name = "paraphrase-multilingual-MiniLM-L12-v2"
            
            # Get HuggingFace API key from environment
            huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")
            
            if not huggingface_api_key:
                logger.error("No HuggingFace API key found in environment variables")
                raise ValueError("HuggingFace API key not configured")
            
            # Format model name to include organization for sentence transformers
            if not model_name.startswith("sentence-transformers/"):
                full_model_name = f"sentence-transformers/{model_name}"
            else:
                full_model_name = model_name
                
            # Get or create the inference client from cache
            client_key = f"{huggingface_api_key}:{full_model_name}"
            if client_key not in HF_CLIENT_CACHE:
                logger.info(f"Creating new HuggingFace InferenceClient for model {full_model_name}")
                HF_CLIENT_CACHE[client_key] = InferenceClient(
                    provider="hf-inference",
                    api_key=huggingface_api_key,
                )
            
            client = HF_CLIENT_CACHE[client_key]
            
            # Use feature-extraction for embedding generation
            logger.info(f"Generating embedding via HuggingFace InferenceClient with model {full_model_name}")
            
            # Get embedding from the API
            embedding = client.feature_extraction(
                text,
                model=full_model_name
            )
            
            # Check if the embedding is a list of lists (for multiple sentences)
            if isinstance(embedding, list) and all(isinstance(item, list) for item in embedding):
                # Average sentence embeddings if multiple sentences are returned
                embedding = [sum(values) / len(values) for values in zip(*embedding)]
            
            logger.info(f"Generated embedding via HuggingFace API with model {full_model_name}, dimensions: {len(embedding)}")
            return self._ensure_list_embedding(embedding)
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return a default embedding with the right dimensions
            # Default to 384 dimensions for all-MiniLM-L6-v2
            model_dimensions = {
                "all-MiniLM-L6-v2": 384,
                "all-mpnet-base-v2": 768,
                "paraphrase-multilingual-MiniLM-L12-v2": 384
            }
            dimension = model_dimensions.get(model_name if 'model_name' in locals() else self.embedding_model, 384)
            logger.warning(f"Returning empty embedding with {dimension} dimensions")
            return [0.0] * dimension
    
    def _ensure_list_embedding(self, embedding) -> List[float]:
        """Helper method to ensure embedding is a Python list, not a NumPy array"""
        # Convert numpy arrays to Python lists if needed
        if hasattr(embedding, '__iter__'):
            if hasattr(embedding, 'tolist'):
                logger.info(f"Converting numpy array embedding to Python list")
                return embedding.tolist()
            elif isinstance(embedding, list):
                # If it's already a list but might contain numpy values, convert nested arrays
                if embedding and hasattr(embedding[0], 'tolist'):
                    return [float(x) for x in embedding]
        return embedding

    def ensure_collection_exists(self, collection_name: str) -> bool:
        """Ensure collection exists in Qdrant"""
        if not self.client:
            logger.warning("Qdrant client not initialized. Cannot ensure collection exists.")
            return False
            
        try:
            # Get list of collections
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            # Check if collection exists
            if collection_name in collection_names:
                logger.info(f"Collection {collection_name} already exists")
                # Check if vectors are indexed
                collection_info = self.client.get_collection(collection_name=collection_name)
                if hasattr(collection_info, "indexed_vectors_count") and collection_info.indexed_vectors_count == 0:
                    logger.warning(f"Collection {collection_name} has 0 indexed vectors. Reindexing...")
                    self._reindex_collection(collection_name)
                return True
                
            # Create collection
            # Get dimension from settings or use default
            dimension = 384  # Default dimension for all-MiniLM-L6-v2
            if self.settings and "dimension" in self.settings:
                dimension = self.settings["dimension"]
            elif self.settings and "embedding_model" in self.settings:
                # Try to determine dimension from model name
                model_dimensions = {
                    "all-MiniLM-L6-v2": 384,
                    "all-mpnet-base-v2": 768,
                    "paraphrase-multilingual-MiniLM-L12-v2": 384
                }
                dimension = model_dimensions.get(self.settings["embedding_model"], 384)
            
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=dimension,
                    distance=qmodels.Distance.COSINE
                )
            )
            logger.info(f"Created collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error ensuring collection exists: {e}")
            return False
    
    def _reindex_collection(self, collection_name: str) -> bool:
        """Force reindexing of vectors in a collection"""
        if not self.client:
            logger.warning("Qdrant client not initialized. Cannot reindex collection.")
            return False
            
        try:
            # Get all points from the collection (up to 1000)
            logger.info(f"Retrieving points from collection {collection_name} for reindexing")
            points = self.client.scroll(
                collection_name=collection_name,
                limit=1000,  # Get up to 1000 points
                with_payload=True,
                with_vectors=True,
            )[0]
            
            logger.info(f"Retrieved {len(points)} points for reindexing")
            
            # No points to reindex
            if not points:
                logger.warning(f"No points found in collection {collection_name} for reindexing")
                return False
                
            # Force reindex by recreating the collection
            try:
                # Get vector size from first point
                vector_size = len(points[0].vector)
                
                # Delete the original collection
                logger.info(f"Deleting original collection {collection_name}")
                self.client.delete_collection(collection_name=collection_name)
                
                # Create a new collection with the same name directly (no temp collection needed)
                logger.info(f"Creating new collection {collection_name}")
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=vector_size,
                        distance=qmodels.Distance.COSINE
                    ),
                )
                
                # Prepare points for upsert
                upsert_points = []
                for point in points:
                    upsert_points.append(
                        qmodels.PointStruct(
                            id=point.id,
                            vector=point.vector,
                            payload=point.payload
                        )
                    )
                
                # Insert points into the new collection
                logger.info(f"Inserting {len(upsert_points)} points into new collection")
                self.client.upsert(
                    collection_name=collection_name,
                    points=upsert_points,
                    wait=True # Make sure the operation completes before proceeding
                )
                
                # Force optimization to build the index
                logger.info(f"Forcing optimization of collection {collection_name}")
                self.force_index_optimization(collection_name=collection_name)
                
                logger.info(f"Successfully reindexed collection {collection_name}")
                return True
                
            except Exception as e:
                logger.error(f"Error reindexing collection: {e}")
                raise
                
        except Exception as e:
            logger.error(f"Error in reindex collection operation: {e}")
            return False
    
    def force_index_optimization(self, collection_name: str = "transcription_embeddings") -> bool:
        """Force Qdrant to optimize the collection and build the HNSW index
        
        This will override the indexing_threshold and force Qdrant to build the HNSW index
        for all points in the collection, even if it's below the normal threshold.
        """
        if not self.client:
            logger.warning("Qdrant client not initialized.")
            return False
            
        try:
            # Get current collection info
            collection_info = self.client.get_collection(collection_name=collection_name)
            
            if hasattr(collection_info, "points_count") and collection_info.points_count == 0:
                logger.warning(f"Collection {collection_name} has no points, nothing to optimize")
                return False
                
            # Force collection optimization by using public API methods instead of _http
            logger.info(f"Setting low indexing threshold for collection {collection_name}")
            
            # Update collection config to use much lower thresholds
            self.client.update_collection(
                collection_name=collection_name,
                optimizer_config=qmodels.OptimizersConfigDiff(
                    indexing_threshold=1,  # Set to 1 to force indexing of any number of vectors
                    flush_interval_sec=1    # Faster flushing
                )
            )
            
            # Wait a moment for configuration update to apply
            import time
            time.sleep(1)
            
            # Check if vectors are now indexed
            collection_info = self.client.get_collection(collection_name=collection_name)
            indexed_count = getattr(collection_info, "indexed_vectors_count", 0)
            points_count = getattr(collection_info, "points_count", 0)
            
            logger.info(f"After optimization: {indexed_count}/{points_count} vectors indexed in {collection_name}")
            
            return indexed_count > 0
            
        except Exception as e:
            logger.error(f"Error forcing index optimization: {e}")
            return False
            
    def store_embedding(self, 
                        user_id: str,
                        document_name: str,
                        chunk_index: int,
                        content: str,
                        embedding: List[float] = None,  # Optional, embeddings are auto-generated
                        metadata: Dict[str, Any] = None,
                        embedding_model: str = None) -> Dict[str, Any]:
        """Store content and its embedding in the vector store"""
        if not self.client:
            logger.warning("Qdrant client not initialized. Cannot store embedding.")
            return {"status": "error", "message": "Vector store client not initialized"}
            
        try:
            # Generate a string ID for compatibility with older code
            string_id = f"{document_name}_{chunk_index}"
            
            # Create UUID from string for Qdrant
            point_id = uuid.uuid5(uuid.NAMESPACE_DNS, string_id)
            
            # Use a standard collection name for all documents
            collection_name = "transcription_embeddings"
            
            # Make sure metadata exists
            record_metadata = metadata or {}
            
            # Use specified embedding model if provided
            active_model = embedding_model or self.embedding_model
            
            # Ensure these fields are in metadata for compatibility
            record_metadata.update({
                "user_id": user_id,
                "document_name": document_name,
                "chunk_index": chunk_index,
                "agent_id": metadata.get("agent_id") if metadata else None,  # Support for multi-agent system
                "content": content,  # Store the content in metadata for easier retrieval
                "string_id": string_id,  # Store the original string ID for reference
                "embedding_model": active_model  # Track which model was used
            })
            
            # Create embeddings if not provided - use the cached model approach for better performance
            if embedding is None:
                embedding = self.generate_embedding(content, embedding_model)
                if not embedding:
                    return {"status": "error", "message": "Failed to generate embedding"}
            
            # Convert numpy arrays to Python lists if needed
            embedding = self._ensure_list_embedding(embedding)
            
            # Make sure the collection exists
            if not self.ensure_collection_exists(collection_name):
                return {"status": "error", "message": f"Failed to ensure collection {collection_name} exists"}
                
            # Store in Qdrant
            self.client.upsert(
                collection_name=collection_name,
                points=[
                    qmodels.PointStruct(
                        id=str(point_id),  # Convert UUID to string
                        vector=embedding,
                        payload=record_metadata
                    )
                ]
            )
            
            # Also store in PostgreSQL tracking table for backwards compatibility
            try:
                # Connect to database
                from .pg_connector import get_pg_db
                db = get_pg_db()
                
                # Prepare data for PostgreSQL - ensure embedding is properly converted to a string
                pg_metadata = record_metadata.copy()
                
                # Remove any data that might cause JSON serialization issues
                if "embedding" in pg_metadata:
                    del pg_metadata["embedding"]
                
                # Insert into tracking table with proper data types - WITH placeholder embedding
                query = """
                INSERT INTO vectors.transcription_embeddings
                    (id, document_name, user_id, agent_id, content, embedding, metadata)
                VALUES
                    (%(id)s, %(document_name)s, %(user_id)s, %(agent_id)s, %(content)s, %(embedding)s, %(metadata)s::jsonb)
                ON CONFLICT (id) DO UPDATE 
                SET 
                    metadata = EXCLUDED.metadata,
                    content = EXCLUDED.content
                RETURNING id
                """
                
                # Create a small placeholder embedding to satisfy not-null constraint
                # Using a vector of zeros that are cast to the VECTOR type
                db.execute_query(query, {
                    "id": string_id,
                    "document_name": document_name,
                    "user_id": user_id,
                    "agent_id": record_metadata.get("agent_id"),
                    "content": content,
                    "embedding": ZERO_VECTOR_384,  # Minimal placeholder embedding to satisfy not-null constraint
                    "metadata": json.dumps(pg_metadata)  # Properly JSON serialize the metadata
                })
                
                logger.info(f"Stored tracking record in PostgreSQL for embedding with ID {string_id}")
            except Exception as e:
                logger.warning(f"Failed to store tracking record in PostgreSQL: {e} - continuing with Qdrant storage only")
            
            logger.info(f"Stored embedding in collection {collection_name} with ID {point_id} (string ID: {string_id})")
            return {"id": string_id, "status": "success"}
            
        except Exception as e:
            logger.error(f"Error storing embedding: {e}")
            return {"status": "error", "message": str(e)}
    
    def search_transcriptions(self, query_text: str, user_id: str = None, document_name: str = None, limit: int = None, agent_id: str = None) -> List[Dict[str, Any]]:
        """Search for similar transcriptions/documents using the vector store"""
        if not self.client:
            logger.warning("Qdrant client not initialized. Cannot search transcriptions.")
            return []
            
        try:
            limit = limit or self.settings.get("match_count", 10)
            similarity_threshold = self.settings.get("similarity_threshold", 0.2)
            logger.info(f"Searching for similar transcriptions to: '{query_text}'")
            
            # Generate embedding for the query text
            embedding = self.generate_embedding(query_text)
            if not embedding:
                logger.warning("Failed to generate embedding for query")
                return []
                
            # Convert numpy arrays to Python lists if needed
            embedding = self._ensure_list_embedding(embedding)
            
            # Build the filter condition
            filter_conditions = []
            
            # Always filter by user_id if provided
            if user_id:
                filter_conditions.append(
                    qmodels.FieldCondition(
                        key="user_id",
                        match=qmodels.MatchValue(value=user_id)
                    )
                )
                
            # Filter by document_name if provided
            if document_name:
                filter_conditions.append(
                    qmodels.FieldCondition(
                        key="document_name",
                        match=qmodels.MatchValue(value=document_name)
                    )
                )
            
            # Try search first without agent filter to see if we get any results
            search_filter = None
            if filter_conditions:
                search_filter = qmodels.Filter(
                    must=filter_conditions
                )
            
            # Use a standard collection name for all documents
            collection_name = "transcription_embeddings"
            
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if collection_name not in collection_names:
                logger.warning(f"Collection {collection_name} does not exist")
                return []
            
            # First try a more forgiving search without agent_id filter
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=embedding,
                limit=limit * 2,  # Get more results to filter later
                query_filter=search_filter,
                with_payload=True,
                score_threshold=0.05  # Very low threshold to catch more potential matches
            )
            
            logger.info(f"Initial search found {len(search_results)} results with threshold 0.05")
            
            # Post-filter by agent_id if provided
            filtered_results = []
            if agent_id:
                logger.info(f"Post-filtering results by agent_id: {agent_id}")
                filtered_results = [
                    r for r in search_results
                    if r.payload and r.payload.get("agent_id") == agent_id
                ]
                logger.info(f"After agent filtering: {len(filtered_results)} results remain")
            else:
                filtered_results = search_results
            
            # If we found results, format and return them
            if filtered_results:
                # Apply actual similarity threshold
                final_results = [r for r in filtered_results if r.score >= similarity_threshold]
                if final_results:
                    logger.info(f"Returning {len(final_results)} results after threshold filtering")
                    return self._format_results(final_results)
                else:
                    # If no results remain after threshold filtering, take the best matches up to the limit
                    final_results = sorted(filtered_results, key=lambda x: x.score, reverse=True)[:limit]
                    logger.info(f"Using top {len(final_results)} results despite being below threshold")
                    return self._format_results(final_results)
            
            # If no results with vector search, try text search
            logger.warning(f"No results found with vector search, retrying with lower threshold: 0.05")
            search_results = self.client.search(
                collection_name=collection_name,
                query_vector=embedding,
                limit=limit * 2,
                query_filter=search_filter,
                with_payload=True,
                score_threshold=0.05
            )
            
            if search_results:
                # Apply agent filter again
                if agent_id:
                    filtered_results = [
                        r for r in search_results
                        if r.payload and r.payload.get("agent_id") == agent_id
                    ]
                else:
                    filtered_results = search_results
                    
                if filtered_results:
                    logger.info(f"Found {len(filtered_results)} results with very low threshold")
                    return self._format_results(filtered_results)
            
            # Fall back to text search as a last resort
            logger.warning("No vector search results found, trying text search")
            return self.search_by_text(query_text, user_id, document_name, limit, agent_id)
            
        except Exception as e:
            logger.error(f"Error searching transcriptions: {e}")
            return []
            
    def search_by_text(self, query_text: str, user_id: str = None, document_name: str = None, limit: int = None, agent_id: str = None) -> List[Dict[str, Any]]:
        """Search for documents by exact text matching rather than vector similarity"""
        if not self.client:
            logger.warning("Qdrant client not initialized. Cannot search documents.")
            return []
            
        try:
            # Clean up the query text - normalize to make searching more robust
            # Convert to lowercase for case-insensitive matching
            original_query = query_text
            cleaned_query = query_text.lower().strip()
            
            # Set a reasonable default limit
            if not limit:
                limit = self.settings.get("match_count", 10)
                
            # Build filter conditions - only apply user_id and document_name filters, not agent_id yet
            filter_conditions = []
                
            # Add user_id filter if provided
            if user_id:
                filter_conditions.append(
                    qmodels.FieldCondition(
                        key="user_id", 
                        match=qmodels.MatchValue(value=user_id)
                    )
                )
                
            # Add document name filter if provided
            if document_name:
                filter_conditions.append(
                    qmodels.FieldCondition(
                        key="document_name", 
                        match=qmodels.MatchValue(value=document_name)
                    )
                )
                
            # Log that we're using agent_id filtering, but don't apply it yet
            if agent_id:
                logger.info(f"Will post-filter results by agent_id: {agent_id}")
            
            # Combine filters with AND logic
            search_filter = None
            if filter_conditions:
                search_filter = qmodels.Filter(
                    must=filter_conditions
                )
            
            # Use a standard collection name for all documents
            collection_name = "transcription_embeddings"
            
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if collection_name not in collection_names:
                logger.warning(f"Collection {collection_name} does not exist")
                return []
            
            # First retrieve all documents matching the basic metadata filters (without agent_id)
            all_potential_matches = self.client.scroll(
                collection_name=collection_name,
                limit=100,  # Retrieve a larger set for text filtering
                scroll_filter=search_filter,
                with_payload=True,
            )[0]  # Get just the points, not the next_page_offset
            
            logger.info(f"Retrieved {len(all_potential_matches)} potential matches without agent_id filtering")
            
            # Print a few examples to debug
            if len(all_potential_matches) > 0:
                sample = all_potential_matches[0]
                logger.info(f"Sample document payload: {sample.payload}")
                logger.info(f"Looking for content containing: '{cleaned_query}'")
            
            # Now perform client-side text matching for more flexibility
            exact_matches = []
            for point in all_potential_matches:
                content = point.payload.get("content", "").lower()  # Case-insensitive matching
                if cleaned_query in content:  # Partial string matching
                    # If we found a match, check if it's for the requested agent_id
                    point_agent_id = point.payload.get("agent_id")
                    if agent_id and point_agent_id != agent_id:
                        logger.info(f"Found match but agent_id doesn't match: found={point_agent_id}, requested={agent_id}")
                        continue
                    exact_matches.append(point)
            
            logger.info(f"Found {len(exact_matches)} text matches after agent_id filtering")
            
            # If no exact matches, try keyword matching
            if not exact_matches and len(cleaned_query.split()) > 2:
                # Try with keywords instead
                # Use longer words (>3 chars) as keywords, keeping them lowercase for case-insensitive matching
                keywords = [word for word in cleaned_query.split() if len(word) > 3]
                if keywords:
                    logger.info(f"No exact matches, trying with keywords: {keywords}")
                    
                    keyword_matches = []
                    for point in all_potential_matches:
                        content = point.payload.get("content", "").lower()
                        # Match if any keyword is in the content
                        if any(keyword in content for keyword in keywords):
                            # Apply agent_id filter
                            point_agent_id = point.payload.get("agent_id")
                            if agent_id and point_agent_id != agent_id:
                                continue
                            keyword_matches.append(point)
                    
                    logger.info(f"Found {len(keyword_matches)} keyword matches after agent_id filtering")
                    return self._format_results(keyword_matches[:limit])  # Respect the limit parameter
            
            # Return exact matches, respecting the limit
            return self._format_results(exact_matches[:limit])
            
        except Exception as e:
            logger.error(f"Error performing text search: {e}")
            return []
    
    def _format_results(self, search_results):
        """Format search results from Qdrant into a standardized format"""
        results = []
        for point in search_results:
            # Extract the original string ID if available, otherwise use the point ID
            point_string_id = point.payload.get("string_id", str(point.id))
            content = point.payload.get("content", "")
            
            results.append({
                "id": point_string_id,
                "content": content,
                "metadata": point.payload,
                "score": point.score,
                "similarity": point.score  # For compatibility with text search results
            })
            
        return results
        
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection from Qdrant"""
        if not self.client:
            logger.warning("Qdrant client not initialized. Cannot delete collection.")
            return False
            
        try:
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            return False
            
    def delete_document_embeddings(self, document_name: str) -> bool:
        """Delete all embeddings for a document"""
        collection_name = sanitize_collection_name(document_name)
        return self.delete_collection(collection_name)

    def _check_and_restore_collection(self, collection_name: str) -> None:
        """Check if collection exists but is empty while documents exist in PostgreSQL"""
        if not self.client:
            return
            
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if collection_name not in collection_names:
                logger.warning(f"Collection {collection_name} doesn't exist, skipping restoration check")
                return
                
            # Check if collection is empty
            try:
                # Get collection info
                collection_info = self.client.get_collection(collection_name=collection_name)
                points_count = getattr(collection_info, "points_count", 0)
                
                if points_count == 0:
                    # Collection is empty, check if we have documents in PostgreSQL
                    logger.warning(f"Collection {collection_name} is empty, checking PostgreSQL for documents")
                    db = get_pg_db()
                    
                    # Check if we have any documents in PostgreSQL
                    query = "SELECT COUNT(*) as count FROM vectors.transcription_embeddings"
                    result = db.execute_query(query)
                    
                    if result and result[0]["count"] > 0:
                        logger.info(f"Found {result[0]['count']} documents in PostgreSQL, restoring embeddings")
                        self.restore_embeddings()
            except Exception as e:
                logger.error(f"Error checking collection points: {e}")
                
        except Exception as e:
            logger.error(f"Error checking collection status: {e}")

    def restore_embeddings(self) -> Dict[str, Any]:
        """Restore embeddings from PostgreSQL records"""
        if not self.client:
            return {"status": "error", "message": "Qdrant client not initialized"}
            
        try:
            # Connect to database
            db = get_pg_db()
            
            # Get all document records from PostgreSQL
            query = """
            SELECT id, document_name, user_id, agent_id, content, metadata 
            FROM vectors.transcription_embeddings
            ORDER BY document_name, id
            """
            
            records = db.execute_query(query)
            
            if not records:
                return {"status": "info", "message": "No documents found in PostgreSQL to restore"}
                
            logger.info(f"Found {len(records)} documents to restore in PostgreSQL")
            
            # Collection name
            collection_name = "transcription_embeddings"
            
            # Ensure collection exists
            self.ensure_collection_exists(collection_name)
            
            # Batch process documents
            processed = 0
            failed = 0
            
            for record in records:
                try:
                    # Extract document information
                    string_id = record["id"]
                    document_name = record["document_name"]
                    user_id = record["user_id"]
                    agent_id = record["agent_id"]
                    content = record["content"]
                    metadata_json = record["metadata"]
                    
                    # Extract chunk index from string_id (format: document_name_index)
                    parts = string_id.split('_')
                    chunk_index = int(parts[-1]) if parts[-1].isdigit() else 0
                    
                    # Process metadata - handle various forms it might be in
                    if isinstance(metadata_json, dict):
                        # It's already a dictionary
                        metadata = metadata_json
                    elif isinstance(metadata_json, str) and metadata_json:
                        # It's a JSON string
                        try:
                            metadata = json.loads(metadata_json)
                        except json.JSONDecodeError:
                            metadata = {}
                    else:
                        # It's None or something else
                        metadata = {}
                    
                    # Update metadata with required fields
                    metadata.update({
                        "user_id": user_id,
                        "document_name": document_name,
                        "chunk_index": chunk_index,
                        "agent_id": agent_id,
                        "string_id": string_id,
                        "embedding_model": self.embedding_model
                    })
                    
                    # Generate embedding
                    embedding = self.generate_embedding(content)
                    
                    if not embedding:
                        logger.warning(f"Failed to generate embedding for document {document_name}, chunk {chunk_index}")
                        failed += 1
                        continue
                    
                    # Convert numpy arrays to Python lists if needed
                    embedding = self._ensure_list_embedding(embedding)
                    
                    # Create UUID from string for Qdrant
                    point_id = uuid.uuid5(uuid.NAMESPACE_DNS, string_id)
                    
                    # Store in Qdrant
                    self.client.upsert(
                        collection_name=collection_name,
                        points=[
                            qmodels.PointStruct(
                                id=str(point_id),  # Convert UUID to string
                                vector=embedding,
                                payload=metadata
                            )
                        ]
                    )
                    
                    processed += 1
                    
                    # Log progress every 10 documents
                    if processed % 10 == 0:
                        logger.info(f"Restored {processed}/{len(records)} documents")
                        
                except Exception as e:
                    logger.error(f"Error restoring embedding for document {record.get('document_name')}: {e}")
                    failed += 1
            
            logger.info(f"Completed restoration: {processed} documents restored, {failed} failed")
            return {
                "status": "success", 
                "message": f"Restored {processed} documents, {failed} failed",
                "processed": processed,
                "failed": failed
            }
                
        except Exception as e:
            logger.error(f"Error restoring embeddings: {e}")
            return {"status": "error", "message": f"Error restoring embeddings: {e}"}

    def get_collections_info(self) -> Dict[str, Any]:
        """Get information about all collections in Qdrant"""
        if not self.client:
            return {
                "status": "error", 
                "message": "Qdrant client not initialized",
                "collections": []
            }
            
        try:
            # Get all collections
            collections_response = self.client.get_collections()
            collections = collections_response.collections
            
            result = []
            
            # Get detailed info for each collection
            for collection in collections:
                try:
                    collection_info = self.client.get_collection(collection_name=collection.name)
                    
                    # Extract relevant info
                    info = {
                        "name": collection.name,
                        "vectors_count": getattr(collection_info, "vectors_count", 0),
                        "points_count": getattr(collection_info, "points_count", 0),
                        "indexed_vectors_count": getattr(collection_info, "indexed_vectors_count", 0),
                        "status": "active"
                    }
                    
                    # Add to results
                    result.append(info)
                except Exception as e:
                    # If we can't get detailed info, still include the collection with error
                    logger.error(f"Error getting collection info for {collection.name}: {e}")
                    result.append({
                        "name": collection.name,
                        "status": "error",
                        "error": str(e)
                    })
            
            # Get primary collection name - the one being used for storage
            primary_collection = "transcription_embeddings"
            
            return {
                "status": "success",
                "collections": result,
                "primary_collection": primary_collection
            }
                
        except Exception as e:
            logger.error(f"Error getting collections info: {e}")
            return {
                "status": "error", 
                "message": f"Error getting collections info: {e}",
                "collections": []
            }

    def reset_collections(self, skip_reindex: bool = False) -> Dict[str, Any]:
        """Delete and recreate all collections to start fresh
        
        Args:
            skip_reindex: If True, doesn't try to reindex after recreating collections
            
        Returns:
            Dictionary with status information
        """
        if not self.client:
            return {"status": "error", "message": "Qdrant client not initialized"}
        
        try:
            # Get list of collections first
            collections_response = self.client.get_collections()
            collection_names = [collection.name for collection in collections_response.collections]
            
            results = {}
            
            # Delete each collection
            for name in collection_names:
                try:
                    logger.info(f"Deleting collection {name}")
                    self.client.delete_collection(collection_name=name)
                    results[name] = "deleted"
                except Exception as e:
                    logger.error(f"Error deleting collection {name}: {e}")
                    results[name] = f"error: {str(e)}"
            
            # Create main collection
            if not skip_reindex:
                logger.info("Creating transcription_embeddings collection")
                self._ensure_collection_exists("transcription_embeddings")
                results["transcription_embeddings"] = "created"
            
            return {
                "status": "success",
                "message": "Collections reset successfully",
                "results": results
            }
        except Exception as e:
            logger.error(f"Error resetting collections: {e}")
            return {"status": "error", "message": f"Error: {str(e)}"}

    def get_all_vectors(self, collection_name="transcription_embeddings", limit=50) -> List[Dict[str, Any]]:
        """Get all vectors from a collection for diagnostic purposes"""
        if not self.client:
            logger.warning("Qdrant client not initialized.")
            return []
            
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if collection_name not in collection_names:
                logger.warning(f"Collection {collection_name} does not exist")
                return []
                
            # Get all points using scroll
            points = self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False  # Don't include vectors to save bandwidth
            )
            
            # Format the results
            formatted_points = []
            for point in points[0]:
                # Extract the payload
                payload = point.payload if hasattr(point, "payload") else {}
                
                # Format the point
                formatted_point = {
                    "id": point.id,
                    "payload": payload
                }
                
                # Add to results
                formatted_points.append(formatted_point)
                
            return formatted_points
        except Exception as e:
            logger.error(f"Error getting vectors: {e}")
            return []
            
    def brute_force_search(self, query_text: str, agent_id: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for vectors in the collection by ignoring similarity and filtering only on agent ID"""
        if not self.client:
            logger.warning("Qdrant client not initialized.")
            return []
            
        try:
            logger.info(f"Performing brute force search for agent_id: {agent_id}")
            
            # First get all points in the collection
            all_points = self.get_all_vectors(limit=100)  # Get up to 100 points
            
            logger.info(f"Found {len(all_points)} total points in collection")
            
            # Filter by agent_id if provided
            if agent_id:
                # Debug all existing agent_ids
                agent_ids = set(p["payload"].get("agent_id") for p in all_points if p["payload"].get("agent_id"))
                logger.info(f"All agent_ids in collection: {agent_ids}")
                
                # Show everything with agent_id
                for point in all_points:
                    if "agent_id" in point["payload"]:
                        logger.info(f"Point {point['id']} has agent_id: {point['payload']['agent_id']}")
                
                filtered_points = [p for p in all_points if p["payload"].get("agent_id") == agent_id]
                logger.info(f"After filtering by agent_id {agent_id}: {len(filtered_points)} points remain")
            else:
                filtered_points = all_points
                
            # Format results similarly to search function
            formatted_results = []
            for point in filtered_points[:limit]:  # Limit to requested number
                formatted_results.append({
                    "id": point["id"],
                    "chunk_index": point["payload"].get("chunk_index", 0),
                    "content": point["payload"].get("content", ""),
                    "score": 1.0,  # No actual similarity score, so use 1.0
                    "document_name": point["payload"].get("document_name", ""),
                    "agent_id": point["payload"].get("agent_id"),
                    "metadata": point["payload"]
                })
                
            return formatted_results
        except Exception as e:
            logger.error(f"Error in brute_force_search: {e}")
            return []

# Function to get the singleton instance of the vector store
def get_vector_store(embedding_model="all-MiniLM-L6-v2") -> FlameVectorStore:
    """Get the singleton instance of the vector store"""
    global _vector_store
    
    if _vector_store is None:
        try:
            _vector_store = FlameVectorStore(embedding_model=embedding_model)
            logger.info("Initialized FlameVectorStore with Qdrant backend")
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            raise
    
    return _vector_store

# Singleton instance
_model_cache = {}

def generate_text_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> List[float]:
    """Generate an embedding for a piece of text using a sentence transformer"""
    global _model_cache
    
    # Fix model name format if it's not in the correct format
    if model_name == "MiniLM L6":
        model_name = "all-MiniLM-L6-v2"
    elif model_name == "MPNet":
        model_name = "all-mpnet-base-v2"
    elif model_name == "Multilingual":
        model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    
    # Check if the model is already loaded
    if model_name not in _model_cache:
        try:
            # Log the model being loaded
            logger.info(f"Loading embedding model: {model_name}")
            
            # Load the model
            from sentence_transformers import SentenceTransformer
            _model_cache[model_name] = SentenceTransformer(model_name)
            
        except Exception as e:
            # Log the error
            logger.error(f"Error loading embedding model {model_name}: {e}")
            
            # Try loading a fallback model
            fallback_model = "all-MiniLM-L6-v2"
            if model_name != fallback_model:
                try:
                    logger.warning(f"Attempting to load fallback model {fallback_model}")
                    _model_cache[model_name] = SentenceTransformer(fallback_model)
                except Exception as fallback_e:
                    logger.error(f"Error loading fallback model {fallback_model}: {fallback_e}")
                    _model_cache[model_name] = None
            else:
                _model_cache[model_name] = None
    
    # Generate the embedding
    try:
        if _model_cache[model_name] is not None:
            # Generate the embedding and convert to list
            embedding = _model_cache[model_name].encode(text)
            return embedding.tolist()
        else:
            # If model failed to load, return an empty embedding
            dimensions = {
                "all-MiniLM-L6-v2": 384,
                "all-mpnet-base-v2": 768,
                "paraphrase-multilingual-MiniLM-L12-v2": 384
            }
            dimension = dimensions.get(model_name, 384)
            logger.warning(f"Returning empty embedding with {dimension} dimensions")
            return [0.0] * dimension
    except Exception as e:
        # Log the error
        logger.error(f"Error generating embedding: {e}")
        
        # Return an empty embedding
        dimensions = {
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768,
            "paraphrase-multilingual-MiniLM-L12-v2": 384
        }
        dimension = dimensions.get(model_name, 384)
        logger.warning(f"Returning empty embedding with {dimension} dimensions")
        return [0.0] * dimension
