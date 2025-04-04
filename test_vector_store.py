"""Test script for vector store document retrieval"""

import os
import sys
import json
import uuid
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import vector store implementation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from database.vector_store import get_vector_store, FlameVectorStore
from database.pg_connector import get_pg_db
from embedding import generate_embedding


def check_environment():
    """Check if the environment is properly set up"""
    # Check if the vectors schema and table exist
    db = get_pg_db()
    
    # Check schemas
    schema_query = """SELECT schema_name FROM information_schema.schemata WHERE schema_name = 'vectors';"""
    result = db.execute_query(schema_query)
    
    if not result or len(result) == 0:
        logger.warning("Vectors schema does not exist. Please run the SQL script first.")
        return False
    
    # Check table
    table_query = """SELECT table_name 
                  FROM information_schema.tables 
                  WHERE table_schema = 'vectors' AND table_name = 'transcription_embeddings';"""
    result = db.execute_query(table_query)
    
    if not result or len(result) == 0:
        logger.warning("Transcription embeddings table does not exist. Please run the SQL script first.")
        return False
    
    logger.info("Environment check passed: vectors schema and table exist")
    return True


def create_test_data(vector_store, user_id="test_user"):
    """Create test data with Eminem/Slim Shady content"""
    # Generate a unique job ID for our test
    job_id = str(uuid.uuid4())
    logger.info(f"Creating test data with job_id: {job_id}")
    
    # Test content about Slim Shady/Eminem
    test_contents = [
        "Slim Shady is the alter ego of the rapper Eminem, created in 1997.",
        "Marshall Bruce Mathers III, known professionally as Eminem, is an American rapper.",
        "The Slim Shady LP was released in 1999 and features the hit song 'My Name Is'.",
        "Eminem's alter ego Slim Shady is known for his controversial lyrics and persona.",
        "The character of Slim Shady represents the dark side of Eminem's personality.",
        "Eminem released a single called 'The Real Slim Shady' in 2000.",
        "The Eminem Show is the fourth studio album by American rapper Eminem.",
        "Eminem's 8 Mile is a movie based on his life in Detroit.",
        "Rap God is a hit song by Eminem released in 2013.",
        "Eminem has won multiple Grammy awards throughout his career."
    ]
    
    # Store test data directly in the database first
    db = get_pg_db()
    
    # Check if we have access to insert directly
    try:
        # First try a direct database insert for testing
        for i, content in enumerate(test_contents):
            # Create metadata
            metadata = {
                "title": f"Test Document {i+1}",
                "chunk_index": i
            }
            
            # Generate embedding
            embedding_vector = generate_embedding(content)
            
            # Generate a unique ID for each record (must be a valid UUID for Supabase)
            record_id = str(uuid.uuid4())
            
            # Insert query
            insert_query = """
            INSERT INTO vectors.transcription_embeddings (id, user_id, job_id, agent_id, content, metadata, embedding) 
            VALUES (%(id)s, %(user_id)s, %(job_id)s, %(agent_id)s, %(content)s, %(metadata)s, %(embedding)s)
            ON CONFLICT (id) DO UPDATE 
            SET content = %(content)s, 
                user_id = %(user_id)s,
                job_id = %(job_id)s,
                agent_id = %(agent_id)s,
                metadata = %(metadata)s, 
                embedding = %(embedding)s
            """
            
            db.execute_query(
                insert_query, 
                {
                    "id": record_id, 
                    "user_id": user_id,
                    "job_id": job_id,
                    "agent_id": "test_agent",
                    "content": content, 
                    "metadata": json.dumps(metadata),
                    "embedding": embedding_vector
                }
            )
            
            logger.info(f"Stored test document {i+1}/{len(test_contents)} with ID: {record_id}")
    except Exception as e:
        logger.warning(f"Direct database insert failed: {e}")
        
        # Try using the vector store implementation instead
        for i, content in enumerate(test_contents):
            try:
                # Create metadata
                metadata = {
                    "user_id": user_id,
                    "job_id": job_id,
                    "title": f"Test Document {i+1}",
                    "agent_id": "test_agent"
                }
                
                # Store with embedding
                result = vector_store.store_embedding(
                    user_id=user_id,
                    job_id=job_id,
                    chunk_index=i,
                    content=content,
                    metadata=metadata
                )
                
                logger.info(f"Stored embedding {i+1}/{len(test_contents)} with ID: {result['id']}")
                
            except Exception as e:
                logger.error(f"Error storing test data through vector store: {e}")
    
    # Check if data was actually stored
    check_query = """
    SELECT COUNT(*) as count FROM vectors.transcription_embeddings 
    WHERE job_id = %(job_id)s
    """
    result = db.execute_query(check_query, {"job_id": job_id})
    
    if result and result[0].get("count", 0) > 0:
        logger.info(f"Successfully stored {result[0]['count']} test documents")
    else:
        logger.warning("No test documents were stored!")
    
    return job_id


def test_search(vector_store, user_id, job_id):
    """Test different search queries against the vector store"""
    test_queries = [
        "slim shady",
        "eminem alter ego",
        "controversial lyrics",
        "marshall mathers",
        "rap god"
    ]
    
    for query in test_queries:
        logger.info(f"\n\n=== Testing query: '{query}' ===")
        
        # 1. Test with original vector store search
        logger.info("--- Vector store search results ---")
        results = vector_store.search_similar_transcriptions(
            query_text=query,
            user_id=user_id,
            job_id=job_id,
            limit=10
        )
        
        if results:
            logger.info(f"Found {len(results)} results")
            for i, result in enumerate(results):
                logger.info(f"Result {i+1}: Similarity={result['similarity']:.4f}")
                logger.info(f"Content: {result['content']}")
        else:
            logger.info("No results found with vector search")
        
        # 2. Test with fallback search
        logger.info("\n--- Fallback search results ---")
        results = vector_store._fallback_search(
            query_text=query,
            user_id=user_id,
            job_id=job_id,
            limit=10
        )
        
        if results:
            logger.info(f"Found {len(results)} results")
            for i, result in enumerate(results):
                logger.info(f"Result {i+1}: Similarity={result['similarity']:.4f}")
                logger.info(f"Content: {result['content']}")
        else:
            logger.info("No results found with fallback search")
        
        # 3. Test with direct database query (fallback fallback)
        logger.info("\n--- Direct database query results ---")
        try:
            db = get_pg_db()
            query_text = f"%{query}%"
            direct_query = """
            SELECT id, content, metadata FROM vectors.transcription_embeddings
            WHERE metadata->>'job_id' = %(job_id)s
            AND content ILIKE %(query)s
            LIMIT 5
            """
            
            result = db.execute_query(direct_query, {"job_id": job_id, "query": query_text})
            
            if result and len(result) > 0:
                logger.info(f"Found {len(result)} results via direct query")
                for i, row in enumerate(result):
                    logger.info(f"Result {i+1}: ID={row.get('id')}")
                    logger.info(f"Content: {row.get('content')}")
            else:
                logger.info("No results found with direct database query")
        except Exception as e:
            logger.error(f"Direct database query failed: {e}")


def main():
    """Main test function"""
    try:
        # Check environment first
        if not check_environment():
            logger.error("Environment check failed. Please run the SQL script first.")
            return
        
        # Initialize vector store
        logger.info("Initializing vector store...")
        vector_store = get_vector_store()
        
        # User ID for testing
        user_id = "test_user_" + str(uuid.uuid4())[:8]
        
        # Create test data
        job_id = create_test_data(vector_store, user_id)
        
        # Test search functionality
        test_search(vector_store, user_id, job_id)
        
        logger.info("Tests completed.")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")


if __name__ == "__main__":
    main()
