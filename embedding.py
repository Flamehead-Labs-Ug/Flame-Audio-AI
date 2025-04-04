import os
import logging
from typing import List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize global variables
_sentence_transformer = None
_current_model_name = None

def generate_embedding(text: str, model_name: str = "all-MiniLM-L6-v2") -> List[float]:
    """Generate an embedding for a text string
    
    Args:
        text (str): The text to embed
        model_name (str, optional): The model to use for embedding. Defaults to "all-MiniLM-L6-v2".
    
    Returns:
        List[float]: The embedding vector
    """
    logger = logging.getLogger(__name__)
    
    # Make the global variables accessible
    global _sentence_transformer
    global _current_model_name
    
    try:
        # Initialize or re-initialize model if needed
        if _sentence_transformer is None or _current_model_name != model_name:
            logger.info(f"Loading embedding model: {model_name}")
            try:
                from sentence_transformers import SentenceTransformer
                _sentence_transformer = SentenceTransformer(model_name)
                _current_model_name = model_name
                logger.info(f"Successfully loaded model: {model_name}")
            except ImportError:
                logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
                raise
        
        # Process text into embedding
        embedding = _sentence_transformer.encode(text)
        logger.info(f"Generated embedding with model {_current_model_name}, dimensions: {len(embedding)}")
        return embedding.tolist()
    except Exception as e:
        logger.warning(f"Error using sentence-transformers: {e}. Falling back to alternatives.")
        
        # Fallback to OpenAI if available and requested
        if model_name == "text-embedding-ada-002" or "openai" in model_name:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    import openai
                    openai.api_key = api_key
                    
                    response = openai.Embedding.create(
                        input=[text],
                        model="text-embedding-ada-002"
                    )
                    
                    embedding = response['data'][0]['embedding']
                    logger.info(f"Generated embedding with OpenAI model, dimensions: {len(embedding)}")
                    return embedding
                except Exception as e:
                    logger.error(f"Error generating OpenAI embedding: {e}")
        
        # If no embedding could be generated, return zero vector with appropriate dimensions
        dimensions = 384  # Default MiniLM-L6 dimensions
        
        # Use appropriate dimensions based on requested model
        if model_name == "all-mpnet-base-v2":
            dimensions = 768
        elif model_name == "text-embedding-ada-002":
            dimensions = 1536
        
        logger.error(f"Returning zero vector with {dimensions} dimensions as fallback")
        return [0.0] * dimensions

def list_embedding_models():
    """List available embedding models with their details
    
    Returns:
        List[Dict]: List of embedding models with their details
    """
    # Define available embedding models
    models = [
        {
            "id": "all-MiniLM-L6-v2",
            "name": "MiniLM L6",
            "description": "Small & fast model (384 dimensions)",
            "dimensions": 384,
            "framework": "sentence-transformers",
            "recommended": True
        },
        {
            "id": "all-mpnet-base-v2",
            "name": "MPNet Base",
            "description": "High quality model (768 dimensions)",
            "dimensions": 768,
            "framework": "sentence-transformers",
            "recommended": False
        },
        {
            "id": "paraphrase-multilingual-MiniLM-L12-v2",
            "name": "Multilingual MiniLM",
            "description": "Multilingual support (384 dimensions)",
            "dimensions": 384,
            "framework": "sentence-transformers",
            "recommended": False
        },
        {
            "id": "text-embedding-ada-002",
            "name": "OpenAI Ada 002",
            "description": "OpenAI's embedding model (requires API key)",
            "dimensions": 1536,
            "framework": "openai",
            "recommended": False,
            "requires_api_key": True
        }
    ]
    
    return models
