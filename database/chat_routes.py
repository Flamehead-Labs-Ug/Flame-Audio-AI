import os
import json
import time
import logging
from datetime import datetime

from typing import Dict, List, Optional, Any, Union, TypedDict, Annotated, Literal
from fastapi import APIRouter, Depends, HTTPException, status, Request, BackgroundTasks
from pydantic import BaseModel, Field

from langchain.globals import set_verbose
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage
import langchain_core.documents
from langchain_core.documents import Document
from langchain_community.chat_message_histories import PostgresChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_groq import ChatGroq

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Define state types for LangGraph
class DocumentChatState(TypedDict):
    messages: List[Union[SystemMessage, HumanMessage, AIMessage]]
    context: str

class GeneralChatState(TypedDict):
    messages: List[Union[SystemMessage, HumanMessage, AIMessage]]

# Local imports
from .pg_connector import get_pg_db
from .vector_store import get_vector_store
from authentication.auth import get_current_user, User
from embedding import generate_embedding

# Initialize router
router = APIRouter(prefix="/chat", tags=["chat"])

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models for request/response
class ChatSessionCreate(BaseModel):
    agent_id: str = None
    document_id: str = None
    title: str = "New Chat"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chat_parameters: Dict[str, Any] = Field(default_factory=dict)

class ChatMessage(BaseModel):
    role: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    message: str
    agent_id: Optional[str] = None
    document_id: Optional[str] = None
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ChatResponse(BaseModel):
    session_id: str
    response: str
    messages: List[ChatMessage]
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ChatModelInfo(BaseModel):
    id: str
    name: str
    description: str = None
    max_tokens: int = None
    is_chat_model: bool = True

# Endpoints
@router.get("/models")
async def get_chat_models(current_user: User = Depends(get_current_user)):
    """Get available chat models from Groq API"""
    try:
        # Use the GROQ_API_KEY from environment
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="GROQ_API_KEY not found in environment variables"
            )
        
        # Configure base URL for Groq API
        BASE_URL = "https://api.groq.com/openai/v1"
        
        # Make API request to fetch models
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/models",
                headers={"Authorization": f"Bearer {groq_api_key}"}
            )
            response.raise_for_status()
            models_data = response.json()
            
            # Filter for chat models (LLaMA, Mixtral, etc.)
            chat_models = [
                {
                    "id": model["id"],
                    "name": model["id"].replace("-", " ").title(),
                    "description": model.get("description", "Chat model"),
                    "is_chat_model": True
                }
                for model in models_data["data"]
                if any(name in model["id"].lower() for name in ["llama", "mixtral", "gemma"]) and not "instruct" in model["id"].lower()
            ]
            
            return chat_models
    except httpx.HTTPError as e:
        logger.error(f"HTTP error fetching models: {str(e)}")
        raise HTTPException(
            status_code=e.response.status_code if hasattr(e, 'response') else 500,
            detail=f"Error fetching models from Groq API: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in get_chat_models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch chat models: {str(e)}"
        )

@router.post("/sessions")
async def create_chat_session(session_data: ChatSessionCreate, current_user: User = Depends(get_current_user)):
    """Create a new chat session"""
    try:
        db = get_pg_db()
        
        # Validate agent_id if provided
        if session_data.agent_id:
            agent_query = """
            SELECT id FROM user_data.user_agents 
            WHERE id = %(agent_id)s AND user_id = %(user_id)s
            """
            agent_result = db.execute_query(agent_query, {
                "agent_id": session_data.agent_id,
                "user_id": current_user.id
            })
            
            if not agent_result:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Agent with ID {session_data.agent_id} not found"
                )
                
        # Validate document_id if provided
        if session_data.document_id:
            doc_query = """
            SELECT id FROM user_data.audio_jobs 
            WHERE id = %(doc_id)s AND user_id = %(user_id)s
            """
            doc_result = db.execute_query(doc_query, {
                "doc_id": session_data.document_id,
                "user_id": current_user.id
            })
            
            if not doc_result:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Document with ID {session_data.document_id} not found"
                )
        
        # Create chat session
        insert_query = """
        INSERT INTO user_data.chat_sessions
            (user_id, agent_id, document_id, title, metadata, chat_parameters)
        VALUES
            (%(user_id)s, %(agent_id)s, %(document_id)s, %(title)s, %(metadata)s::jsonb, %(parameters)s::jsonb)
        RETURNING id, created_at
        """
        
        result = db.execute_query(insert_query, {
            "user_id": current_user.id,
            "agent_id": session_data.agent_id,
            "document_id": session_data.document_id,
            "title": session_data.title,
            "metadata": json.dumps(session_data.metadata),
            "parameters": json.dumps(session_data.chat_parameters)
        })
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create chat session"
            )
            
        return {
            "session_id": result[0]["id"],
            "created_at": result[0]["created_at"],
            "title": session_data.title,
            "agent_id": session_data.agent_id,
            "document_id": session_data.document_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error creating chat session: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create chat session: {str(e)}"
        )

@router.get("/sessions")
async def get_user_chat_sessions(agent_id: str = None, current_user: User = Depends(get_current_user)):
    """Get all chat sessions for the current user, optionally filtered by agent"""
    try:
        db = get_pg_db()
        
        # Build query based on whether agent_id is provided
        query = """
        SELECT cs.id, cs.title, cs.agent_id, cs.document_id, cs.created_at, cs.updated_at,
               a.name as agent_name, aj.file_name as document_name
        FROM user_data.chat_sessions cs
        LEFT JOIN user_data.user_agents a ON cs.agent_id = a.id
        LEFT JOIN user_data.audio_jobs aj ON cs.document_id = aj.id
        WHERE cs.user_id = %(user_id)s
        """
        
        params = {"user_id": current_user.id}
        
        if agent_id:
            query += " AND cs.agent_id = %(agent_id)s"
            params["agent_id"] = agent_id
            
        query += " ORDER BY cs.updated_at DESC"
        
        result = db.execute_query(query, params)
        return result or []
        
    except Exception as e:
        logging.error(f"Error fetching chat sessions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch chat sessions: {str(e)}"
        )

@router.get("/sessions/{session_id}")
async def get_chat_session(session_id: str, current_user: User = Depends(get_current_user)):
    """Get a specific chat session and its messages"""
    try:
        db = get_pg_db()
        
        # Get session info
        session_query = """
        SELECT cs.id, cs.title, cs.agent_id, cs.document_id, cs.created_at, cs.updated_at, 
               cs.metadata, cs.chat_parameters,
               a.name as agent_name, a.system_message as agent_system_message,
               aj.file_name as document_name
        FROM user_data.chat_sessions cs
        LEFT JOIN user_data.user_agents a ON cs.agent_id = a.id
        LEFT JOIN user_data.audio_jobs aj ON cs.document_id = aj.id
        WHERE cs.id = %(session_id)s AND cs.user_id = %(user_id)s
        """
        
        session_result = db.execute_query(session_query, {
            "session_id": session_id,
            "user_id": current_user.id
        })
        
        if not session_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat session with ID {session_id} not found"
            )
            
        session_info = session_result[0]
        
        # Get messages
        msg_query = """
        SELECT id, role, content, message_timestamp, metadata
        FROM user_data.chat_messages
        WHERE session_id = %(session_id)s
        ORDER BY message_timestamp ASC
        """
        
        messages = db.execute_query(msg_query, {"session_id": session_id})
        
        return {
            "session": session_info,
            "messages": messages or []
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error fetching chat session: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch chat session: {str(e)}"
        )

@router.delete("/sessions/{session_id}")
async def delete_chat_session(session_id: str, current_user: User = Depends(get_current_user)):
    """Delete a chat session and all its messages"""
    try:
        db = get_pg_db()
        
        # Check if session exists and belongs to user
        check_query = """
        SELECT id FROM user_data.chat_sessions 
        WHERE id = %(session_id)s AND user_id = %(user_id)s
        """
        
        check_result = db.execute_query(check_query, {
            "session_id": session_id,
            "user_id": current_user.id
        })
        
        if not check_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Chat session with ID {session_id} not found"
            )
            
        # Delete session (messages will be cascade deleted)
        delete_query = """
        DELETE FROM user_data.chat_sessions 
        WHERE id = %(session_id)s
        """
        
        db.execute_query(delete_query, {"session_id": session_id}, fetch=False)
        
        return {"status": "success", "message": f"Chat session {session_id} deleted"}
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error deleting chat session: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete chat session: {str(e)}"
        )

@router.get("/debug/vector_store")
async def debug_vector_store(current_user: User = Depends(get_current_user)):
    """Debug route to check vector store connection"""
    try:
        # Get vector store instance with proper connection parameters
        db_url = os.getenv("VECTOR_STORE_URL")
        db_key = os.getenv("VECTOR_STORE_SERVICE_KEY")
        vector_store = get_vector_store(db_url, db_key)
        
        # Run debug function
        debug_info = vector_store.debug_vector_store()
        
        return {"status": "success", "debug_info": debug_info}
    except Exception as e:
        logging.error(f"Error debugging vector store: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to debug vector store: {str(e)}")

@router.post("/", response_model=ChatResponse)
async def chat_with_agent(chat_request: ChatRequest, background_tasks: BackgroundTasks, current_user: User = Depends(get_current_user)):
    """Chat with an agent, optionally focused on a specific document"""
    try:
        db = get_pg_db()
        session_id = chat_request.session_id
        agent_id = chat_request.agent_id
        document_id = chat_request.document_id
        
        # Initialize variables to track session data
        agent_system_message = None
        chat_parameters = chat_request.parameters or {}
        
        # If no session_id provided, create a new session
        is_new_session = False
        if not session_id:
            is_new_session = True
            # Validate agent_id if provided
            if agent_id:
                agent_query = """
                SELECT id, system_message FROM user_data.user_agents 
                WHERE id = %(agent_id)s AND user_id = %(user_id)s
                """
                agent_result = db.execute_query(agent_query, {
                    "agent_id": agent_id,
                    "user_id": current_user.id
                })
                
                if not agent_result:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Agent with ID {agent_id} not found"
                    )
                    
                agent_system_message = agent_result[0]["system_message"]
                
            # Validate document_id if provided
            doc_name = None
            if document_id:
                doc_query = """
                SELECT file_name as document_name FROM user_data.audio_jobs 
                WHERE id = %(document_id)s AND user_id = %(user_id)s
                """
                doc_result = db.execute_query(doc_query, {
                    "document_id": document_id,
                    "user_id": current_user.id
                })
                
                if not doc_result:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Document with ID {document_id} not found"
                    )
                    
                doc_name = doc_result[0]["document_name"]
                title = f"Chat about {doc_name}"
            else:
                title = "New Chat"
            
            # Create new session
            insert_query = """
            INSERT INTO user_data.chat_sessions
                (user_id, agent_id, document_id, title, metadata, chat_parameters)
            VALUES
                (%(user_id)s, %(agent_id)s, %(document_id)s, %(title)s, %(metadata)s::jsonb, %(parameters)s::jsonb)
            RETURNING id
            """
            
            result = db.execute_query(insert_query, {
                "user_id": current_user.id,
                "agent_id": agent_id,
                "document_id": document_id,
                "title": title,
                "metadata": json.dumps(chat_request.metadata),
                "parameters": json.dumps(chat_parameters)
            })
            
            if not result:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Failed to create chat session"
                )
                
            session_id = result[0]["id"]
        else:
            # Retrieve existing session info
            session_query = """
            SELECT cs.agent_id, cs.document_id, cs.chat_parameters,
                   a.system_message as agent_system_message
            FROM user_data.chat_sessions cs
            LEFT JOIN user_data.user_agents a ON cs.agent_id = a.id
            WHERE cs.id = %(session_id)s AND cs.user_id = %(user_id)s
            """
            
            session_result = db.execute_query(session_query, {
                "session_id": session_id,
                "user_id": current_user.id
            })
            
            if not session_result:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Chat session with ID {session_id} not found"
                )
                
            # Get session parameters and use the agent_id/document_id from session if not provided
            session_data = session_result[0]
            agent_id = agent_id or session_data["agent_id"]
            document_id = document_id or session_data["document_id"]
            
            if session_data["chat_parameters"] and isinstance(session_data["chat_parameters"], dict):
                # Merge parameters, prioritizing new parameters from the request
                for key, value in session_data["chat_parameters"].items():
                    if key not in chat_parameters:
                        chat_parameters[key] = value
                        
            agent_system_message = session_data["agent_system_message"]
            
            # Update session's updated_at timestamp
            update_query = """
            UPDATE user_data.chat_sessions
            SET updated_at = NOW(), chat_parameters = %(parameters)s::jsonb
            WHERE id = %(session_id)s
            """
            
            db.execute_query(update_query, {
                "session_id": session_id,
                "parameters": json.dumps(chat_parameters)
            }, fetch=False)
                
        # If a new session is created and agent has a system message, save it first
        if is_new_session and agent_system_message:
            system_message_query = """
            INSERT INTO user_data.chat_messages
                (session_id, role, content, metadata, embedding, message_timestamp)
            VALUES
                (%(session_id)s, %(role)s, %(content)s, %(metadata)s, %(embedding)s, NOW())
            RETURNING id
            """
            
            # Generate embedding for system message
            system_message_embedding = generate_embedding(agent_system_message)
            
            # Save system message
            db.execute_query(system_message_query, {
                "session_id": session_id,
                "role": "system",
                "content": agent_system_message,
                "metadata": json.dumps({"auto_generated": True}),
                "embedding": system_message_embedding
            })
        
        # Save the user message to the database first
        message_query = """
        INSERT INTO user_data.chat_messages
            (session_id, role, content, metadata, embedding, message_timestamp)
        VALUES
            (%(session_id)s, %(role)s, %(content)s, %(metadata)s, %(embedding)s, NOW())
        RETURNING id
        """
        
        # Generate embedding for user message
        user_message_embedding = generate_embedding(chat_request.message)
        
        # Save user message with metadata
        user_metadata = {
            "agent_id": agent_id,
            "document_id": document_id,
            "parameters": chat_parameters
        }
        
        user_msg_result = db.execute_query(message_query, {
            "session_id": session_id,
            "role": "user",
            "content": chat_request.message,
            "metadata": json.dumps(user_metadata),
            "embedding": user_message_embedding
        })
        
        user_message_id = user_msg_result[0]["id"] if user_msg_result else None
        
        # Initialize LangChain for chat
        # Configure model parameters
        model_name = chat_parameters.get("model", "llama3-70b-8192")
        temperature = float(chat_parameters.get("temperature", 0.7))
        max_tokens = int(chat_parameters.get("max_tokens", 1024))
        top_p = float(chat_parameters.get("top_p", 0.9))
        
        # Set up LLM
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="GROQ_API_KEY not found in environment variables"
            )
            
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        
        # Set up message history from the database
        messages_query = """
        SELECT role, content, message_timestamp FROM user_data.chat_messages
        WHERE session_id = %(session_id)s
        ORDER BY message_timestamp ASC
        """
        
        messages_result = db.execute_query(messages_query, {"session_id": session_id})
        
        chat_history = []
        for msg in messages_result:
            if msg["role"] == "system":
                chat_history.append(SystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                chat_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                chat_history.append(AIMessage(content=msg["content"]))
                
        # Process the message and get response
        response_text = ""
        
        # Check if we need document retrieval
        if document_id:
            # Define a simple direct retrieval function
            def retrieve_document_content(query: str):
                """Retrieve document content directly using the vector store"""
                try:
                    # Generate embedding for the query
                    query_embedding = generate_embedding(query)
                    
                    # Get vector store instance
                    vector_store = get_vector_store()
                    
                    # Search for similar content directly with text
                    similar_documents = vector_store.search_transcriptions(
                        query_text=query,
                        user_id=str(current_user.id),  # Important for security - filter by current user
                        document_name=document_id if document_id else None,
                        agent_id=agent_id,  # Pass the agent_id to the search function
                        limit=3  # Retrieve top 3 results
                    )
                    
                    # Fallbacks removed to ensure only top 3 vector search results are used
                    
                    # Handle case where no results are found
                    if not similar_documents:
                        logger.warning(f"No similar content found for document {document_id}")
                        documents = [{
                            "content": "No relevant content found in the document. Please try a different question.",
                            "metadata": {
                                "job_id": document_id,
                                "chunk_index": 0,
                                "similarity": 0
                            }
                        }]
                        return documents
                    
                    # Convert to simple list of text and metadata
                    documents = []
                    for result in similar_documents:
                        try:
                            # Handle different result formats gracefully
                            content = result.get("content", "")
                            if not content and "text" in result:
                                content = result.get("text", "")
                                
                            documents.append({
                                "content": content,
                                "metadata": {
                                    "job_id": result.get("job_id", document_id),
                                    "chunk_index": result.get("chunk_index", 0),
                                    "similarity": result.get("similarity", 0),
                                    "metadata": result.get("metadata", {})
                                }
                            })
                        except Exception as chunk_error:
                            logger.error(f"Error processing document chunk: {chunk_error}")
                            
                    return documents
                except Exception as e:
                    logger.error(f"Error in document retrieval: {e}")
                    # Return a fallback document with error information
                    return [{
                        "content": "There was an error retrieving relevant content from the document.",
                        "metadata": {
                            "job_id": document_id,
                            "error": str(e)
                        }
                    }]
            
            # Define Document Chat Workflow using LangGraph
            
            # Define nodes for the document chat graph
            def retrieve_context(state: DocumentChatState) -> DocumentChatState:
                """Retrieve relevant context from document"""
                # Use our direct retrieval function
                query = state["messages"][-1].content
                documents = retrieve_document_content(query)
                
                # Handle empty or failed document retrieval
                if not documents or all("error" in doc.get("metadata", {}) for doc in documents):
                    logger.warning(f"No document content found for query: {query}")
                    state["context"] = "No relevant content found in the document. Please try a different question."
                    return state
                
                # Get job metadata to provide better context
                try:
                    # Get document title/details
                    job_ids = list(set([doc["metadata"].get("job_id") for doc in documents if "job_id" in doc.get("metadata", {})]))
                    job_info = {}
                    
                    if job_ids:
                        # Get job information to enhance context
                        job_query = """
                        SELECT id, file_name, file_type, original_language, created_at 
                        FROM user_data.audio_jobs 
                        WHERE id = ANY(%(job_ids)s::uuid[])
                        """
                        job_results = db.execute_query(job_query, {"job_ids": job_ids})
                        
                        if job_results:
                            for job in job_results:
                                job_info[str(job["id"])] = {
                                    "file_name": job.get("file_name", "Unknown"),
                                    "file_size_bytes": job.get("file_size_bytes", 0),
                                    "original_language": job.get("original_language", "Unknown"),
                                    "created_at": job.get("created_at", "")
                                }
                except Exception as e:
                    logger.error(f"Error retrieving job metadata: {e}")
                
                # Format the context with document content and metadata
                context_parts = []
                
                # Add document overview
                if job_info:
                    context_parts.append("DOCUMENT INFORMATION:")
                    for job_id, info in job_info.items():
                        context_parts.append(f"File: {info['file_name']}")
                        if info.get('file_size_bytes'):
                            size_kb = info['file_size_bytes'] / 1024
                            context_parts.append(f"Size: {size_kb:.1f} KB")
                        if info.get('original_language'):
                            context_parts.append(f"Language: {info['original_language']}")
                    context_parts.append("")
                
                # Add document content sections
                context_parts.append("DOCUMENT CONTENT:")
                
                for i, doc in enumerate(documents):
                    # Add source identification
                    if "job_id" in doc.get("metadata", {}):
                        job_id = doc["metadata"]["job_id"]
                        file_name = job_info.get(job_id, {}).get("file_name", "Unknown Document")
                        chunk = doc["metadata"].get("chunk_index", 0)
                        context_parts.append(f"[Source {i+1}: {file_name} (Section {chunk})]")
                    else:
                        context_parts.append(f"[Source {i+1}]")
                        
                    # Add the document content
                    context_parts.append(doc["content"])
                    context_parts.append("")
                
                # Combine everything into one context string
                state["context"] = "\n".join(context_parts)
                
                # Add context to message metadata for the frontend
                state["document_sources"] = documents
                
                return state
                
            def generate_response(state: DocumentChatState) -> DocumentChatState:
                """Generate a response based on the retrieved context"""
                # Construct messages
                messages = state["messages"].copy()
                
                # Create a system message directing the model to use the context
                system_message = """You are a helpful AI assistant that answers questions based on document content.
                
                The user has provided a document, and I'll show you the most relevant parts based on their question.
                
                When answering:
                1. Only use information from the provided document context
                2. If the context doesn't contain the answer, say you don't know but don't make up information
                3. Be concise and accurate
                4. If the context is empty or says there's no relevant content, explain that you don't have enough information
                
                Here is the document context for this question:
                
                {context}
                """.format(context=state["context"])
                
                # Add the system message
                final_messages = [SystemMessage(content=system_message)]
                # Add only the user's question as the last message
                final_messages.append(messages[-1])
                
                # Use the chat model to generate a response
                ai_response = llm.invoke(final_messages)
                
                # Add the response to the state
                state["messages"].append(ai_response)
                
                return state
            
            # Build document chat workflow
            doc_workflow = StateGraph(DocumentChatState)
            
            # Add nodes for document chat
            doc_workflow.add_node("retrieve_context", retrieve_context)
            doc_workflow.add_node("generate_response", generate_response)
            
            # Set the entry point
            doc_workflow.set_entry_point("retrieve_context")
            
            # Add edges to connect the nodes
            doc_workflow.add_edge("retrieve_context", "generate_response")
            doc_workflow.add_edge("generate_response", END)
            
            # Compile the workflow
            doc_app = doc_workflow.compile()
            
            # Initial state for document chat
            doc_chat_state = {
                "messages": [
                    # Add system message if one was provided in the agent settings
                    SystemMessage(content=agent_system_message) if agent_system_message else None,
                    # Add the user's current message
                    HumanMessage(content=chat_request.message)
                ]
            }
            
            # Remove None values from messages
            doc_chat_state["messages"] = [msg for msg in doc_chat_state["messages"] if msg]
            
            # Execute document chat workflow
            final_state = doc_app.invoke(doc_chat_state)
            
            # Get response from final state
            response_text = final_state["messages"][-1].content
            
        else:
            # Define General Chat Workflow using LangGraph
            
            # Define node for regular chat
            def process_chat(state: GeneralChatState) -> GeneralChatState:
                """Process regular chat without document context"""
                # Make synchronous - LangGraph handles async internally
                # Format conversation history for the model
                conversation = []
                
                # Add system message if available
                if agent_system_message:
                    conversation.append({"role": "system", "content": agent_system_message})
                    
                # Add previous messages from history
                for message in state["messages"]:
                    if isinstance(message, HumanMessage):
                        conversation.append({"role": "user", "content": message.content})
                    elif isinstance(message, AIMessage):
                        conversation.append({"role": "assistant", "content": message.content})
                    elif isinstance(message, SystemMessage):
                        conversation.append({"role": "system", "content": message.content})
                
                # Get chat response
                response = llm.invoke(conversation)
                response_text = response.content if hasattr(response, 'content') else str(response)
                
                # Return the response to add to messages
                state["messages"].append(AIMessage(content=response_text))
                return state
            
            # Build general chat workflow
            chat_workflow = StateGraph(GeneralChatState)
            
            # Add node and edges
            chat_workflow.add_node("process_chat", process_chat)
            chat_workflow.set_entry_point("process_chat")
            chat_workflow.add_edge("process_chat", END)
            
            # Compile workflow
            chat_app = chat_workflow.compile()
            
            # Initialize general chat state
            general_chat_state = {
                "messages": [
                    # Add system message if one was provided in the agent settings
                    SystemMessage(content=agent_system_message) if agent_system_message else None,
                    # Add the user's current message
                    HumanMessage(content=chat_request.message)
                ]
            }
            
            # Remove None values from messages
            general_chat_state["messages"] = [msg for msg in general_chat_state["messages"] if msg]
            
            # Execute general chat workflow
            final_state = chat_app.invoke(general_chat_state)
            
            # Get response from final state
            response_text = final_state["messages"][-1].content
        
        # Store the response message
        if user_message_id:
            # Save AI response message linked to the user message
            save_query = """
            INSERT INTO user_data.chat_messages
                (session_id, role, content, embedding, metadata, message_timestamp)
            VALUES
                (%(session_id)s, 'assistant', %(content)s, %(embedding)s, %(metadata)s, NOW())
            RETURNING id
            """
            
            # Generate embedding for the response
            response_embedding = generate_embedding(response_text)
            
            # Create metadata with parent reference
            message_metadata = {
                "model": model_name,
                "parameters": chat_parameters,
                "parent_id": user_message_id  # Store parent reference in metadata
            }
            
            ai_message = db.execute_query(save_query, {
                "session_id": session_id,
                "content": response_text,
                "embedding": response_embedding,
                "metadata": json.dumps(message_metadata)
            })
            
            ai_message_id = ai_message[0]["id"] if ai_message else None
        
        # Extract document sources when using document chat mode
        source_documents = []
        if document_id and 'retrieve_document_content' in locals():
            # If document sources were used, include them in the response metadata
            try:
                # Get the most recent user query
                last_user_query = chat_request.message  # Use the current message directly
                        
                if last_user_query:
                    # Get document sources for this query to include in metadata
                    document_sources = retrieve_document_content(last_user_query)
                    source_documents = document_sources if document_sources else []
                    logger.info(f"Including {len(source_documents)} source documents in response metadata")
            except Exception as e:
                logger.error(f"Error retrieving document sources for metadata: {e}")
        
        # Construct the response
        messages = []
        if user_message_id:
            messages.append({
                "id": user_message_id,
                "role": "user",
                "content": chat_request.message,
                "timestamp": datetime.now().isoformat(),
                "metadata": user_metadata  # Include metadata in response
            })
        
        if ai_message_id:
            messages.append({
                "id": ai_message_id,
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.now().isoformat(),
                "metadata": message_metadata  # Include metadata in response
            })
        
        # Convert to ChatMessage objects for the response model
        chat_messages = []
        for msg in messages:
            message_metadata = msg.get("metadata", {})
            # Add source documents only to the last assistant message
            if msg["role"] == "assistant" and msg["id"] == ai_message_id and source_documents:
                message_metadata["source_documents"] = source_documents
                
            chat_messages.append(ChatMessage(
                role=msg["role"], 
                content=msg["content"], 
                metadata=message_metadata
            ))
        
        # Return the response
        return ChatResponse(
            session_id=session_id,
            response=response_text,
            messages=chat_messages,
            metadata={}
        )
            
    except Exception as e:
        logger.error(f"Error in chat_with_agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat request: {str(e)}"
        )

# Background task to generate embeddings for messages
async def generate_message_embeddings(session_id: str, message_ids: List[str]):
    """Generate embeddings for chat messages in the background"""
    try:
        db = get_pg_db()
        
        for msg_id in message_ids:
            # Get message content
            query = """
            SELECT content FROM user_data.chat_messages
            WHERE id = %(msg_id)s AND embedding IS NULL
            """
            
            result = db.execute_query(query, {"msg_id": msg_id})
            
            if not result:
                continue  # Skip if message doesn't exist or already has embedding
                
            content = result[0]["content"]
            
            # Generate embedding
            embedding = generate_embedding(content)
            
            # Update message with embedding
            update_query = """
            UPDATE user_data.chat_messages
            SET embedding = %(embedding)s
            WHERE id = %(msg_id)s
            """
            
            db.execute_query(update_query, {
                "msg_id": msg_id,
                "embedding": embedding
            }, fetch=False)
            
            # Add delay to avoid overwhelming the embedding service
            time.sleep(0.5)
            
    except Exception as e:
        logging.error(f"Error generating message embeddings: {str(e)}")
