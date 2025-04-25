"""
LangGraph Client Utility Module

This module provides a client for interacting with the LangGraph endpoints of the Flame MCP service (flame-mcp).
"""

import requests
import time
from typing import Dict, List, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("langgraph_client")

class LangGraphClient:
    """
    Client for interacting with the LangGraph endpoints of the Flame MCP service (flame-mcp).
    """

    def __init__(self, mcp_url: str = "http://localhost:8001", auth_token: Optional[str] = None):
        """
        Initialize the LangGraph client.

        Args:
            mcp_url: URL of the Flame MCP service (flame-mcp)
            auth_token: Authentication token for the Flame MCP service
        """
        self.mcp_url = mcp_url.rstrip('/')
        self.auth_token = auth_token
        self.session_id = None
        self.timeout = 30  # Default timeout in seconds

    def set_auth_token(self, auth_token: str) -> None:
        """
        Set the authentication token.

        Args:
            auth_token: Authentication token for the Flame MCP service
        """
        self.auth_token = auth_token

    def set_session_id(self, session_id: str) -> None:
        """
        Set the session ID.

        Args:
            session_id: Session ID to use
        """
        self.session_id = session_id

    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers.

        Returns:
            Dictionary of authentication headers
        """
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    def check_health(self) -> Dict[str, Any]:
        """
        Check the health of the Flame MCP service (flame-mcp).

        Returns:
            Dictionary with status and details
        """
        try:
            # First try the LangGraph endpoint
            try:
                response = requests.get(
                    f"{self.mcp_url}/langgraph/health",
                    headers=self.get_auth_headers(),
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    return {"status": "online", "details": response.json()}
            except Exception as langgraph_error:
                logger.warning(f"LangGraph health check failed: {str(langgraph_error)}")

            # Fall back to the regular health endpoint
            response = requests.get(
                f"{self.mcp_url}/health",
                headers=self.get_auth_headers(),
                timeout=self.timeout
            )

            if response.status_code == 200:
                return {"status": "online", "details": response.json()}
            else:
                return {"status": "error", "details": {"message": f"HTTP {response.status_code}: {response.text}"}}
        except Exception as e:
            return {"status": "offline", "details": {"message": str(e)}}

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all available tools.

        Returns:
            List of tools
        """
        try:
            response = requests.get(
                f"{self.mcp_url}/langgraph/tools",
                headers=self.get_auth_headers(),
                timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json().get("tools", [])
            else:
                logger.error(f"Error listing tools: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error listing tools: {str(e)}")
            return []

    def create_session(
        self,
        user_id: str,
        tools: Optional[List[str]] = None,
        model_name: str = "default",
        system_message: Optional[str] = None,
        document_id: Optional[str] = None
    ) -> bool:
        """
        Create a new LangGraph session.

        Args:
            user_id: The user ID to associate with the session (required)
            tools: List of tool names to enable
            model_name: Model to use
            system_message: System message to use
            document_id: Optional document ID to use for context

        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare the request
            payload = {
                "user_id": user_id,
                "expiry_days": 7  # Default expiry days
            }

            # Optional parameters - these will be used by the backend
            # but aren't directly passed to the session creation endpoint
            if tools is not None:
                payload["tools"] = tools

            if model_name and model_name != "default":
                payload["model_name"] = model_name

            if system_message:
                payload["system_message"] = system_message

            if document_id:
                payload["document_id"] = document_id

            # Send the request
            response = requests.post(
                f"{self.mcp_url}/langgraph/sessions",
                json=payload,
                headers=self.get_auth_headers(),
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                self.session_id = data.get("session_id")
                logger.info(f"Created session: {self.session_id}")
                return True
            else:
                logger.error(f"Error creating session: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all LangGraph sessions.

        Returns:
            List of sessions
        """
        try:
            response = requests.get(
                f"{self.mcp_url}/langgraph/sessions",
                headers=self.get_auth_headers(),
                timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error listing sessions: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error listing sessions: {str(e)}")
            return []

    def get_session(self, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get a LangGraph session.

        Args:
            session_id: Session ID to get (uses self.session_id if not provided)

        Returns:
            Session details or None if not found
        """
        if not session_id and not self.session_id:
            logger.error("No session ID provided")
            return None

        session_id = session_id or self.session_id

        try:
            response = requests.get(
                f"{self.mcp_url}/langgraph/sessions/{session_id}",
                headers=self.get_auth_headers(),
                timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error getting session: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error getting session: {str(e)}")
            return None

    def delete_session(self, session_id: Optional[str] = None) -> bool:
        """
        Delete a LangGraph session.

        Args:
            session_id: Session ID to delete (uses self.session_id if not provided)

        Returns:
            True if successful, False otherwise
        """
        if not session_id and not self.session_id:
            logger.error("No session ID provided")
            return False

        session_id = session_id or self.session_id

        try:
            response = requests.delete(
                f"{self.mcp_url}/langgraph/sessions/{session_id}",
                headers=self.get_auth_headers(),
                timeout=self.timeout
            )

            if response.status_code == 200:
                if session_id == self.session_id:
                    self.session_id = None
                return True
            else:
                logger.error(f"Error deleting session: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error deleting session: {str(e)}")
            return False

    def call_tool(
        self,
        tool_name: str,
        tool_args: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Optional[Dict[str, Any]]:
        """
        Call a tool without requiring a session.

        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments to pass to the tool
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds

        Returns:
            Tool result or None if failed
        """
        # Prepare the request - match the server's expected format
        payload = {
            "tool": tool_name,
            "args": tool_args or {}
        }

        # Try to call the tool with retries
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.mcp_url}/langgraph/tools/call",
                    json=payload,
                    headers=self.get_auth_headers(),
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    # Extract the result field from the response
                    response_data = response.json()
                    return response_data
                else:
                    logger.error(f"Error calling tool (attempt {attempt+1}/{max_retries}): {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
            except Exception as e:
                logger.error(f"Error calling tool (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        return None

    def call_tool_sync(
        self,
        tool_name: str,
        tool_args: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Optional[Dict[str, Any]]:
        """
        Call a tool synchronously.

        Args:
            tool_name: Name of the tool to call
            tool_args: Arguments to pass to the tool
            session_id: Session ID to use (uses self.session_id if not provided)
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds

        Returns:
            Tool result or None if failed
        """
        if not session_id and not self.session_id:
            logger.error("No session ID provided")
            return None

        session_id = session_id or self.session_id

        # Prepare the request - match the server's expected format
        payload = {
            "tool": tool_name,  # Changed from tool_name to tool to match server implementation
            "args": tool_args or {}  # Changed from tool_args to args to match server implementation
        }

        # Try to call the tool with retries
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.mcp_url}/langgraph/tools/call",
                    json=payload,
                    headers=self.get_auth_headers(),
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    # Extract the result field from the response
                    response_data = response.json()
                    return response_data.get("result")
                else:
                    logger.error(f"Error calling tool (attempt {attempt+1}/{max_retries}): {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
            except Exception as e:
                logger.error(f"Error calling tool (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        return None

    def get_tools(self) -> List[Dict[str, Any]]:
        """
        Get all available tools.

        Returns:
            List of tools
        """
        try:
            response = requests.get(
                f"{self.mcp_url}/langgraph/tools",
                headers=self.get_auth_headers(),
                timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json().get("tools", [])
            else:
                logger.error(f"Error getting tools: {response.status_code} - {response.text}")
                return []
        except Exception as e:
            logger.error(f"Error getting tools: {str(e)}")
            return []

    def send_message(
        self,
        message: str,
        session_id: Optional[str] = None,
        document_id: Optional[str] = None,
        model_name: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> Optional[Dict[str, Any]]:
        """
        Send a message to a LangGraph session.

        Args:
            message: Message to send
            session_id: Session ID to send to (uses self.session_id if not provided)
            document_id: Optional document ID to use for context
            model_name: Optional model name to use for this message
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds

        Returns:
            Response from the LangGraph session or None if failed
        """
        if not session_id and not self.session_id:
            logger.error("No session ID provided")
            return None

        session_id = session_id or self.session_id

        tool_args = {
            "token": self.auth_token,
            "chat_request": {
                "message": message,
                "document_id": document_id,
                "parameters": {"model": model_name} if model_name else {},
                "session_id": session_id,
            }
        }
        # Remove None values from chat_request
        tool_args["chat_request"] = {k: v for k, v in tool_args["chat_request"].items() if v is not None}
        return self.call_tool_sync("chat_with_agent", tool_args, session_id, max_retries, retry_delay)


# Convenience functions for common operations

def get_langgraph_client(mcp_url: str, auth_token: Optional[str] = None) -> LangGraphClient:
    """
    Get a LangGraph client instance.

    Args:
        mcp_url: URL of the Flame MCP service (flame-mcp)
        auth_token: Authentication token

    Returns:
        LangGraphClient instance
    """
    return LangGraphClient(mcp_url, auth_token)

def list_tools(client: LangGraphClient) -> List[Dict[str, Any]]:
    """
    List all available tools.

    Args:
        client: LangGraphClient instance

    Returns:
        List of tools
    """
    return client.list_tools()

def create_session(
    client: LangGraphClient,
    user_id: str,
    tools: Optional[List[str]] = None,
    model_name: str = "default",
    system_message: Optional[str] = None,
    document_id: Optional[str] = None
) -> bool:
    """
    Create a new LangGraph session.

    Args:
        client: LangGraphClient instance
        user_id: The user ID to associate with the session (required)
        tools: List of tool names to enable
        model_name: Model to use
        system_message: System message to use
        document_id: Optional document ID to use for context

    Returns:
        True if successful, False otherwise
    """
    return client.create_session(user_id, tools, model_name, system_message, document_id)

def list_sessions(client: LangGraphClient) -> List[Dict[str, Any]]:
    """
    List all LangGraph sessions.

    Args:
        client: LangGraphClient instance

    Returns:
        List of sessions
    """
    return client.list_sessions()

def get_session(client: LangGraphClient, session_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Get a LangGraph session.

    Args:
        client: LangGraphClient instance
        session_id: Session ID to get (uses client.session_id if not provided)

    Returns:
        Session details or None if not found
    """
    return client.get_session(session_id)

def delete_session(client: LangGraphClient, session_id: Optional[str] = None) -> bool:
    """
    Delete a LangGraph session.

    Args:
        client: LangGraphClient instance
        session_id: Session ID to delete (uses client.session_id if not provided)

    Returns:
        True if successful, False otherwise
    """
    return client.delete_session(session_id)

def call_tool(
    mcp_url: str,
    tool_name: str,
    tool_args: Optional[Dict[str, Any]] = None,
    auth_token: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> Optional[Dict[str, Any]]:
    """
    Call a tool without requiring a client instance.

    Args:
        mcp_url: URL of the Flame MCP service (flame-mcp)
        tool_name: Name of the tool to call
        tool_args: Arguments to pass to the tool
        auth_token: Authentication token
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds

    Returns:
        Tool result or None if failed
    """
    client = get_langgraph_client(mcp_url, auth_token)
    return client.call_tool(tool_name, tool_args, max_retries, retry_delay)

def chat_with_agent_rag(
    client: LangGraphClient,
    token: str,
    question: str,
    retrieved_chunks: list = None,
    system_prompt: str = None,
    agent_id: str = None,
    document_id: str = None,
    parameters: dict = None,
    metadata: dict = None,
    session_id: str = None
) -> dict:
    """
    Call the 'chat_with_agent' tool with structured RAG chat arguments.

    Args:
        client: LangGraphClient instance
        token: Authentication token
        question: The user's question (message)
        retrieved_chunks: List of retrieved chunk dicts (optional)
        system_prompt: Custom system prompt (optional)
        agent_id: Agent ID (optional)
        document_id: Document ID (optional)
        parameters: Additional model/chat parameters (optional)
        metadata: Extra metadata (optional)
        session_id: Chat session ID (optional)

    Returns:
        The tool call result as a dict
    """
    # Debug output
    print(f"DEBUG: chat_with_agent_rag called with session_id: {session_id}")

    tool_args = {
        "token": token,
        "chat_request": {
            "message": question,
            "retrieved_chunks": retrieved_chunks,
            "system_prompt": system_prompt,
            "agent_id": agent_id,
            "document_id": document_id,
            "parameters": parameters,
            "metadata": metadata,
            "session_id": session_id,
        }
    }
    # Remove None values from chat_request
    tool_args["chat_request"] = {k: v for k, v in tool_args["chat_request"].items() if v is not None}

    # Make sure to set the session ID in the client to avoid the "No session ID provided" error
    if session_id:
        client.set_session_id(session_id)
        print(f"DEBUG: Set session_id in client to: {session_id}")
    else:
        print("DEBUG: No session_id provided for chat_with_agent_rag")

    # Call the tool
    print(f"DEBUG: Calling chat_with_agent with args: {tool_args}")
    result = client.call_tool_sync("chat_with_agent", tool_args)
    print(f"DEBUG: chat_with_agent result: {result}")

    return result

def send_message(
    client: LangGraphClient,
    message: str,
    session_id: Optional[str] = None,
    document_id: Optional[str] = None,
    model_name: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> Optional[Dict[str, Any]]:
    """
    Send a message to a LangGraph session.

    Args:
        client: LangGraphClient instance
        message: Message to send
        session_id: Session ID to send to (uses client.session_id if not provided)
        document_id: Optional document ID to use for context
        model_name: Optional model name to use for this message
        max_retries: Maximum number of retries
        retry_delay: Delay between retries in seconds

    Returns:
        Response from the LangGraph session or None if failed
    """
    tool_args = {
        "token": client.auth_token,
        "chat_request": {
            "question": message,
            "document_id": document_id,
            "parameters": {"model": model_name} if model_name else {},
            "session_id": session_id,
        }
    }
    # Remove None values from chat_request
    tool_args["chat_request"] = {k: v for k, v in tool_args["chat_request"].items() if v is not None}

    # Make sure to set the session ID in the client to avoid the "No session ID provided" error
    if session_id:
        client.set_session_id(session_id)

    return client.call_tool_sync("chat_with_agent", tool_args, session_id, max_retries, retry_delay)
