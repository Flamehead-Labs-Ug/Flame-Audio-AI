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

        # Prepare the request
        payload = {
            "content": message
        }

        # Add document_id if provided
        if document_id:
            payload["document_id"] = document_id

        # Add model_name if provided
        if model_name:
            payload["model_name"] = model_name

        # Try to send the message with retries
        for attempt in range(max_retries):
            try:
                # Since the /langgraph/sessions/{session_id}/messages endpoint doesn't exist,
                # we'll use the tool call mechanism to send messages
                # Prepare the chat request payload
                chat_request = {
                    "message": message,
                    "parameters": {},
                    "metadata": {}
                }

                # Don't include session_id in the chat_request to let the backend create a new session
                # This avoids the 404 error when the session doesn't exist

                # Add optional parameters if provided
                if document_id:
                    chat_request["document_id"] = document_id
                if model_name:
                    chat_request["parameters"]["model"] = model_name

                tool_payload = {
                    "tool": "chat_with_agent",
                    "args": {
                        "token": self.auth_token,
                        "chat_request": chat_request
                    }
                }

                response = requests.post(
                    f"{self.mcp_url}/langgraph/tools/call",
                    json=tool_payload,
                    headers=self.get_auth_headers(),
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    return response.json()
                else:
                    logger.error(f"Error sending message (attempt {attempt+1}/{max_retries}): {response.status_code} - {response.text}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
            except Exception as e:
                logger.error(f"Error sending message (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)

        return None


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
    return client.send_message(message, session_id, document_id, model_name, max_retries, retry_delay)
