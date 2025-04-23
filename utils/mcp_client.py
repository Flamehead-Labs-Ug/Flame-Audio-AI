"""
Flame MCP Client Utility Module

This module provides a consistent interface for interacting with the Flame MCP service (flame-mcp).
It handles authentication, request formatting, and response parsing for the Flame MCP integration.
"""

import requests
import json
import time
import uuid
from typing import Dict, List, Any, Optional
import logging
from utils.mcp_sse_client import McpSseClient

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_client")

class McpClient:
    """
    Client for interacting with the Flame MCP service (flame-mcp).

    This class provides methods for calling Flame MCP tools and managing sessions.
    """

    def __init__(self, mcp_url: str = "http://localhost:8001", auth_token: Optional[str] = None):
        """
        Initialize the Flame MCP client.

        Args:
            mcp_url: URL of the Flame MCP service (flame-mcp)
            auth_token: Authentication token for the Flame MCP service
        """
        self.mcp_url = mcp_url
        self.auth_token = auth_token
        self.session_id = str(uuid.uuid4())
        self.timeout = 10  # Default timeout in seconds
        self.sse_client = None

    def set_auth_token(self, auth_token: str) -> None:
        """
        Set the authentication token.

        Args:
            auth_token: Authentication token for the Flame MCP service (flame-mcp)
        """
        self.auth_token = auth_token

    def set_session_id(self, session_id: str) -> None:
        """
        Set the session ID.

        Args:
            session_id: Session ID for the Flame MCP service (flame-mcp)
        """
        self.session_id = session_id

    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for requests to the Flame MCP service (flame-mcp).

        Returns:
            Dictionary of authentication headers
        """
        if self.auth_token:
            logger.info("Using authentication token for request")
            return {"Authorization": f"Bearer {self.auth_token}"}
        logger.warning("No authentication token available for request")
        return {}

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any] = None,
        wait_for_response: bool = True,
        max_retries: int = 30,
        retry_delay: float = 1.0
    ) -> Any:
        """
        Call a Flame MCP tool and get the result.

        Args:
            tool_name: Name of the Flame MCP tool to call
            arguments: Arguments to pass to the tool
            wait_for_response: Whether to wait for the response
            max_retries: Maximum number of retries when polling for response
            retry_delay: Delay between retries in seconds

        Returns:
            Tool result or None if wait_for_response is False
        """
        if arguments is None:
            arguments = {}

        # Add auth_token to arguments if available
        if self.auth_token and "auth_token" not in arguments:
            arguments["auth_token"] = self.auth_token

        # Prepare the request
        headers = self.get_auth_headers()
        params = {"session_id": self.session_id}

        # Construct a message that will trigger the tool
        payload = {
            "messages": [{"role": "user", "content": f"Call {tool_name}"}],
            "tools": [{"name": tool_name}]
        }

        # If arguments are provided, add them to the payload
        if arguments:
            payload["arguments"] = arguments

        try:
            # Send the request
            logger.info(f"Calling tool: {tool_name} with arguments: {arguments}")
            response = requests.post(
                f"{self.mcp_url}/messages/",  # Use trailing slash to avoid redirects
                json=payload,
                params=params,
                headers=headers,
                timeout=self.timeout
            )

            # Check if the request was accepted
            if response.status_code == 202:
                logger.info(f"Tool call accepted: {tool_name}")

                # If we don't need to wait for the response, return None
                if not wait_for_response:
                    return None

                # Poll for the response
                retry_count = 0
                while retry_count < max_retries:
                    # Wait before polling
                    time.sleep(retry_delay)

                    # Get the messages
                    messages_response = requests.get(
                        f"{self.mcp_url}/messages/",  # Use trailing slash to avoid redirects
                        params=params,
                        headers=headers,
                        timeout=self.timeout
                    )

                    if messages_response.status_code == 200:
                        messages = messages_response.json().get("messages", [])

                        # Look for the tool response in the messages
                        for message in reversed(messages):  # Start from the most recent
                            if message.get("role") == "assistant" and "tool_calls" in message:
                                tool_calls = message.get("tool_calls", [])
                                for tool_call in tool_calls:
                                    if tool_call.get("name") == tool_name:
                                        # Parse the arguments as JSON
                                        try:
                                            result = json.loads(tool_call.get("arguments", "{}"))
                                            logger.info(f"Tool call successful: {tool_name}")
                                            return result
                                        except json.JSONDecodeError:
                                            logger.error(f"Failed to parse tool result: {tool_call.get('arguments')}")
                                            return None

                    # Increment retry count
                    retry_count += 1

                # If we've exhausted retries, return None
                logger.warning(f"Timeout waiting for tool response: {tool_name}")
                return None
            else:
                # Request was not accepted
                logger.error(f"Tool call failed: {tool_name}, status: {response.status_code}, response: {response.text}")
                return None

        except Exception as e:
            logger.exception(f"Error calling tool {tool_name}: {str(e)}")
            return None

    def call_tool_sync(
        self,
        tool_name: str,
        arguments: Dict[str, Any] = None,
        wait_for_response: bool = True,
        max_retries: int = 30,
        retry_delay: float = 1.0
    ) -> Any:
        """
        Synchronous version of call_tool.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            wait_for_response: Whether to wait for the response
            max_retries: Maximum number of retries when polling for response
            retry_delay: Delay between retries in seconds

        Returns:
            Tool result or None if wait_for_response is False
        """
        if arguments is None:
            arguments = {}

        # Add auth_token to arguments if available
        if self.auth_token and "auth_token" not in arguments:
            arguments["auth_token"] = self.auth_token
            logger.info(f"Added auth token to {tool_name} call")

        # Ensure we have a valid session before proceeding
        if not self.sse_client or not self.sse_client.connected:
            logger.info("No active SSE connection, creating a new session")
            if not self.create_session():
                logger.error(f"Failed to create session for tool call: {tool_name}")
                return None

        # Log the request for debugging
        logger.info(f"Calling tool {tool_name} with arguments: {arguments}")

        try:
            # Prepare the request
            payload = {
                "messages": [{"role": "user", "content": f"Call {tool_name}"}],
                "tools": [{"name": tool_name}],
                "arguments": arguments
            }

            # Send the message
            send_result = self.sse_client.send_message(payload)

            if not send_result.get("success", False):
                error_message = send_result.get("error", "Unknown error")
                logger.error(f"Failed to send tool call: {error_message}")

                # If the error is related to the session, try to create a new one and retry
                if "session" in error_message.lower() or "not connected" in error_message.lower():
                    logger.warning("Session error, creating a new session and retrying")

                    # Close the current session if it exists
                    if self.sse_client:
                        self.sse_client.disconnect()

                    # Create a new session
                    if self.create_session():
                        logger.info(f"Created new session: {self.session_id}")
                        # Retry the tool call with the new session
                        return self.call_tool_sync(tool_name, arguments, wait_for_response, max_retries, retry_delay)

                return None

            # If we don't need to wait for the response, return None
            if not wait_for_response:
                return None

            # Poll for the response
            retry_count = 0
            while retry_count < max_retries:
                # Wait before polling
                time.sleep(retry_delay)

                # Get the messages
                messages_result = self.sse_client.get_messages()

                if not messages_result.get("success", False):
                    logger.warning(f"Failed to get messages: {messages_result.get('error', 'Unknown error')}")
                    retry_count += 1
                    continue

                messages = messages_result.get("messages", [])

                # Look for the tool response in the messages
                for message in reversed(messages):  # Start from the most recent
                    if message.get("role") == "assistant" and "tool_calls" in message:
                        tool_calls = message.get("tool_calls", [])
                        for tool_call in tool_calls:
                            if tool_call.get("name") == tool_name:
                                # Parse the arguments as JSON
                                try:
                                    result = json.loads(tool_call.get("arguments", "{}"))
                                    logger.info(f"Tool call successful: {tool_name}")
                                    return result
                                except json.JSONDecodeError:
                                    logger.error(f"Failed to parse tool result: {tool_call.get('arguments')}")
                                    return None

                # Increment retry count
                retry_count += 1

            # If we've exhausted retries, return None
            logger.warning(f"Timeout waiting for tool response: {tool_name}")
            return None
        except Exception as e:
            logger.exception(f"Error calling tool {tool_name}: {str(e)}")
            return None

    def check_health(self) -> Dict[str, Any]:
        """
        Check the health of the MCP service.

        Returns:
            Dictionary with health status
        """
        try:
            # Try without trailing slash first
            response = requests.get(
                f"{self.mcp_url}/health",  # No trailing slash
                headers=self.get_auth_headers(),
                timeout=5  # Shorter timeout for health checks
            )

            if response.status_code == 200:
                return {
                    "status": "online",
                    "details": response.json()
                }
            else:
                # Try with trailing slash as fallback
                response = requests.get(
                    f"{self.mcp_url}/health/",  # With trailing slash
                    headers=self.get_auth_headers(),
                    timeout=5
                )

                if response.status_code == 200:
                    return {
                        "status": "online",
                        "details": response.json()
                    }
                else:
                    return {
                        "status": "error",
                        "details": {
                            "message": f"HTTP {response.status_code}: {response.text}"
                        }
                    }
        except requests.exceptions.ConnectTimeout:
            return {
                "status": "offline",
                "details": {
                    "message": "Connection timed out. MCP service is not responding."
                }
            }
        except requests.exceptions.ConnectionError:
            return {
                "status": "offline",
                "details": {
                    "message": "Connection error. MCP service is not running or not accessible."
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "details": {
                    "message": f"Error: {str(e)}"
                }
            }

    def create_session(self) -> bool:
        """
        Create a new session with the MCP service.

        Returns:
            True if session was created successfully, False otherwise
        """
        try:
            # First, handle any redirects to get the correct base URL
            try:
                # Make a HEAD request to check for redirects
                head_response = requests.head(
                    f"{self.mcp_url}/sse",  # Try without trailing slash
                    allow_redirects=False,
                    headers=self.get_auth_headers(),
                    timeout=3
                )

                # If we get a redirect, update the URL
                if head_response.status_code in [301, 302, 307, 308]:
                    redirect_url = head_response.headers.get('Location')
                    if redirect_url:
                        # Extract the base URL from the redirect
                        if redirect_url.endswith('/sse/') or redirect_url.endswith('/sse'):
                            self.mcp_url = redirect_url.split('/sse')[0]
                            logger.info(f"Updated MCP URL to: {self.mcp_url}")
            except Exception as head_error:
                logger.warning(f"Error checking for redirects: {str(head_error)}")

            # Generate a new session ID
            self.session_id = str(uuid.uuid4())

            # First try to create a session using the messages endpoint with session_id as a parameter
            logger.info(f"Creating session with ID: {self.session_id} using messages endpoint")
            try:
                # Try to get messages for this session to register it
                messages_response = requests.get(
                    f"{self.mcp_url}/messages",
                    params={"session_id": self.session_id},
                    headers=self.get_auth_headers(),
                    timeout=5
                )

                if messages_response.status_code == 200:
                    logger.info(f"Successfully registered session via messages endpoint: {self.session_id}")
                    session_created = True
                else:
                    logger.warning(f"Failed to register session via messages endpoint: {messages_response.status_code}")
                    session_created = False
            except Exception as msg_error:
                logger.warning(f"Error registering session via messages endpoint: {str(msg_error)}")
                session_created = False

            # If that didn't work, try the sessions endpoint
            if not session_created:
                try:
                    # Try without trailing slash
                    response = requests.post(
                        f"{self.mcp_url}/sessions/{self.session_id}",
                        headers=self.get_auth_headers(),
                        timeout=5
                    )

                    if response.status_code in [200, 201, 202]:
                        logger.info(f"Successfully created session via sessions endpoint: {self.session_id}")
                        session_created = True
                    else:
                        # Try with trailing slash
                        response = requests.post(
                            f"{self.mcp_url}/sessions/{self.session_id}/",
                            headers=self.get_auth_headers(),
                            timeout=5
                        )

                        if response.status_code in [200, 201, 202]:
                            logger.info(f"Successfully created session via sessions endpoint with trailing slash: {self.session_id}")
                            session_created = True
                        else:
                            logger.warning(f"Failed to create session via sessions endpoint: {response.status_code} - {response.text}")
                            session_created = False
                except Exception as session_error:
                    logger.warning(f"Error creating session via sessions endpoint: {str(session_error)}")
                    session_created = False

            # If we couldn't create a session through the API, we'll still try to connect via SSE
            # as the session might be created automatically when connecting

            # Create a new SSE client
            logger.info(f"Creating new SSE client for MCP URL: {self.mcp_url}")
            self.sse_client = McpSseClient(self.mcp_url, self.auth_token)
            self.sse_client.session_id = self.session_id

            # Connect to the Flame MCP server
            if self.sse_client.connect():
                logger.info(f"Successfully connected to Flame MCP server with session ID: {self.session_id}")
                return True
            else:
                logger.warning("Failed to connect to Flame MCP server")
                return False
        except Exception as e:
            logger.exception(f"Error creating session: {str(e)}")
            return False

    def get_tools(self) -> List[Dict[str, Any]]:
        """
        Get available tools from the MCP service.

        Returns:
            List of available tools
        """
        try:
            # Try without trailing slash first
            response = requests.get(
                f"{self.mcp_url}/tools",  # No trailing slash
                headers=self.get_auth_headers(),
                timeout=self.timeout
            )

            if response.status_code == 200:
                return response.json().get("tools", [])
            else:
                # Try with trailing slash as fallback
                response = requests.get(
                    f"{self.mcp_url}/tools/",  # With trailing slash
                    headers=self.get_auth_headers(),
                    timeout=self.timeout
                )

                if response.status_code == 200:
                    return response.json().get("tools", [])
                else:
                    logger.error(f"Failed to get tools: {response.status_code} - {response.text}")
                    return []
        except Exception as e:
            logger.exception(f"Error getting tools: {str(e)}")
            return []

    def send_chat_message(
        self,
        message: str,
        tools: List[str] = None,
        agent_id: str = None,
        document_id: str = None,
        chat_session_id: str = None,
        model: str = None
    ) -> Dict[str, Any]:
        """
        Send a chat message to the MCP service.

        Args:
            message: Message to send
            tools: List of tool names to enable
            agent_id: ID of the agent to use
            document_id: ID of the document to use
            chat_session_id: ID of the chat session
            model: Model to use for the response

        Returns:
            Dictionary with success status and messages
        """
        if tools is None:
            tools = []

        # Ensure we have a valid session before proceeding
        if not self.sse_client or not self.sse_client.connected:
            logger.info("No active SSE connection, creating a new session")
            if not self.create_session():
                logger.error("Failed to create session for chat message")
                return {
                    "success": False,
                    "error": "Failed to create MCP session"
                }

        # Construct the payload
        payload = {
            "messages": [{"role": "user", "content": message}]
        }

        # Add tools if specified
        if tools:
            payload["tools"] = [{"name": tool} for tool in tools]

        # Add agent_id if specified
        if agent_id:
            payload["agent_id"] = agent_id

        # Add document_id if specified
        if document_id:
            payload["document_id"] = document_id

        # Add chat_session_id if specified
        if chat_session_id:
            payload["chat_session_id"] = chat_session_id

        # Add model if specified
        if model:
            payload["model"] = model

        try:
            # Log the request details
            logger.info(f"Sending chat message using SSE client with session_id={self.session_id}")

            # Send the message using the SSE client
            send_result = self.sse_client.send_message(payload)

            if not send_result.get("success", False):
                error_message = send_result.get("error", "Unknown error")
                logger.error(f"Failed to send message: {error_message}")

                # If the error is related to the session, try to create a new one and retry
                if "session" in error_message.lower() or "not connected" in error_message.lower():
                    logger.warning("Session error, creating a new session and retrying")

                    # Close the current session if it exists
                    if self.sse_client:
                        self.sse_client.disconnect()

                    # Create a new session
                    if self.create_session():
                        logger.info(f"Created new session: {self.session_id}")
                        # Retry the chat message with the new session
                        return self.send_chat_message(message, tools, agent_id, document_id, chat_session_id, model)
                    else:
                        logger.error("Failed to create a new session for retry")
                        return {
                            "success": False,
                            "error": "Failed to create MCP session for retry"
                        }

                return {
                    "success": False,
                    "error": error_message
                }

            # Message was sent successfully, now get the response
            logger.info("Chat message sent successfully, getting response")

            # Poll for the response
            max_retries = 30
            retry_count = 0

            while retry_count < max_retries:
                # Wait before polling
                time.sleep(1)

                # Get the messages
                logger.info(f"Polling for messages with session_id={self.session_id}")
                messages_result = self.sse_client.get_messages()

                if not messages_result.get("success", False):
                    logger.warning(f"Failed to get messages: {messages_result.get('error', 'Unknown error')}")
                    retry_count += 1
                    continue

                messages = messages_result.get("messages", [])
                logger.info(f"Received {len(messages)} messages")

                # Check if there's a new assistant message
                if len(messages) > 1:  # At least user message + assistant response
                    logger.info("Found assistant response, returning messages")
                    return {
                        "success": True,
                        "messages": messages
                    }

                # Increment retry count
                retry_count += 1

            # If we've exhausted retries, return timeout error
            logger.warning("Timeout waiting for chat response")
            return {
                "success": False,
                "error": "Timeout waiting for response"
            }

        except Exception as e:
            logger.exception(f"Error sending chat message: {str(e)}")
            return {
                "success": False,
                "error": f"Error: {str(e)}"
            }


# Convenience functions for common operations

def get_mcp_client(mcp_url: str, auth_token: Optional[str] = None) -> McpClient:
    """
    Get an Flame MCP client instance.

    Args:
        mcp_url: URL of the MCP service
        auth_token: Authentication token

    Returns:
        McpClient instance
    """
    return McpClient(mcp_url, auth_token)

def list_agents(client: McpClient) -> List[Dict[str, Any]]:
    """
    List all agents for the current user.

    Args:
        client: McpClient instance

    Returns:
        List of agents
    """
    # Ensure we have an auth token
    if not client.auth_token:
        logger.error("No authentication token provided for list_agents")
        return []

    # Call the tool with explicit token parameter (renamed from list_agents to get_user_agents)
    result = client.call_tool_sync("get_user_agents", {"token": client.auth_token})

    # Log the result for debugging
    if result:
        logger.info(f"Successfully retrieved {len(result)} agents")
    else:
        logger.warning("No agents returned from get_user_agents call")

    return result or []

def get_chat_models(client: McpClient) -> List[Dict[str, Any]]:
    """
    Get available chat models.

    Args:
        client: McpClient instance

    Returns:
        List of chat models
    """
    # Ensure we have an auth token
    if not client.auth_token:
        logger.error("No authentication token provided for get_chat_models")
        return []

    # Call the tool with explicit token parameter
    result = client.call_tool_sync("get_chat_models", {"token": client.auth_token})

    return result or []

def list_documents(client: McpClient, agent_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List documents, optionally filtered by agent.

    Args:
        client: McpClient instance
        agent_id: Optional agent ID to filter documents

    Returns:
        List of documents
    """
    # Ensure we have an auth token
    if not client.auth_token:
        logger.error("No authentication token provided for list_documents")
        return []

    # Prepare arguments
    arguments = {"token": client.auth_token}
    if agent_id:
        arguments["agent_id"] = agent_id

    # Call the tool
    result = client.call_tool_sync("get_user_documents", arguments)

    return result or []

def list_chat_sessions(client: McpClient, agent_id: str) -> List[Dict[str, Any]]:
    """
    List chat sessions for an agent.

    Args:
        client: McpClient instance
        agent_id: Agent ID

    Returns:
        List of chat sessions
    """
    # Ensure we have an auth token
    if not client.auth_token:
        logger.error("No authentication token provided for list_chat_sessions")
        return []

    # Prepare arguments
    arguments = {
        "token": client.auth_token,
        "agent_id": agent_id
    }

    # Call the tool
    result = client.call_tool_sync("get_user_chat_sessions", arguments)

    return result or []

def get_chat_session(client: McpClient, session_id: str) -> Dict[str, Any]:
    """
    Get a specific chat session.

    Args:
        client: McpClient instance
        session_id: Session ID

    Returns:
        Chat session details
    """
    # Ensure we have an auth token
    if not client.auth_token:
        logger.error("No authentication token provided for get_chat_session")
        return {}

    # Prepare arguments
    arguments = {
        "token": client.auth_token,
        "session_id": session_id
    }

    # Call the tool
    result = client.call_tool_sync("get_chat_session", arguments)

    return result or {}

def create_chat_session(
    client: McpClient,
    agent_id: str,
    title: str,
    document_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a new chat session.

    Args:
        client: McpClient instance
        agent_id: Agent ID
        title: Session title
        document_id: Optional document ID

    Returns:
        Created chat session details
    """
    # Ensure we have an auth token
    if not client.auth_token:
        logger.error("No authentication token provided for create_chat_session")
        return {}

    # Prepare arguments
    arguments = {
        "token": client.auth_token,
        "agent_id": agent_id,
        "title": title
    }

    if document_id:
        arguments["document_id"] = document_id

    # Call the tool
    result = client.call_tool_sync("create_chat_session", arguments)

    return result or {}

def delete_chat_session(client: McpClient, session_id: str) -> bool:
    """
    Delete a chat session.

    Args:
        client: McpClient instance
        session_id: Session ID

    Returns:
        True if successful, False otherwise
    """
    # Ensure we have an auth token
    if not client.auth_token:
        logger.error("No authentication token provided for delete_chat_session")
        return False

    # Prepare arguments
    arguments = {
        "token": client.auth_token,
        "session_id": session_id
    }

    # Call the tool
    result = client.call_tool_sync("delete_chat_session", arguments)

    return result is not None and result.get("success", False)
