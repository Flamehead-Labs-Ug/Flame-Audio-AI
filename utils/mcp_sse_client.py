"""
Flame MCP SSE Client

This module provides a client for maintaining SSE connections with the Flame MCP service (flame-mcp).
"""

import logging
import threading
import time
import uuid
import requests
import json
from typing import Dict, Any, Optional, List, Callable

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp_sse_client")

class McpSseClient:
    """
    Client for maintaining SSE connections with the Flame MCP service (flame-mcp).
    """

    def __init__(self, mcp_url: str, auth_token: Optional[str] = None):
        """
        Initialize the Flame MCP SSE client.

        Args:
            mcp_url: URL of the Flame MCP service (flame-mcp)
            auth_token: Authentication token for the Flame MCP service (flame-mcp)
        """
        self.mcp_url = mcp_url.rstrip('/')
        self.auth_token = auth_token
        self.session_id = None
        self.sse_thread = None
        self.stop_event = threading.Event()
        self.connected = False
        self.message_handlers: List[Callable[[Dict[str, Any]], None]] = []

    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get authentication headers for requests to the MCP server.

        Returns:
            Dictionary of headers
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"

        return headers

    def _sse_listener(self):
        """
        Background thread function that maintains the SSE connection.
        """
        logger.info(f"Starting SSE listener for session {self.session_id}")

        try:
            # Connect to the SSE endpoint - try without trailing slash first
            sse_url = f"{self.mcp_url}/sse"
            params = {}
            if self.session_id:
                params["session_id"] = self.session_id
            headers = self.get_auth_headers()

            # Try without trailing slash
            response = requests.get(
                sse_url,
                params=params,
                headers=headers,
                stream=True,
                timeout=3600  # 1 hour timeout
            )

            # If not successful, try with trailing slash
            if response.status_code != 200:
                logger.info(f"SSE connection failed with status {response.status_code}, trying with trailing slash")
                response = requests.get(
                    f"{self.mcp_url}/sse/",
                    params=params,
                    headers=headers,
                    stream=True,
                    timeout=3600  # 1 hour timeout
                )

            # Set up a streaming connection with a long timeout
            with response:
                if response.status_code == 200:
                    logger.info(f"SSE connection established for session {self.session_id}")
                    self.connected = True

                    # Process the SSE stream
                    for line in response.iter_lines():
                        # Check if we should stop
                        if self.stop_event.is_set():
                            logger.info("Stop event received, closing SSE connection")
                            break

                        if line:
                            try:
                                # Parse the SSE event
                                line_text = line.decode('utf-8')
                                logger.debug(f"SSE line: {line_text}")

                                # Handle different event types
                                if line_text.startswith("event:"):
                                    event_type = line_text.split(":", 1)[1].strip()
                                    logger.debug(f"SSE event type: {event_type}")
                                elif line_text.startswith("data:"):
                                    data_text = line_text.split(":", 1)[1].strip()
                                    try:
                                        data = json.loads(data_text)
                                        logger.debug(f"SSE data: {data}")

                                        # Call message handlers
                                        for handler in self.message_handlers:
                                            try:
                                                handler(data)
                                            except Exception as handler_error:
                                                logger.error(f"Error in message handler: {str(handler_error)}")
                                    except json.JSONDecodeError:
                                        logger.warning(f"Failed to parse SSE data as JSON: {data_text}")
                            except Exception as parse_error:
                                logger.error(f"Error parsing SSE line: {str(parse_error)}")
                else:
                    logger.error(f"Failed to establish SSE connection: {response.status_code}")
                    self.connected = False
        except Exception as e:
            logger.error(f"Error in SSE listener: {str(e)}")
            self.connected = False

        logger.info("SSE listener stopped")
        self.connected = False

    def connect(self) -> bool:
        """
        Connect to the MCP server and establish an SSE connection.

        Returns:
            True if connection was successful, False otherwise
        """
        # If no session ID is provided, generate a new one
        if not self.session_id:
            self.session_id = str(uuid.uuid4())

        logger.info(f"Connecting with session ID: {self.session_id}")

        # First, verify the session exists or create it
        try:
            # Check if the session exists
            session_response = requests.get(
                f"{self.mcp_url}/sessions/{self.session_id}",
                headers=self.get_auth_headers(),
                timeout=5
            )

            # If the session doesn't exist, create it
            if session_response.status_code != 200:
                logger.info(f"Session {self.session_id} doesn't exist, creating it")
                create_response = requests.post(
                    f"{self.mcp_url}/sessions/{self.session_id}",
                    headers=self.get_auth_headers(),
                    timeout=5
                )

                if create_response.status_code not in [200, 201, 202]:
                    logger.warning(f"Failed to create session: {create_response.status_code} - {create_response.text}")
                    return False

                logger.info(f"Session {self.session_id} created successfully")
        except Exception as e:
            logger.warning(f"Error verifying/creating session: {str(e)}")
            # Continue anyway, as the SSE connection might still work

        # Start the SSE listener thread
        self.stop_event.clear()
        self.sse_thread = threading.Thread(target=self._sse_listener)
        self.sse_thread.daemon = True
        self.sse_thread.start()

        # Wait for the connection to be established
        for _ in range(10):  # Wait up to 5 seconds
            if self.connected:
                return True
            time.sleep(0.5)

        # If we get here, the connection wasn't established
        logger.warning("Failed to establish SSE connection within timeout")
        return False

    def disconnect(self):
        """
        Disconnect from the MCP server.
        """
        if self.sse_thread and self.sse_thread.is_alive():
            logger.info(f"Disconnecting session {self.session_id}")
            self.stop_event.set()

            # Try to close the session gracefully
            try:
                # First try the custom endpoint
                delete_response = requests.delete(
                    f"{self.mcp_url}/sessions/{self.session_id}",
                    headers=self.get_auth_headers(),
                    timeout=5
                )

                if delete_response.status_code in [200, 202, 204]:
                    logger.info(f"Session {self.session_id} deleted successfully")
                else:
                    # Fallback to the SSE endpoint
                    logger.warning(f"Failed to delete session via custom endpoint: {delete_response.status_code}")
                    sse_response = requests.delete(
                        f"{self.mcp_url}/sse/",
                        params={"session_id": self.session_id},
                        headers=self.get_auth_headers(),
                        timeout=5
                    )

                    if sse_response.status_code in [200, 202, 204]:
                        logger.info(f"Session {self.session_id} closed via SSE endpoint")
                    else:
                        logger.warning(f"Failed to close session via SSE endpoint: {sse_response.status_code}")
            except Exception as e:
                logger.warning(f"Error closing session: {str(e)}")

            # Wait for the thread to stop
            self.sse_thread.join(timeout=5)
            if self.sse_thread.is_alive():
                logger.warning("SSE thread did not stop gracefully")

        self.connected = False

    def add_message_handler(self, handler: Callable[[Dict[str, Any]], None]):
        """
        Add a handler for SSE messages.

        Args:
            handler: Function that takes a message dictionary and returns None
        """
        self.message_handlers.append(handler)

    def send_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a message to the MCP server.

        Args:
            message: Message to send

        Returns:
            Response from the server
        """
        if not self.connected:
            logger.warning("Not connected to MCP server")
            return {"success": False, "error": "Not connected to MCP server"}

        # Ensure we have a session ID
        if not self.session_id:
            import uuid
            self.session_id = str(uuid.uuid4())
            logger.warning(f"No session ID provided, generated new ID: {self.session_id}")

        try:
            # Try multiple endpoint variations to ensure compatibility
            endpoints_to_try = [
                # Standard endpoint without trailing slash
                (f"{self.mcp_url}/messages", {"session_id": self.session_id}),
                # Standard endpoint with trailing slash
                (f"{self.mcp_url}/messages/", {"session_id": self.session_id}),
                # Custom endpoint without trailing slash
                (f"{self.mcp_url}/messages/{self.session_id}", {}),
                # Custom endpoint with trailing slash
                (f"{self.mcp_url}/messages/{self.session_id}/", {})
            ]

            for endpoint, params in endpoints_to_try:
                try:
                    logger.info(f"Sending message to endpoint: {endpoint}")
                    response = requests.post(
                        endpoint,
                        json=message,
                        params=params,
                        headers=self.get_auth_headers(),
                        timeout=30
                    )

                    if response.status_code in [200, 201, 202]:
                        logger.info(f"Message sent successfully to {endpoint}")
                        return {"success": True}
                    else:
                        logger.warning(f"Failed to send message to {endpoint}: {response.status_code}")
                except Exception as e:
                    logger.warning(f"Error sending message to {endpoint}: {str(e)}")
                    continue

            # If we get here, all endpoints failed
            logger.error("All message endpoints failed")
            return {"success": False, "error": "Failed to send message to any endpoint"}
        except Exception as e:
            logger.error(f"Error sending message: {str(e)}")
            return {"success": False, "error": f"Error: {str(e)}"}

    def get_messages(self) -> Dict[str, Any]:
        """
        Get messages from the MCP server.

        Returns:
            Dictionary with messages
        """
        if not self.connected:
            logger.warning("Not connected to MCP server")
            return {"success": False, "error": "Not connected to MCP server"}

        # Ensure we have a session ID
        if not self.session_id:
            import uuid
            self.session_id = str(uuid.uuid4())
            logger.warning(f"No session ID provided, generated new ID: {self.session_id}")

        try:
            # Try multiple endpoint variations to ensure compatibility
            endpoints_to_try = [
                # Standard endpoint without trailing slash
                (f"{self.mcp_url}/messages", {"session_id": self.session_id}),
                # Standard endpoint with trailing slash
                (f"{self.mcp_url}/messages/", {"session_id": self.session_id}),
                # Custom endpoint without trailing slash
                (f"{self.mcp_url}/messages/{self.session_id}", {}),
                # Custom endpoint with trailing slash
                (f"{self.mcp_url}/messages/{self.session_id}/", {})
            ]

            for endpoint, params in endpoints_to_try:
                try:
                    logger.info(f"Getting messages from endpoint: {endpoint}")
                    response = requests.get(
                        endpoint,
                        params=params,
                        headers=self.get_auth_headers(),
                        timeout=10
                    )

                    if response.status_code == 200:
                        logger.info(f"Messages retrieved successfully from {endpoint}")
                        return {"success": True, "messages": response.json().get("messages", [])}
                    else:
                        logger.warning(f"Failed to get messages from {endpoint}: {response.status_code}")
                except Exception as e:
                    logger.warning(f"Error getting messages from {endpoint}: {str(e)}")
                    continue

            # If we get here, all endpoints failed
            logger.error("All message endpoints failed")
            return {"success": False, "error": "Failed to get messages from any endpoint"}
        except Exception as e:
            logger.error(f"Error getting messages: {str(e)}")
            return {"success": False, "error": f"Error: {str(e)}"}

    def call_tool(self, tool_name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            Tool result or error
        """
        if not self.connected:
            logger.warning("Not connected to MCP server")
            return {"success": False, "error": "Not connected to MCP server"}

        if arguments is None:
            arguments = {}

        # Add auth_token to arguments if available
        if self.auth_token and "auth_token" not in arguments:
            arguments["auth_token"] = self.auth_token

        # Construct a message that will trigger the tool
        payload = {
            "messages": [{"role": "user", "content": f"Call {tool_name}"}],
            "tools": [{"name": tool_name}]
        }

        # If arguments are provided, add them to the payload
        if arguments:
            payload["arguments"] = arguments

        # Send the message
        send_result = self.send_message(payload)
        if not send_result.get("success", False):
            return send_result

        # Poll for the response
        max_retries = 30
        retry_count = 0

        while retry_count < max_retries:
            # Wait before polling
            time.sleep(1)

            # Get the messages
            messages_result = self.get_messages()
            if not messages_result.get("success", False):
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
                                return {"success": True, "result": result}
                            except json.JSONDecodeError:
                                logger.error(f"Failed to parse tool result: {tool_call.get('arguments')}")
                                return {"success": False, "error": "Failed to parse tool result"}

            # Increment retry count
            retry_count += 1

        # If we've exhausted retries, return timeout error
        logger.warning(f"Timeout waiting for tool response: {tool_name}")
        return {"success": False, "error": "Timeout waiting for tool response"}
