from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from typing import Dict, Any
import json
from datetime import datetime

from database.pg_connector import get_pg_db
from authentication.auth import get_current_user, User

# Create router
router = APIRouter(prefix="/mcp", tags=["mcp"])

# Pydantic models for request validation
import os
class MCPConfigurationBase(BaseModel):
    mcp_url: str = os.environ.get("MCP_URL", "http://localhost:8001")
    active_tools: Dict[str, bool] = {}
    remote_agents_enabled: bool = False
    workflow_enabled: bool = False
    mcp_status: Dict[str, Any] = {"status": "unknown", "details": {"message": "Status not checked"}}

class MCPConfigurationCreate(MCPConfigurationBase):
    pass

class MCPConfigurationUpdate(MCPConfigurationBase):
    pass

class MCPConfigurationResponse(MCPConfigurationBase):
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime

# Get MCP configuration for the current user
@router.get("/config", response_model=MCPConfigurationResponse)
async def get_mcp_configuration(current_user: User = Depends(get_current_user)):
    """Get MCP configuration for the current user"""
    try:
        # Connect to database
        db = get_pg_db()

        # Query user's MCP configuration
        query = """
        SELECT
            id,
            user_id,
            mcp_url,
            active_tools,
            remote_agents_enabled,
            workflow_enabled,
            mcp_status,
            created_at,
            updated_at
        FROM
            user_data.mcp_configurations
        WHERE
            user_id = %(user_id)s
        """

        result = db.execute_query(query, {"user_id": current_user.id})

        # If no configuration exists, return a default one
        if not result or len(result) == 0:
            # Create a default configuration
            insert_query = """
            INSERT INTO user_data.mcp_configurations
                (user_id, mcp_url, active_tools, remote_agents_enabled, workflow_enabled, mcp_status)
            VALUES
                (%(user_id)s, %(mcp_url)s, %(active_tools)s, %(remote_agents_enabled)s, %(workflow_enabled)s, %(mcp_status)s)
            RETURNING
                id, user_id, mcp_url, active_tools, remote_agents_enabled, workflow_enabled, mcp_status, created_at, updated_at
            """

            params = {
                "user_id": current_user.id,
                "mcp_url": os.environ.get("MCP_URL", "http://localhost:8001"),
                "active_tools": json.dumps({}),
                "remote_agents_enabled": False,
                "workflow_enabled": False,
                "mcp_status": json.dumps({"status": "unknown", "details": {"message": "Status not checked"}})
            }

            result = db.execute_query(insert_query, params)

        # Return the configuration
        return result[0]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get MCP configuration: {str(e)}"
        )

# Save MCP configuration for the current user
@router.post("/config", response_model=MCPConfigurationResponse)
async def save_mcp_configuration(config: MCPConfigurationUpdate, current_user: User = Depends(get_current_user)):
    """Save MCP configuration for the current user"""
    try:
        # Connect to database
        db = get_pg_db()

        # Check if configuration exists
        check_query = """
        SELECT id FROM user_data.mcp_configurations WHERE user_id = %(user_id)s
        """

        check_result = db.execute_query(check_query, {"user_id": current_user.id})

        if check_result and len(check_result) > 0:
            # Update existing configuration
            update_query = """
            UPDATE user_data.mcp_configurations
            SET
                mcp_url = %(mcp_url)s,
                active_tools = %(active_tools)s,
                remote_agents_enabled = %(remote_agents_enabled)s,
                workflow_enabled = %(workflow_enabled)s,
                mcp_status = %(mcp_status)s,
                updated_at = NOW()
            WHERE
                user_id = %(user_id)s
            RETURNING
                id, user_id, mcp_url, active_tools, remote_agents_enabled, workflow_enabled, mcp_status, created_at, updated_at
            """

            params = {
                "user_id": current_user.id,
                "mcp_url": config.mcp_url,
                "active_tools": json.dumps(config.active_tools),
                "remote_agents_enabled": config.remote_agents_enabled,
                "workflow_enabled": config.workflow_enabled,
                "mcp_status": json.dumps(config.mcp_status)
            }

            result = db.execute_query(update_query, params)
        else:
            # Insert new configuration
            insert_query = """
            INSERT INTO user_data.mcp_configurations
                (user_id, mcp_url, active_tools, remote_agents_enabled, workflow_enabled, mcp_status)
            VALUES
                (%(user_id)s, %(mcp_url)s, %(active_tools)s, %(remote_agents_enabled)s, %(workflow_enabled)s, %(mcp_status)s)
            RETURNING
                id, user_id, mcp_url, active_tools, remote_agents_enabled, workflow_enabled, mcp_status, created_at, updated_at
            """

            params = {
                "user_id": current_user.id,
                "mcp_url": config.mcp_url,
                "active_tools": json.dumps(config.active_tools),
                "remote_agents_enabled": config.remote_agents_enabled,
                "workflow_enabled": config.workflow_enabled,
                "mcp_status": json.dumps(config.mcp_status)
            }

            result = db.execute_query(insert_query, params)

        # Return the saved configuration
        return result[0]

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save MCP configuration: {str(e)}"
        )

# Delete MCP configuration for the current user
@router.delete("/config")
async def delete_mcp_configuration(current_user: User = Depends(get_current_user)):
    """Delete MCP configuration for the current user"""
    try:
        # Connect to database
        db = get_pg_db()

        # Delete configuration
        delete_query = """
        DELETE FROM user_data.mcp_configurations
        WHERE user_id = %(user_id)s
        RETURNING id
        """

        result = db.execute_query(delete_query, {"user_id": current_user.id})

        # Check if any rows were deleted
        if not result or len(result) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="MCP configuration not found"
            )

        # Return success message
        return {"message": "MCP configuration deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete MCP configuration: {str(e)}"
        )
