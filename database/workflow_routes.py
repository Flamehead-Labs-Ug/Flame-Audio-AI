"""
Workflow Routes

This module provides API routes for managing workflows in the database.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

from database.pg_connector import get_pg_db
from authentication.auth import get_current_user, User

# Create router
router = APIRouter(prefix="/workflows", tags=["workflows"])

# Pydantic models for request validation
class WorkflowBase(BaseModel):
    name: str
    type: str
    description: Optional[str] = None
    configuration: Dict[str, Any]

class WorkflowCreate(WorkflowBase):
    pass

class WorkflowUpdate(WorkflowBase):
    pass

class WorkflowResponse(WorkflowBase):
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime

class WorkflowExecutionCreate(BaseModel):
    workflow_id: str
    initial_context: Optional[Dict[str, Any]] = None

class WorkflowExecutionResponse(BaseModel):
    id: str
    workflow_id: str
    user_id: str
    status: str
    start_time: datetime
    end_time: Optional[datetime] = None
    initial_context: Optional[Dict[str, Any]] = None
    result_context: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime

# Get all workflows for the current user
@router.get("/", response_model=List[WorkflowResponse])
async def get_workflows(current_user: User = Depends(get_current_user)):
    """Get all workflows for the current user"""
    try:
        db = get_pg_db()
        
        query = """
        SELECT
            id,
            user_id,
            name,
            type,
            description,
            configuration,
            created_at,
            updated_at
        FROM
            user_data.workflows
        WHERE
            user_id = %(user_id)s
        ORDER BY
            created_at DESC
        """
        
        result = db.execute_query(query, {"user_id": current_user.id})
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflows: {str(e)}"
        )

# Get a specific workflow
@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(workflow_id: str, current_user: User = Depends(get_current_user)):
    """Get a specific workflow"""
    try:
        db = get_pg_db()
        
        query = """
        SELECT
            id,
            user_id,
            name,
            type,
            description,
            configuration,
            created_at,
            updated_at
        FROM
            user_data.workflows
        WHERE
            id = %(workflow_id)s
            AND user_id = %(user_id)s
        """
        
        result = db.execute_query(query, {
            "workflow_id": workflow_id,
            "user_id": current_user.id
        })
        
        if not result or len(result) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow with ID {workflow_id} not found"
            )
            
        return result[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow: {str(e)}"
        )

# Create a new workflow
@router.post("/", response_model=WorkflowResponse)
async def create_workflow(workflow: WorkflowCreate, current_user: User = Depends(get_current_user)):
    """Create a new workflow"""
    try:
        db = get_pg_db()
        
        # Validate workflow type
        if workflow.type not in ["sequential", "parallel", "loop"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid workflow type: {workflow.type}. Must be one of: sequential, parallel, loop"
            )
        
        query = """
        INSERT INTO user_data.workflows
            (user_id, name, type, description, configuration)
        VALUES
            (%(user_id)s, %(name)s, %(type)s, %(description)s, %(configuration)s)
        RETURNING
            id, user_id, name, type, description, configuration, created_at, updated_at
        """
        
        params = {
            "user_id": current_user.id,
            "name": workflow.name,
            "type": workflow.type,
            "description": workflow.description,
            "configuration": json.dumps(workflow.configuration)
        }
        
        result = db.execute_query(query, params)
        return result[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create workflow: {str(e)}"
        )

# Update a workflow
@router.put("/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(workflow_id: str, workflow: WorkflowUpdate, current_user: User = Depends(get_current_user)):
    """Update a workflow"""
    try:
        db = get_pg_db()
        
        # Validate workflow type
        if workflow.type not in ["sequential", "parallel", "loop"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid workflow type: {workflow.type}. Must be one of: sequential, parallel, loop"
            )
        
        # Check if workflow exists and belongs to the user
        check_query = """
        SELECT id FROM user_data.workflows
        WHERE id = %(workflow_id)s AND user_id = %(user_id)s
        """
        
        check_result = db.execute_query(check_query, {
            "workflow_id": workflow_id,
            "user_id": current_user.id
        })
        
        if not check_result or len(check_result) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow with ID {workflow_id} not found"
            )
        
        # Update the workflow
        update_query = """
        UPDATE user_data.workflows
        SET
            name = %(name)s,
            type = %(type)s,
            description = %(description)s,
            configuration = %(configuration)s,
            updated_at = NOW()
        WHERE
            id = %(workflow_id)s
            AND user_id = %(user_id)s
        RETURNING
            id, user_id, name, type, description, configuration, created_at, updated_at
        """
        
        params = {
            "workflow_id": workflow_id,
            "user_id": current_user.id,
            "name": workflow.name,
            "type": workflow.type,
            "description": workflow.description,
            "configuration": json.dumps(workflow.configuration)
        }
        
        result = db.execute_query(update_query, params)
        return result[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update workflow: {str(e)}"
        )

# Delete a workflow
@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: str, current_user: User = Depends(get_current_user)):
    """Delete a workflow"""
    try:
        db = get_pg_db()
        
        # Check if workflow exists and belongs to the user
        check_query = """
        SELECT id FROM user_data.workflows
        WHERE id = %(workflow_id)s AND user_id = %(user_id)s
        """
        
        check_result = db.execute_query(check_query, {
            "workflow_id": workflow_id,
            "user_id": current_user.id
        })
        
        if not check_result or len(check_result) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow with ID {workflow_id} not found"
            )
        
        # Delete the workflow
        delete_query = """
        DELETE FROM user_data.workflows
        WHERE id = %(workflow_id)s AND user_id = %(user_id)s
        """
        
        db.execute_query(delete_query, {
            "workflow_id": workflow_id,
            "user_id": current_user.id
        })
        
        return {"message": f"Workflow with ID {workflow_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete workflow: {str(e)}"
        )

# Create a workflow execution
@router.post("/executions", response_model=WorkflowExecutionResponse)
async def create_workflow_execution(execution: WorkflowExecutionCreate, current_user: User = Depends(get_current_user)):
    """Create a workflow execution"""
    try:
        db = get_pg_db()
        
        # Check if workflow exists and belongs to the user
        check_query = """
        SELECT id FROM user_data.workflows
        WHERE id = %(workflow_id)s AND user_id = %(user_id)s
        """
        
        check_result = db.execute_query(check_query, {
            "workflow_id": execution.workflow_id,
            "user_id": current_user.id
        })
        
        if not check_result or len(check_result) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow with ID {execution.workflow_id} not found"
            )
        
        # Create the execution
        insert_query = """
        INSERT INTO user_data.workflow_executions
            (workflow_id, user_id, status, initial_context)
        VALUES
            (%(workflow_id)s, %(user_id)s, 'running', %(initial_context)s)
        RETURNING
            id, workflow_id, user_id, status, start_time, end_time, 
            initial_context, result_context, error_message, created_at
        """
        
        params = {
            "workflow_id": execution.workflow_id,
            "user_id": current_user.id,
            "initial_context": json.dumps(execution.initial_context) if execution.initial_context else None
        }
        
        result = db.execute_query(insert_query, params)
        return result[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create workflow execution: {str(e)}"
        )

# Get workflow executions for a workflow
@router.get("/{workflow_id}/executions", response_model=List[WorkflowExecutionResponse])
async def get_workflow_executions(workflow_id: str, current_user: User = Depends(get_current_user)):
    """Get workflow executions for a workflow"""
    try:
        db = get_pg_db()
        
        query = """
        SELECT
            id, workflow_id, user_id, status, start_time, end_time,
            initial_context, result_context, error_message, created_at
        FROM
            user_data.workflow_executions
        WHERE
            workflow_id = %(workflow_id)s
            AND user_id = %(user_id)s
        ORDER BY
            created_at DESC
        """
        
        result = db.execute_query(query, {
            "workflow_id": workflow_id,
            "user_id": current_user.id
        })
        
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow executions: {str(e)}"
        )

# Get a specific workflow execution
@router.get("/executions/{execution_id}", response_model=WorkflowExecutionResponse)
async def get_workflow_execution(execution_id: str, current_user: User = Depends(get_current_user)):
    """Get a specific workflow execution"""
    try:
        db = get_pg_db()
        
        query = """
        SELECT
            id, workflow_id, user_id, status, start_time, end_time,
            initial_context, result_context, error_message, created_at
        FROM
            user_data.workflow_executions
        WHERE
            id = %(execution_id)s
            AND user_id = %(user_id)s
        """
        
        result = db.execute_query(query, {
            "execution_id": execution_id,
            "user_id": current_user.id
        })
        
        if not result or len(result) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow execution with ID {execution_id} not found"
            )
            
        return result[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow execution: {str(e)}"
        )

# Update a workflow execution
@router.put("/executions/{execution_id}", response_model=WorkflowExecutionResponse)
async def update_workflow_execution(
    execution_id: str, 
    status: str, 
    result_context: Optional[Dict[str, Any]] = None,
    error_message: Optional[str] = None,
    current_user: User = Depends(get_current_user)
):
    """Update a workflow execution"""
    try:
        db = get_pg_db()
        
        # Validate status
        if status not in ["running", "completed", "failed"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status: {status}. Must be one of: running, completed, failed"
            )
        
        # Check if execution exists and belongs to the user
        check_query = """
        SELECT id FROM user_data.workflow_executions
        WHERE id = %(execution_id)s AND user_id = %(user_id)s
        """
        
        check_result = db.execute_query(check_query, {
            "execution_id": execution_id,
            "user_id": current_user.id
        })
        
        if not check_result or len(check_result) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow execution with ID {execution_id} not found"
            )
        
        # Update the execution
        update_query = """
        UPDATE user_data.workflow_executions
        SET
            status = %(status)s,
            result_context = %(result_context)s,
            error_message = %(error_message)s
        """
        
        # If status is completed or failed, set end_time
        if status in ["completed", "failed"]:
            update_query += ", end_time = NOW()"
            
        update_query += """
        WHERE
            id = %(execution_id)s
            AND user_id = %(user_id)s
        RETURNING
            id, workflow_id, user_id, status, start_time, end_time,
            initial_context, result_context, error_message, created_at
        """
        
        params = {
            "execution_id": execution_id,
            "user_id": current_user.id,
            "status": status,
            "result_context": json.dumps(result_context) if result_context else None,
            "error_message": error_message
        }
        
        result = db.execute_query(update_query, params)
        return result[0]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update workflow execution: {str(e)}"
        )
