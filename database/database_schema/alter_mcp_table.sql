-- Add mcp_status column to mcp_configurations table
ALTER TABLE user_data.mcp_configurations 
ADD COLUMN IF NOT EXISTS mcp_status JSONB DEFAULT '{"status": "unknown", "details": {"message": "Status not checked"}}'::jsonb;
