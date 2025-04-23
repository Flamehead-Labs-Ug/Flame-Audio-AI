-- Workflow and Remote Agent Schema for Supabase
-- Run this in the Supabase SQL Editor to create the necessary tables and functions

-- Create schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS user_data;

-- =============================================
-- Workflow Tables
-- =============================================

-- Workflow definitions table
CREATE TABLE IF NOT EXISTS user_data.workflows (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users ON DELETE CASCADE,
  name TEXT NOT NULL,
  type TEXT NOT NULL CHECK (type IN ('sequential', 'parallel', 'loop')),
  description TEXT,
  configuration JSONB NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Workflow execution history
CREATE TABLE IF NOT EXISTS user_data.workflow_executions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  workflow_id UUID NOT NULL REFERENCES user_data.workflows(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES auth.users ON DELETE CASCADE,
  status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed')),
  start_time TIMESTAMP WITH TIME ZONE DEFAULT now(),
  end_time TIMESTAMP WITH TIME ZONE,
  initial_context JSONB,
  result_context JSONB,
  error_message TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- =============================================
-- Remote Agent Tables
-- =============================================

-- Remote agent registrations
CREATE TABLE IF NOT EXISTS user_data.remote_agents (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users ON DELETE CASCADE,
  name TEXT NOT NULL,
  url TEXT NOT NULL,
  api_key TEXT,
  description TEXT,
  status TEXT NOT NULL DEFAULT 'active',
  capabilities JSONB,
  last_heartbeat TIMESTAMP WITH TIME ZONE DEFAULT now(),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Remote agent tool invocations
CREATE TABLE IF NOT EXISTS user_data.remote_agent_invocations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  agent_id UUID NOT NULL REFERENCES user_data.remote_agents(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES auth.users ON DELETE CASCADE,
  tool_name TEXT NOT NULL,
  parameters JSONB,
  result JSONB,
  status TEXT NOT NULL CHECK (status IN ('pending', 'success', 'failed')),
  error_message TEXT,
  execution_time FLOAT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- =============================================
-- Update MCP Configuration Table
-- =============================================

-- Add remote_agents_enabled column to MCP configurations if it doesn't exist
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_schema = 'user_data' 
    AND table_name = 'mcp_configurations' 
    AND column_name = 'remote_agents_enabled'
  ) THEN
    ALTER TABLE user_data.mcp_configurations 
    ADD COLUMN remote_agents_enabled BOOLEAN DEFAULT false;
  END IF;
END $$;

-- Add workflow_enabled column to MCP configurations if it doesn't exist
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns 
    WHERE table_schema = 'user_data' 
    AND table_name = 'mcp_configurations' 
    AND column_name = 'workflow_enabled'
  ) THEN
    ALTER TABLE user_data.mcp_configurations 
    ADD COLUMN workflow_enabled BOOLEAN DEFAULT false;
  END IF;
END $$;

-- =============================================
-- Functions
-- =============================================

-- Function to update the 'updated_at' timestamp
CREATE OR REPLACE FUNCTION user_data.set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =============================================
-- Triggers
-- =============================================

-- Trigger for workflows table
DROP TRIGGER IF EXISTS set_workflows_updated_at ON user_data.workflows;
CREATE TRIGGER set_workflows_updated_at
BEFORE UPDATE ON user_data.workflows
FOR EACH ROW
EXECUTE FUNCTION user_data.set_updated_at();

-- Trigger for remote_agents table
DROP TRIGGER IF EXISTS set_remote_agents_updated_at ON user_data.remote_agents;
CREATE TRIGGER set_remote_agents_updated_at
BEFORE UPDATE ON user_data.remote_agents
FOR EACH ROW
EXECUTE FUNCTION user_data.set_updated_at();

-- =============================================
-- Row Level Security Policies
-- =============================================

-- Enable Row Level Security
ALTER TABLE user_data.workflows ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_data.workflow_executions ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_data.remote_agents ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_data.remote_agent_invocations ENABLE ROW LEVEL SECURITY;

-- Workflow policies
CREATE POLICY "Users can view own workflows" 
  ON user_data.workflows FOR SELECT 
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own workflows" 
  ON user_data.workflows FOR INSERT 
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own workflows" 
  ON user_data.workflows FOR UPDATE 
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own workflows" 
  ON user_data.workflows FOR DELETE 
  USING (auth.uid() = user_id);

-- Workflow execution policies
CREATE POLICY "Users can view own workflow executions" 
  ON user_data.workflow_executions FOR SELECT 
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own workflow executions" 
  ON user_data.workflow_executions FOR INSERT 
  WITH CHECK (auth.uid() = user_id);

-- Remote agent policies
CREATE POLICY "Users can view own remote agents" 
  ON user_data.remote_agents FOR SELECT 
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own remote agents" 
  ON user_data.remote_agents FOR INSERT 
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own remote agents" 
  ON user_data.remote_agents FOR UPDATE 
  USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own remote agents" 
  ON user_data.remote_agents FOR DELETE 
  USING (auth.uid() = user_id);

-- Remote agent invocation policies
CREATE POLICY "Users can view own remote agent invocations" 
  ON user_data.remote_agent_invocations FOR SELECT 
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own remote agent invocations" 
  ON user_data.remote_agent_invocations FOR INSERT 
  WITH CHECK (auth.uid() = user_id);

-- =============================================
-- Indexes
-- =============================================

-- Workflow indexes
CREATE INDEX IF NOT EXISTS workflows_user_id_idx ON user_data.workflows(user_id);
CREATE INDEX IF NOT EXISTS workflows_type_idx ON user_data.workflows(type);

-- Workflow execution indexes
CREATE INDEX IF NOT EXISTS workflow_executions_workflow_id_idx ON user_data.workflow_executions(workflow_id);
CREATE INDEX IF NOT EXISTS workflow_executions_user_id_idx ON user_data.workflow_executions(user_id);
CREATE INDEX IF NOT EXISTS workflow_executions_status_idx ON user_data.workflow_executions(status);

-- Remote agent indexes
CREATE INDEX IF NOT EXISTS remote_agents_user_id_idx ON user_data.remote_agents(user_id);
CREATE INDEX IF NOT EXISTS remote_agents_status_idx ON user_data.remote_agents(status);

-- Remote agent invocation indexes
CREATE INDEX IF NOT EXISTS remote_agent_invocations_agent_id_idx ON user_data.remote_agent_invocations(agent_id);
CREATE INDEX IF NOT EXISTS remote_agent_invocations_user_id_idx ON user_data.remote_agent_invocations(user_id);
CREATE INDEX IF NOT EXISTS remote_agent_invocations_status_idx ON user_data.remote_agent_invocations(status);
