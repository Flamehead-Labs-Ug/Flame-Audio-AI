"""
Script to create the MCP configuration table in the Supabase database.
"""

import os
from dotenv import load_dotenv
from pg_connector import get_pg_db

# Load environment variables
load_dotenv()

def create_mcp_table():
    """Create the MCP configuration table in the database."""
    try:
        # Connect to database
        db = get_pg_db()

        # SQL to create the MCP configurations table
        create_table_sql = """
        -- MCP Configuration Schema
        -- This script creates the necessary tables and policies for storing MCP configurations

        -- Create MCP configurations table
        CREATE TABLE IF NOT EXISTS user_data.mcp_configurations (
          id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
          user_id UUID NOT NULL REFERENCES auth.users ON DELETE CASCADE,
          -- The default should be set via the MCP_URL environment variable in application config
mcp_url TEXT NOT NULL DEFAULT 'http://localhost:8001',
          active_tools JSONB DEFAULT '{}'::jsonb,
          remote_agents_enabled BOOLEAN DEFAULT false,
          workflow_enabled BOOLEAN DEFAULT false,
          created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
          updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
        );

        -- Enable Row Level Security
        ALTER TABLE user_data.mcp_configurations ENABLE ROW LEVEL SECURITY;

        -- MCP configurations access policies
        CREATE POLICY "Users can view own MCP configurations"
          ON user_data.mcp_configurations
          FOR SELECT
          USING (auth.uid() = user_id);

        CREATE POLICY "Users can update own MCP configurations"
          ON user_data.mcp_configurations
          FOR UPDATE
          USING (auth.uid() = user_id);

        CREATE POLICY "Users can insert own MCP configurations"
          ON user_data.mcp_configurations
          FOR INSERT
          WITH CHECK (auth.uid() = user_id);

        CREATE POLICY "Users can delete own MCP configurations"
          ON user_data.mcp_configurations
          FOR DELETE
          USING (auth.uid() = user_id);

        -- Add updated_at trigger
        CREATE TRIGGER mcp_configurations_updated_at
          BEFORE UPDATE ON user_data.mcp_configurations
          FOR EACH ROW
          EXECUTE FUNCTION public.handle_updated_at();

        -- Create index for faster lookups
        CREATE INDEX IF NOT EXISTS idx_mcp_configurations_user_id
          ON user_data.mcp_configurations (user_id);
        """

        # Execute the SQL
        db.execute_query(create_table_sql)

        print("MCP configuration table created successfully!")

    except Exception as e:
        print(f"Error creating MCP configuration table: {str(e)}")
        raise

if __name__ == "__main__":
    create_mcp_table()
