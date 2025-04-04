-- Fix Vector Storage Schema
-- Run this script in the Supabase SQL editor to fix vector storage issues

-- First make sure pgvector extension is enabled
CREATE EXTENSION IF NOT EXISTS "vector";

-- Create the vectors schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS vectors;

-- Drop the existing table and recreate it with the correct schema
DROP TABLE IF EXISTS vectors.transcription_embeddings;

CREATE TABLE vectors.transcription_embeddings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users ON DELETE CASCADE,
  job_id UUID NOT NULL REFERENCES user_data.audio_jobs ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  content TEXT NOT NULL,
  embedding VECTOR(384),  -- 384 dimensions for the all-MiniLM-L6-v2 model
  metadata JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Enable Row Level Security
ALTER TABLE vectors.transcription_embeddings ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist
DROP POLICY IF EXISTS "Users can view own embeddings" ON vectors.transcription_embeddings;
DROP POLICY IF EXISTS "Users can insert own embeddings" ON vectors.transcription_embeddings;

-- Vector embeddings access policies
CREATE POLICY "Users can view own embeddings"
  ON vectors.transcription_embeddings
  FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own embeddings"
  ON vectors.transcription_embeddings
  FOR INSERT
  WITH CHECK (auth.uid() = user_id);

-- Create/replace the public function for storing embeddings
CREATE OR REPLACE FUNCTION public.store_embedding(
  p_user_id UUID,
  p_job_id UUID,
  p_chunk_index INTEGER,
  p_content TEXT,
  p_embedding VECTOR(384),
  p_metadata JSONB
)
RETURNS UUID
SECURITY DEFINER
LANGUAGE plpgsql
AS $$
DECLARE
  v_id UUID;
BEGIN
  -- Insert the embedding
  INSERT INTO vectors.transcription_embeddings (
    user_id,
    job_id,
    chunk_index,
    content,
    embedding,
    metadata
  )
  VALUES (
    p_user_id,
    p_job_id,
    p_chunk_index,
    p_content,
    p_embedding,
    p_metadata
  )
  RETURNING id INTO v_id;
  
  RETURN v_id;
END;
$$;

-- Grant permissions for REST API access to vectors schema
GRANT USAGE ON SCHEMA vectors TO anon, authenticated, service_role;
GRANT SELECT ON vectors.transcription_embeddings TO anon, authenticated, service_role;
GRANT INSERT ON vectors.transcription_embeddings TO authenticated, service_role;

-- Create a function for similarity search
CREATE OR REPLACE FUNCTION public.search_transcriptions(
  p_user_id UUID,
  p_query_embedding VECTOR(384),
  p_match_threshold FLOAT DEFAULT 0.5,
  p_match_count INTEGER DEFAULT 5
)
RETURNS TABLE (
  id UUID,
  job_id UUID,
  chunk_index INTEGER,
  content TEXT,
  similarity FLOAT,
  metadata JSONB
)
SECURITY DEFINER
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    te.id,
    te.job_id,
    te.chunk_index,
    te.content,
    1 - (te.embedding <=> p_query_embedding) AS similarity,
    te.metadata
  FROM
    vectors.transcription_embeddings te
  WHERE
    te.user_id = p_user_id
    AND (1 - (te.embedding <=> p_query_embedding)) > p_match_threshold
  ORDER BY
    te.embedding <=> p_query_embedding
  LIMIT p_match_count;
END;
$$;

-- Add vector_store_settings table to user_data schema if it doesn't exist
CREATE TABLE IF NOT EXISTS user_data.vector_store_settings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    settings JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT unique_user_settings UNIQUE (user_id)
);

-- Add RLS policies for vector_store_settings
ALTER TABLE user_data.vector_store_settings ENABLE ROW LEVEL SECURITY;

-- Policy for selecting own settings
DROP POLICY IF EXISTS "Users can view own vector settings" ON user_data.vector_store_settings;
CREATE POLICY "Users can view own vector settings" ON user_data.vector_store_settings 
    FOR SELECT USING (auth.uid() = user_id);

-- Policy for inserting own settings
DROP POLICY IF EXISTS "Users can insert own vector settings" ON user_data.vector_store_settings;
CREATE POLICY "Users can insert own vector settings" ON user_data.vector_store_settings 
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Policy for updating own settings
DROP POLICY IF EXISTS "Users can update own vector settings" ON user_data.vector_store_settings;
CREATE POLICY "Users can update own vector settings" ON user_data.vector_store_settings 
    FOR UPDATE USING (auth.uid() = user_id);

-- Policy for deleting own settings
DROP POLICY IF EXISTS "Users can delete own vector settings" ON user_data.vector_store_settings;
CREATE POLICY "Users can delete own vector settings" ON user_data.vector_store_settings 
    FOR DELETE USING (auth.uid() = user_id);

-- Enable realtime for vector store tables
ALTER PUBLICATION supabase_realtime ADD TABLE vectors.transcription_embeddings;
ALTER PUBLICATION supabase_realtime ADD TABLE user_data.vector_store_settings;

-- Create a trigger to update the updated_at timestamp
CREATE OR REPLACE FUNCTION user_data.update_timestamp_function()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_vector_store_settings_timestamp
BEFORE UPDATE ON user_data.vector_store_settings
FOR EACH ROW EXECUTE FUNCTION user_data.update_timestamp_function();
