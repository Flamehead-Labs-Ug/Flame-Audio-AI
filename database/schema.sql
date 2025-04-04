-- Flame Audio Platform Database Schema (Fixed Version)
-- This script creates the necessary tables, functions, and policies for user management

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS user_data;
CREATE SCHEMA IF NOT EXISTS vectors;

-----------------------------------------------
-- USER MANAGEMENT TABLES
-----------------------------------------------

-- User Profiles Table
CREATE TABLE IF NOT EXISTS public.profiles (
  id UUID PRIMARY KEY REFERENCES auth.users ON DELETE CASCADE,
  full_name TEXT,
  avatar_url TEXT,
  username TEXT UNIQUE,
  preferred_language TEXT DEFAULT 'en',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Enable Row Level Security
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

-- Profile access policies
CREATE POLICY "Users can view their own profile"
  ON public.profiles
  FOR SELECT
  USING (auth.uid() = id);

CREATE POLICY "Users can update own profile"
  ON public.profiles
  FOR UPDATE
  USING (auth.uid() = id);

-- Add profile update trigger
CREATE OR REPLACE FUNCTION public.handle_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER profiles_updated_at
  BEFORE UPDATE ON public.profiles
  FOR EACH ROW
  EXECUTE FUNCTION public.handle_updated_at();

-- Add user creation trigger
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER AS $$
BEGIN
  INSERT INTO public.profiles (
    id, 
    full_name, 
    avatar_url,
    preferred_language
  )
  VALUES (
    NEW.id, 
    NEW.raw_user_meta_data->>'full_name', 
    NEW.raw_user_meta_data->>'avatar_url',
    coalesce(NEW.raw_user_meta_data->>'preferred_language', 'en')
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER SET search_path = '';

CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW
  EXECUTE FUNCTION public.handle_new_user();

-----------------------------------------------
-- SESSION MANAGEMENT
-----------------------------------------------

-- User Sessions Table
CREATE TABLE IF NOT EXISTS user_data.sessions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users ON DELETE CASCADE,
  token TEXT NOT NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
  user_agent TEXT,
  ip_address TEXT,
  is_valid BOOLEAN DEFAULT TRUE
);

-- Enable Row Level Security
ALTER TABLE user_data.sessions ENABLE ROW LEVEL SECURITY;

-- Session access policies
CREATE POLICY "Users can view own sessions"
  ON user_data.sessions
  FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "System can insert sessions"
  ON user_data.sessions
  FOR INSERT
  WITH CHECK (TRUE);

CREATE POLICY "System can update sessions"
  ON user_data.sessions
  FOR UPDATE
  USING (TRUE);

-- Session Functions
-- Fixed: Removed default parameter for consistency
CREATE OR REPLACE FUNCTION user_data.create_session(p_user_id UUID, p_expires_interval TEXT)
RETURNS UUID
SECURITY DEFINER
LANGUAGE plpgsql
AS $$
DECLARE
  v_session_id UUID;
  v_token TEXT;
BEGIN
  -- Generate a unique session ID and token
  v_session_id := gen_random_uuid();
  v_token := encode(gen_random_bytes(32), 'hex');
  
  -- Insert the new session
  INSERT INTO user_data.sessions (
    id,
    user_id,
    token,
    expires_at,
    user_agent,
    ip_address
  )
  VALUES (
    v_session_id,
    p_user_id,
    v_token,
    now() + p_expires_interval::interval,
    current_setting('request.headers', true)::json->>'user-agent',
    current_setting('request.headers', true)::json->>'x-forwarded-for'
  );
  
  RETURN v_session_id;
END;
$$;

CREATE OR REPLACE FUNCTION user_data.validate_session(p_token TEXT)
RETURNS UUID
SECURITY DEFINER
LANGUAGE plpgsql
AS $$
DECLARE
  v_user_id UUID;
BEGIN
  -- Get the user ID if session is valid
  SELECT user_id INTO v_user_id
  FROM user_data.sessions
  WHERE token = p_token
    AND expires_at > now()
    AND is_valid = TRUE;
  
  IF v_user_id IS NULL THEN
    RAISE EXCEPTION 'Invalid or expired session';
  END IF;
  
  -- Update the session with latest user agent and IP
  UPDATE user_data.sessions
  SET 
    user_agent = current_setting('request.headers', true)::json->>'user-agent',
    ip_address = current_setting('request.headers', true)::json->>'x-forwarded-for'
  WHERE token = p_token;
  
  RETURN v_user_id;
END;
$$;

CREATE OR REPLACE FUNCTION user_data.invalidate_session(p_token TEXT)
RETURNS BOOLEAN
SECURITY DEFINER
LANGUAGE plpgsql
AS $$
BEGIN
  -- Mark session as invalid
  UPDATE user_data.sessions
  SET is_valid = FALSE
  WHERE token = p_token;
  
  IF FOUND THEN
    RETURN TRUE;
  ELSE
    RETURN FALSE;
  END IF;
END;
$$;

-----------------------------------------------
-- USER AGENTS
-----------------------------------------------

-- Create table for user agents
CREATE TABLE IF NOT EXISTS user_data.user_agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users ON DELETE CASCADE,
    name TEXT NOT NULL,
    system_message TEXT,
    settings JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(user_id, name)
);

-- Enable Row Level Security
ALTER TABLE user_data.user_agents ENABLE ROW LEVEL SECURITY;

-- Create policy for select
CREATE POLICY "Users can view own agents"
    ON user_data.user_agents
    FOR SELECT
    USING (auth.uid() = user_id);

-- Create policy for insert/update
CREATE POLICY "Users can create and update own agents"
    ON user_data.user_agents
    FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own agents"
    ON user_data.user_agents
    FOR UPDATE
    USING (auth.uid() = user_id);

-----------------------------------------------
-- TRANSCRIPTIONS AND TRANSLATIONS
-----------------------------------------------

-- Audio Jobs Table
CREATE TABLE IF NOT EXISTS user_data.audio_jobs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users ON DELETE CASCADE,
  session_id UUID REFERENCES user_data.sessions ON DELETE SET NULL,
  agent_id UUID REFERENCES user_data.user_agents(id) ON DELETE SET NULL,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now(),
  file_name TEXT NOT NULL,
  task_type TEXT NOT NULL CHECK (task_type IN ('transcribe', 'translate')),
  original_language TEXT,
  file_type TEXT,
  file_size_bytes INTEGER,
  settings JSONB DEFAULT '{}'::jsonb
);

-- Enable Row Level Security
ALTER TABLE user_data.audio_jobs ENABLE ROW LEVEL SECURITY;

-- Audio Jobs access policies
CREATE POLICY "Users can view own audio jobs"
  ON user_data.audio_jobs
  FOR SELECT
  USING (auth.uid() = user_id);

CREATE POLICY "Users can create own audio jobs"
  ON user_data.audio_jobs
  FOR INSERT
  WITH CHECK (auth.uid() = user_id);

CREATE POLICY "System can update audio jobs"
  ON user_data.audio_jobs
  FOR UPDATE
  USING (TRUE);

-- Audio Job Functions
CREATE OR REPLACE FUNCTION user_data.create_audio_job(
  p_user_id UUID,
  p_session_id UUID,
  p_agent_id UUID,
  p_file_name TEXT,
  p_task_type TEXT,
  p_original_language TEXT,
  p_file_type TEXT,
  p_file_size_bytes INTEGER,
  p_duration_seconds FLOAT,
  p_settings JSONB
)
RETURNS UUID
SECURITY DEFINER
LANGUAGE plpgsql
AS $$
DECLARE
  v_job_id UUID;
BEGIN
  -- Insert audio job
  INSERT INTO user_data.audio_jobs (
    user_id,
    session_id,
    agent_id,
    file_name,
    task_type,
    original_language,
    file_type,
    file_size_bytes,
    settings
  )
  VALUES (
    p_user_id,
    p_session_id,
    p_agent_id,
    p_file_name,
    p_task_type,
    p_original_language,
    p_file_type,
    p_file_size_bytes,
    p_settings
  )
  RETURNING id INTO v_job_id;
  
  RETURN v_job_id;
END;
$$;

-- Transcriptions Table
CREATE TABLE IF NOT EXISTS user_data.transcriptions (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id UUID NOT NULL REFERENCES user_data.audio_jobs ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  start_time FLOAT,
  end_time FLOAT,
  text TEXT NOT NULL,
  language TEXT,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Enable Row Level Security
ALTER TABLE user_data.transcriptions ENABLE ROW LEVEL SECURITY;

-- Transcriptions access policies
CREATE POLICY "Users can view own transcriptions"
  ON user_data.transcriptions
  FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM user_data.audio_jobs 
      WHERE user_data.audio_jobs.id = user_data.transcriptions.job_id 
      AND user_data.audio_jobs.user_id = auth.uid()
    )
  );

CREATE POLICY "System can insert transcriptions"
  ON user_data.transcriptions
  FOR INSERT
  WITH CHECK (TRUE);

-- Transcription Functions
-- Fixed: All parameters have proper defaults
CREATE OR REPLACE FUNCTION user_data.save_transcription(
  p_job_id UUID,
  p_chunk_index INTEGER,
  p_start_time FLOAT,
  p_end_time FLOAT,
  p_text TEXT,
  p_language TEXT
)
RETURNS UUID
SECURITY DEFINER
LANGUAGE plpgsql
AS $$
DECLARE
  v_transcription_id UUID;
BEGIN
  -- Insert the transcription
  INSERT INTO user_data.transcriptions (
    job_id,
    chunk_index,
    start_time,
    end_time,
    text,
    language
  )
  VALUES (
    p_job_id,
    p_chunk_index,
    p_start_time,
    p_end_time,
    p_text,
    p_language
  )
  RETURNING id INTO v_transcription_id;
  
  RETURN v_transcription_id;
END;
$$;

-- Translations Table
CREATE TABLE IF NOT EXISTS user_data.translations (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  job_id UUID NOT NULL REFERENCES user_data.audio_jobs ON DELETE CASCADE,
  transcription_id UUID REFERENCES user_data.transcriptions ON DELETE SET NULL,
  chunk_index INTEGER NOT NULL,
  start_time FLOAT,
  end_time FLOAT,
  original_text TEXT,
  translated_text TEXT NOT NULL,
  source_language TEXT,
  target_language TEXT DEFAULT 'en',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Enable Row Level Security
ALTER TABLE user_data.translations ENABLE ROW LEVEL SECURITY;

-- Translations access policies
CREATE POLICY "Users can view own translations"
  ON user_data.translations
  FOR SELECT
  USING (
    EXISTS (
      SELECT 1 FROM user_data.audio_jobs 
      WHERE user_data.audio_jobs.id = user_data.translations.job_id 
      AND user_data.audio_jobs.user_id = auth.uid()
    )
  );

CREATE POLICY "System can insert translations"
  ON user_data.translations
  FOR INSERT
  WITH CHECK (TRUE);

-- Translation Functions
-- Fixed: All parameters have defaults removed for consistency
CREATE OR REPLACE FUNCTION user_data.save_translation(
  p_job_id UUID,
  p_transcription_id UUID,
  p_chunk_index INTEGER,
  p_start_time FLOAT,
  p_end_time FLOAT,
  p_original_text TEXT,
  p_translated_text TEXT,
  p_source_language TEXT,
  p_target_language TEXT
)
RETURNS UUID
SECURITY DEFINER
LANGUAGE plpgsql
AS $$
DECLARE
  v_translation_id UUID;
BEGIN
  -- Insert the translation
  INSERT INTO user_data.translations (
    job_id,
    transcription_id,
    chunk_index,
    start_time,
    end_time,
    original_text,
    translated_text,
    source_language,
    target_language
  )
  VALUES (
    p_job_id,
    p_transcription_id,
    p_chunk_index,
    p_start_time,
    p_end_time,
    p_original_text,
    p_translated_text,
    p_source_language,
    p_target_language
  )
  RETURNING id INTO v_translation_id;
  
  RETURN v_translation_id;
END;
$$;

-----------------------------------------------
-- VECTOR EMBEDDINGS
-----------------------------------------------

-- Ensure pgvector extension is enabled
CREATE EXTENSION IF NOT EXISTS "vector";

-- Create vector table in vectors schema
CREATE TABLE IF NOT EXISTS vectors.transcription_embeddings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id UUID NOT NULL REFERENCES auth.users ON DELETE CASCADE,
  job_id UUID NOT NULL REFERENCES user_data.audio_jobs ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  content TEXT NOT NULL,
  embedding VECTOR(384),  -- 384 dimensions for the all-MiniLM-L6-v2 model
  metadata JSONB DEFAULT '{}'::jsonb,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT now()
);

-- Create an index for similarity search
CREATE INDEX IF NOT EXISTS transcription_embeddings_embedding_idx 
  ON vectors.transcription_embeddings 
  USING ivfflat (embedding vector_cosine_ops) 
  WITH (lists = 100);

-- Enable Row Level Security
ALTER TABLE vectors.transcription_embeddings ENABLE ROW LEVEL SECURITY;

-- Vector embeddings access policies
-- Drop existing policies first to avoid conflicts
DROP POLICY IF EXISTS "Users can view own embeddings" ON vectors.transcription_embeddings;
CREATE POLICY "Users can view own embeddings"
  ON vectors.transcription_embeddings
  FOR SELECT
  USING (auth.uid() = user_id);

DROP POLICY IF EXISTS "Users can insert own embeddings" ON vectors.transcription_embeddings;
CREATE POLICY "Users can insert own embeddings"
  ON vectors.transcription_embeddings
  FOR INSERT
  WITH CHECK (auth.uid() = user_id);

-- Create a function for similarity search
CREATE OR REPLACE FUNCTION vectors.search_transcriptions(
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
    AND 1 - (te.embedding <=> p_query_embedding) > p_match_threshold
  ORDER BY
    te.embedding <=> p_query_embedding
  LIMIT p_match_count;
  
  RETURN;
END;
$$;

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

-----------------------------------------------
-- VECTOR STORAGE
-----------------------------------------------

-- Vector Functions
-- Fixed: Parameters arranged for consistency, all defaults removed
CREATE OR REPLACE FUNCTION vectors.store_transcription_embedding(
  p_user_id UUID,
  p_transcription_id UUID,
  p_chunk_index INTEGER,
  p_content TEXT,
  p_start_time FLOAT,
  p_end_time FLOAT,
  p_metadata JSONB,
  p_embedding vector(1536)
)
RETURNS UUID
SECURITY DEFINER
LANGUAGE plpgsql
AS $$
DECLARE
  v_embedding_id UUID;
BEGIN
  -- Insert the embedding
  INSERT INTO vectors.transcription_embeddings (
    user_id,
    transcription_id,
    chunk_index,
    content,
    embedding,
    metadata
  )
  VALUES (
    p_user_id,
    p_transcription_id,
    p_chunk_index,
    p_content,
    p_embedding,
    jsonb_build_object(
      'start_time', p_start_time,
      'end_time', p_end_time
    ) || p_metadata
  )
  RETURNING id INTO v_embedding_id;
  
  RETURN v_embedding_id;
END;
$$;

-- Fixed: Keep defaults consistent and all at the end
CREATE OR REPLACE FUNCTION vectors.match_transcriptions(
  query_embedding vector(1536),
  match_count INT,
  similarity_threshold FLOAT,
  filter JSONB
)
RETURNS TABLE (
  id UUID,
  transcription_id UUID,
  content TEXT,
  metadata JSONB,
  similarity FLOAT
)
SECURITY DEFINER
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT
    te.id,
    te.transcription_id,
    te.content,
    te.metadata,
    1 - (te.embedding <=> query_embedding) AS similarity
  FROM
    vectors.transcription_embeddings te
  WHERE
    te.user_id = auth.uid()
    AND (filter->'transcription_id' IS NULL OR te.transcription_id::text = filter->>'transcription_id')
    AND 1 - (te.embedding <=> query_embedding) > similarity_threshold
  ORDER BY
    te.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
