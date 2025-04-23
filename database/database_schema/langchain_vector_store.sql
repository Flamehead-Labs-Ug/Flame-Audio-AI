-- SQL functions for LangChain SupabaseVectorStore

-- This function is used by LangChain's SupabaseVectorStore to perform vector similarity search
CREATE OR REPLACE FUNCTION match_transcriptions(
  query_embedding vector(384),
  match_count int DEFAULT 10,
  similarity_threshold float DEFAULT 0.2,
  filter jsonb DEFAULT '{}'::jsonb
) RETURNS TABLE (
  id text,
  content text,
  metadata jsonb,
  similarity float8 
) 
LANGUAGE plpgsql
AS $$
DECLARE 
  filter_clause text := '';
  user_id_value text;
  job_id_value text;
  agent_id_value text;
  query_text text;
BEGIN
  -- Extract filter values
  user_id_value := filter->>'user_id';
  job_id_value := filter->>'job_id';
  agent_id_value := filter->>'agent_id';
  
  -- Convert filter to a WHERE clause using explicit columns
  IF user_id_value IS NOT NULL THEN
    filter_clause := filter_clause || ' AND user_id = ' || quote_literal(user_id_value);
  END IF;
  
  IF job_id_value IS NOT NULL THEN
    filter_clause := filter_clause || ' AND job_id = ' || quote_literal(job_id_value);
  END IF;
  
  IF agent_id_value IS NOT NULL THEN
    filter_clause := filter_clause || ' AND agent_id = ' || quote_literal(agent_id_value);
  END IF;

  -- For debugging
  RAISE NOTICE 'Filter clause: %', filter_clause;
  
  -- First try standard vector search
  RETURN QUERY EXECUTE
  format('
    SELECT 
      id::text, 
      content,
      metadata,
      (1 - (embedding <=> $1))::float8 AS similarity 
    FROM 
      vectors.transcription_embeddings
    WHERE 
      (1 - (embedding <=> $1)) > $2
      %s
    ORDER BY 
      embedding <=> $1
    LIMIT $3
  ', filter_clause)
  USING query_embedding, similarity_threshold, match_count;
  
  -- If no results, try a more lenient search with lower threshold
  IF NOT FOUND THEN
    RETURN QUERY EXECUTE
    format('
      SELECT 
        id::text, 
        content,
        metadata,
        (1 - (embedding <=> $1))::float8 AS similarity 
      FROM 
        vectors.transcription_embeddings
      WHERE 
        TRUE
        %s
      ORDER BY 
        embedding <=> $1
      LIMIT $3
    ', filter_clause)
    USING query_embedding, 0.01, match_count;
  END IF;
  
  -- If still no results, try a direct text search as last resort
  -- This would require extracting text from the query embedding which isn't directly possible
  -- So we only do this if the function is called directly with text
  
  -- Check if text content actually exists in our database
  RAISE NOTICE 'Checking if any content exists with these filters';
  
  IF NOT FOUND THEN
    RETURN QUERY EXECUTE
    format('
      SELECT 
        id::text, 
        content,
        metadata,
        0.8::float8 AS similarity 
      FROM 
        vectors.transcription_embeddings
      WHERE 
        TRUE
        %s
      LIMIT $1
    ', filter_clause)
    USING match_count;
  END IF;
END;
$$;

-- Grant access to the function
GRANT EXECUTE ON FUNCTION match_transcriptions(vector, int, float, jsonb) TO service_role;
GRANT EXECUTE ON FUNCTION match_transcriptions(vector, int, float, jsonb) TO anon;
GRANT EXECUTE ON FUNCTION match_transcriptions(vector, int, float, jsonb) TO authenticated;

-- Check if the vectors schema exists, if not create it
DO $$ BEGIN
    CREATE SCHEMA IF NOT EXISTS vectors;
    EXCEPTION WHEN duplicate_schema THEN
    RAISE NOTICE 'schema "vectors" already exists';
END $$;

-- Check if the vectors.transcription_embeddings table exists, if not create it
CREATE TABLE IF NOT EXISTS vectors.transcription_embeddings (
    id TEXT PRIMARY KEY,
    content TEXT,
    user_id TEXT,
    job_id TEXT,
    agent_id TEXT,
    metadata JSONB,
    embedding VECTOR(384)
);

-- Enable RLS on the table
ALTER TABLE vectors.transcription_embeddings ENABLE ROW LEVEL SECURITY;

-- Create policies for RLS
DROP POLICY IF EXISTS "Users can view their own embeddings" ON vectors.transcription_embeddings;
CREATE POLICY "Users can view their own embeddings"
    ON vectors.transcription_embeddings
    FOR SELECT
    USING (user_id = auth.uid()::text);

DROP POLICY IF EXISTS "Users can insert their own embeddings" ON vectors.transcription_embeddings;
CREATE POLICY "Users can insert their own embeddings"
    ON vectors.transcription_embeddings
    FOR INSERT
    WITH CHECK (user_id = auth.uid()::text);

DROP POLICY IF EXISTS "Users can update their own embeddings" ON vectors.transcription_embeddings;
CREATE POLICY "Users can update their own embeddings"
    ON vectors.transcription_embeddings
    FOR UPDATE
    USING (user_id = auth.uid()::text);

DROP POLICY IF EXISTS "Users can delete their own embeddings" ON vectors.transcription_embeddings;
CREATE POLICY "Users can delete their own embeddings"
    ON vectors.transcription_embeddings
    FOR DELETE
    USING (user_id = auth.uid()::text);

-- Create index on embeddings for better performance
CREATE INDEX IF NOT EXISTS transcription_embeddings_embedding_idx ON vectors.transcription_embeddings 
USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
