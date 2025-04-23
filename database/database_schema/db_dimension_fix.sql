-- Fix dimension mismatch in vector database functions

-- 1. Update the store_transcription_embedding function to use 384 dimensions
CREATE OR REPLACE FUNCTION vectors.store_transcription_embedding(
  p_user_id UUID,
  p_transcription_id UUID,
  p_chunk_index INTEGER,
  p_content TEXT,
  p_start_time FLOAT,
  p_end_time FLOAT,
  p_metadata JSONB,
  p_embedding vector(384)  -- Changed from 1536 to 384 dimensions
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

-- 2. Update the match_transcriptions function to use 384 dimensions
CREATE OR REPLACE FUNCTION vectors.match_transcriptions(
  query_embedding vector(384),  -- Changed from 1536 to 384 dimensions
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

-- 3. Also create the search_transcriptions function that allows explicit user_id passing
-- This provides a more flexible search option
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
    te.transcription_id AS job_id,
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
