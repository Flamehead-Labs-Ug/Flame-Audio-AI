-- SQL function for LangChain SupabaseVectorStore

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
  similarity float8 -- IMPORTANT: Must be float8 (double precision) for LangChain
) 
LANGUAGE plpgsql
AS $$
DECLARE 
  filter_clause text := '';
  query_text text;
BEGIN
  -- Convert filter to a WHERE clause
  IF filter->>'user_id' IS NOT NULL THEN
    filter_clause := filter_clause || ' AND metadata->>''user_id'' = ' || quote_literal(filter->>'user_id');
  END IF;
  
  IF filter->>'job_id' IS NOT NULL THEN
    filter_clause := filter_clause || ' AND metadata->>''job_id'' = ' || quote_literal(filter->>'job_id');
  END IF;
  
  IF filter->>'agent_id' IS NOT NULL THEN
    filter_clause := filter_clause || ' AND metadata->>''agent_id'' = ' || quote_literal(filter->>'agent_id');
  END IF;

  -- First try standard vector search
  RETURN QUERY EXECUTE
  format('
    SELECT 
      id::text, 
      content,
      metadata,
      (1 - (embedding <=> $1))::float8 AS similarity -- Cast to float8 explicitly
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
        (1 - (embedding <=> $1))::float8 AS similarity -- Cast to float8 explicitly
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
  IF NOT FOUND THEN
    RETURN QUERY EXECUTE
    format('
      SELECT 
        id::text, 
        content,
        metadata,
        0.8::float8 AS similarity -- Cast to float8 explicitly
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
