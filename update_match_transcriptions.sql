-- Update match_transcriptions function to use explicit columns for filtering instead of trying to extract from metadata JSON
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
  IF NOT FOUND THEN
    RETURN QUERY EXECUTE
    format('
      SELECT 
        id::text, 
        content,
        metadata,
        0.9::float8 AS similarity 
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
