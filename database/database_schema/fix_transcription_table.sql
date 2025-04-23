-- Drop existing table if it exists
DROP TABLE IF EXISTS vectors.transcription_embeddings;

-- Create the transcription_embeddings table with explicit columns
CREATE TABLE vectors.transcription_embeddings (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    job_id TEXT,
    agent_id TEXT,
    content TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
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

-- Create indexes for common query patterns
CREATE INDEX IF NOT EXISTS transcription_embeddings_user_id_idx ON vectors.transcription_embeddings(user_id);
CREATE INDEX IF NOT EXISTS transcription_embeddings_job_id_idx ON vectors.transcription_embeddings(job_id);
CREATE INDEX IF NOT EXISTS transcription_embeddings_agent_id_idx ON vectors.transcription_embeddings(agent_id);

-- Grant access to the table
GRANT ALL ON vectors.transcription_embeddings TO service_role;
GRANT ALL ON vectors.transcription_embeddings TO anon;
GRANT ALL ON vectors.transcription_embeddings TO authenticated;
