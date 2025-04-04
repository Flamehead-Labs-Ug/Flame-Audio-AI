-- Fix Schema for Document Names and Vector Store References
-- This script addresses issues with vector store retrieval by ensuring proper
-- relationships between documents, audio jobs, and embedding records

-- Run as a transaction to ensure all changes apply together or none at all
BEGIN;

-- Create sanitize_collection_name function in SQL to match the Python implementation
CREATE OR REPLACE FUNCTION user_data.sanitize_collection_name(name TEXT)
RETURNS TEXT AS $$
BEGIN
    IF name IS NULL THEN
        RETURN '';
    END IF;
    
    -- Replace spaces with underscores and remove invalid characters
    -- Keep only alphanumeric, underscore, and hyphen characters
    RETURN SUBSTRING(
             REGEXP_REPLACE(
                 LOWER(name),
                 '[^\w\-_]', -- Keep only word chars, hyphens and underscores
                 '',
                 'g'
             ),
             1, 200); -- Limit to 200 chars
END;
$$ LANGUAGE plpgsql;

-- 1. Update audio_jobs table to properly track vector store collections
ALTER TABLE user_data.audio_jobs 
    ADD COLUMN IF NOT EXISTS qdrant_collection_name TEXT,
    ADD COLUMN IF NOT EXISTS embedding_count INTEGER DEFAULT 0;

-- Create indexes for improved query performance
CREATE INDEX IF NOT EXISTS idx_audio_jobs_qdrant_collection 
    ON user_data.audio_jobs (qdrant_collection_name);
    
CREATE INDEX IF NOT EXISTS idx_audio_jobs_agent_id
    ON user_data.audio_jobs (agent_id);

-- 2. Fix the transcriptions table to ensure proper linking
ALTER TABLE user_data.transcriptions
    ADD COLUMN IF NOT EXISTS embedding_id TEXT,
    ADD COLUMN IF NOT EXISTS embedding_model TEXT DEFAULT 'all-MiniLM-L6-v2';

-- Add index for faster embedding lookups
CREATE INDEX IF NOT EXISTS idx_transcription_embedding
    ON user_data.transcriptions (embedding_id);

-- 3. Create a tracking table for embedding records if it doesn't exist
CREATE TABLE IF NOT EXISTS user_data.embedding_tracking (
    id SERIAL PRIMARY KEY,
    job_id UUID REFERENCES user_data.audio_jobs(id) ON DELETE CASCADE,
    document_name TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding_id TEXT NOT NULL,
    qdrant_collection TEXT NOT NULL,
    content TEXT,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    user_id UUID NOT NULL,
    embedding_model TEXT DEFAULT 'all-MiniLM-L6-v2'
);

-- Add indexes for embedding tracking
CREATE INDEX IF NOT EXISTS idx_embedding_tracking_job
    ON user_data.embedding_tracking (job_id);
    
CREATE INDEX IF NOT EXISTS idx_embedding_tracking_document
    ON user_data.embedding_tracking (document_name);

CREATE INDEX IF NOT EXISTS idx_embedding_tracking_embedding_id
    ON user_data.embedding_tracking (embedding_id);

-- 4. Set up RLS policies for embedding tracking
ALTER TABLE user_data.embedding_tracking ENABLE ROW LEVEL SECURITY;

-- Create policy for users to only see their own embedding records
DROP POLICY IF EXISTS embedding_tracking_user_isolation ON user_data.embedding_tracking;
CREATE POLICY embedding_tracking_user_isolation
    ON user_data.embedding_tracking
    FOR ALL
    TO authenticated
    USING (user_id = auth.uid());

-- 5. Ensure unique constraint on audio_jobs for user_id and file_name
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 
        FROM pg_constraint 
        WHERE conname = 'audio_jobs_user_id_file_name_key'
    ) THEN
        ALTER TABLE user_data.audio_jobs
            ADD CONSTRAINT audio_jobs_user_id_file_name_key UNIQUE (user_id, file_name);
    END IF;
END $$;

-- 6. Create function to sync embedding counts with actual stored embeddings
CREATE OR REPLACE FUNCTION user_data.update_embedding_count()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE user_data.audio_jobs
    SET embedding_count = (
        SELECT COUNT(*) 
        FROM user_data.embedding_tracking 
        WHERE job_id = NEW.job_id
    )
    WHERE id = NEW.job_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger for automatic embedding count updates
DROP TRIGGER IF EXISTS update_embedding_count_trigger ON user_data.embedding_tracking;
CREATE TRIGGER update_embedding_count_trigger
AFTER INSERT OR DELETE ON user_data.embedding_tracking
FOR EACH ROW
EXECUTE FUNCTION user_data.update_embedding_count();

-- 7. Create function to backfill missing embedding IDs in transcriptions
CREATE OR REPLACE FUNCTION user_data.backfill_embedding_ids()
RETURNS void AS $$
BEGIN
    -- Update transcriptions with embedding IDs from embedding_tracking
    UPDATE user_data.transcriptions t
    SET embedding_id = e.embedding_id,
        embedding_model = e.embedding_model
    FROM user_data.embedding_tracking e
    WHERE t.job_id = e.job_id
    AND t.chunk_index = e.chunk_index
    AND (t.embedding_id IS NULL OR t.embedding_id = '');
END;
$$ LANGUAGE plpgsql;

-- 8. Backfill any existing embedding collections into audio_jobs
UPDATE user_data.audio_jobs
SET qdrant_collection_name = user_data.sanitize_collection_name(file_name)
WHERE qdrant_collection_name IS NULL;

-- 9. Run backfill function
SELECT user_data.backfill_embedding_ids();

-- Commit all changes
COMMIT;

-- Note: After running this script, you may need to restart the application
-- to ensure the vector store properly reinitializes with the updated schema
