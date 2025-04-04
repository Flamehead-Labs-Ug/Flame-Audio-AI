-- Add Qdrant collection tracking to the database schema

-- Add qdrant_collection_name column to the audio_jobs table
ALTER TABLE user_data.audio_jobs 
    ADD COLUMN IF NOT EXISTS qdrant_collection_name TEXT;

-- Create an index on the qdrant_collection_name column for faster lookups
CREATE INDEX IF NOT EXISTS idx_audio_jobs_qdrant_collection 
    ON user_data.audio_jobs (qdrant_collection_name);

-- Add comment to document the column's purpose
COMMENT ON COLUMN user_data.audio_jobs.qdrant_collection_name IS 'Name of the corresponding Qdrant collection where embeddings are stored';

-- Update RLS policies to ensure security
DROP POLICY IF EXISTS "Users can update their own audio jobs' qdrant collection" ON user_data.audio_jobs;
CREATE POLICY "Users can update their own audio jobs' qdrant collection"
    ON user_data.audio_jobs
    FOR UPDATE 
    TO authenticated
    USING (user_id = auth.uid())
    WITH CHECK (user_id = auth.uid());

-- Populate the column for existing records with the current naming convention
UPDATE user_data.audio_jobs
    SET qdrant_collection_name = 'job_' || id
    WHERE qdrant_collection_name IS NULL;
