-- Clear document data script: Remove all document content without altering table structure

-- This script deletes all document content including:
-- 1. Transcriptions and translations
-- 2. Document segments/chunks
-- 3. Vector embeddings references
-- 4. Audio job history
-- While preserving all tables, schemas, and RLS policies

-- First add a comment explaining what this script does for safety
DO $$ 
BEGIN
  RAISE NOTICE 'This script will delete ALL document content while preserving database structure.';
END $$;

-- For safety, wrap everything in a transaction
BEGIN;

-- Delete all document content with proper RLS enforcement
-- This ensures only accessible records are deleted when run as a normal user
-- When run as superuser or table owner, it will delete all records

-- Check if tables exist before attempting to delete from them
DO $$ 
BEGIN
  -- 1. Clear embeddings tracking records if the table exists
  IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'vectors' AND tablename = 'transcription_embeddings') THEN
    EXECUTE 'DELETE FROM vectors.transcription_embeddings';
    RAISE NOTICE 'Cleared vectors.transcription_embeddings';
  ELSE
    RAISE NOTICE 'Table vectors.transcription_embeddings does not exist, skipping';
  END IF;

  -- 2. Clear transcription chunks/segments if the table exists
  IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'user_data' AND tablename = 'transcription_chunks') THEN
    EXECUTE 'DELETE FROM user_data.transcription_chunks';
    RAISE NOTICE 'Cleared user_data.transcription_chunks';
  ELSE
    RAISE NOTICE 'Table user_data.transcription_chunks does not exist, skipping';
  END IF;

  -- 3. Clear translations if the table exists
  IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'user_data' AND tablename = 'translations') THEN
    EXECUTE 'DELETE FROM user_data.translations';
    RAISE NOTICE 'Cleared user_data.translations';
  ELSE
    RAISE NOTICE 'Table user_data.translations does not exist, skipping';
  END IF;

  -- 4. Clear transcriptions (main documents) if the table exists
  IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'user_data' AND tablename = 'transcriptions') THEN
    EXECUTE 'DELETE FROM user_data.transcriptions';
    RAISE NOTICE 'Cleared user_data.transcriptions';
  ELSE
    RAISE NOTICE 'Table user_data.transcriptions does not exist, skipping';
  END IF;

  -- 5. Clear audio processing jobs if the table exists
  IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'user_data' AND tablename = 'audio_jobs') THEN
    EXECUTE 'DELETE FROM user_data.audio_jobs';
    RAISE NOTICE 'Cleared user_data.audio_jobs';
  ELSE
    RAISE NOTICE 'Table user_data.audio_jobs does not exist, skipping';
  END IF;

  -- If you want to clear agent-associated documents but keep agents
  -- Uncomment this section:
  /*
  IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'user_data' AND tablename = 'agents') THEN
    EXECUTE 'UPDATE user_data.agents SET document_count = 0 WHERE document_count > 0';
    RAISE NOTICE 'Reset document counts for user_data.agents';
  ELSE
    RAISE NOTICE 'Table user_data.agents does not exist, skipping';
  END IF;
  */
END $$;

-- Reset sequence values to start from 1 again
DO $$ 
BEGIN
  -- Reset transcriptions sequence if it exists
  IF EXISTS (SELECT FROM pg_sequences WHERE schemaname = 'user_data' AND sequencename = 'transcriptions_id_seq') THEN
    EXECUTE 'ALTER SEQUENCE user_data.transcriptions_id_seq RESTART WITH 1';
    RAISE NOTICE 'Reset sequence user_data.transcriptions_id_seq';
  END IF;

  -- Reset translations sequence if it exists
  IF EXISTS (SELECT FROM pg_sequences WHERE schemaname = 'user_data' AND sequencename = 'translations_id_seq') THEN
    EXECUTE 'ALTER SEQUENCE user_data.translations_id_seq RESTART WITH 1';
    RAISE NOTICE 'Reset sequence user_data.translations_id_seq';
  END IF;

  -- Reset transcription_chunks sequence if it exists
  IF EXISTS (SELECT FROM pg_sequences WHERE schemaname = 'user_data' AND sequencename = 'transcription_chunks_id_seq') THEN
    EXECUTE 'ALTER SEQUENCE user_data.transcription_chunks_id_seq RESTART WITH 1';
    RAISE NOTICE 'Reset sequence user_data.transcription_chunks_id_seq';
  END IF;

  -- Reset audio_jobs sequence if it exists
  IF EXISTS (SELECT FROM pg_sequences WHERE schemaname = 'user_data' AND sequencename = 'audio_jobs_id_seq') THEN
    EXECUTE 'ALTER SEQUENCE user_data.audio_jobs_id_seq RESTART WITH 1';
    RAISE NOTICE 'Reset sequence user_data.audio_jobs_id_seq';
  END IF;
END $$;

-- Add statements to clear vector store counts if needed
-- This maintains database schema but resets the counters
DO $$ 
BEGIN
  RAISE NOTICE 'Document data has been cleared successfully.';
  RAISE NOTICE 'To clear Qdrant vector store completely, use the Python function vector_store.reset_collections()';
END $$;

-- Uncomment to run automatically, comment to review before committing
COMMIT;
