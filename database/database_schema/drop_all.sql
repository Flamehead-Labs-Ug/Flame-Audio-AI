-- Emergency drop script: Completely clean the database before rebuilding

-- First disable RLS to avoid permission errors (only if tables exist)
DO $$ 
BEGIN
  -- Check if tables exist before disabling RLS
  IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'public' AND tablename = 'profiles') THEN
    EXECUTE 'ALTER TABLE public.profiles DISABLE ROW LEVEL SECURITY';
  END IF;
  
  IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'user_data' AND tablename = 'sessions') THEN
    EXECUTE 'ALTER TABLE user_data.sessions DISABLE ROW LEVEL SECURITY';
  END IF;
  
  IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'user_data' AND tablename = 'audio_jobs') THEN
    EXECUTE 'ALTER TABLE user_data.audio_jobs DISABLE ROW LEVEL SECURITY';
  END IF;
  
  IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'user_data' AND tablename = 'transcriptions') THEN
    EXECUTE 'ALTER TABLE user_data.transcriptions DISABLE ROW LEVEL SECURITY';
  END IF;
  
  IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'user_data' AND tablename = 'translations') THEN
    EXECUTE 'ALTER TABLE user_data.translations DISABLE ROW LEVEL SECURITY';
  END IF;
  
  IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'vectors' AND tablename = 'transcription_embeddings') THEN
    EXECUTE 'ALTER TABLE vectors.transcription_embeddings DISABLE ROW LEVEL SECURITY';
  END IF;
  
  -- Add the newly discovered table
  IF EXISTS (SELECT FROM pg_tables WHERE schemaname = 'user_data' AND tablename = 'transcription_chunks') THEN
    EXECUTE 'ALTER TABLE user_data.transcription_chunks DISABLE ROW LEVEL SECURITY';
  END IF;
  
  -- Drop all triggers (only if they exist)
  IF EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'on_auth_user_created') THEN
    EXECUTE 'DROP TRIGGER on_auth_user_created ON auth.users';
  END IF;
  
  IF EXISTS (SELECT 1 FROM pg_trigger WHERE tgname = 'profiles_updated_at') THEN
    EXECUTE 'DROP TRIGGER profiles_updated_at ON public.profiles';
  END IF;
END $$;

-- Drop all tables with CASCADE
DROP TABLE IF EXISTS vectors.transcription_embeddings CASCADE;
DROP TABLE IF EXISTS user_data.translations CASCADE;
DROP TABLE IF EXISTS user_data.transcriptions CASCADE;
DROP TABLE IF EXISTS user_data.transcription_chunks CASCADE;
DROP TABLE IF EXISTS user_data.audio_jobs CASCADE;
DROP TABLE IF EXISTS user_data.sessions CASCADE;
DROP TABLE IF EXISTS public.profiles CASCADE;

-- Use a more generic approach to drop ALL functions in these schemas
DO $$ 
DECLARE
  r RECORD;
BEGIN
  -- Try to drop schemas with all objects inside
  BEGIN
    EXECUTE 'DROP SCHEMA IF EXISTS vectors CASCADE';
    EXCEPTION WHEN OTHERS THEN
      -- Ignore errors
  END;
  
  BEGIN
    EXECUTE 'DROP SCHEMA IF EXISTS user_data CASCADE';
    EXCEPTION WHEN OTHERS THEN
      -- Ignore errors
  END;
  
  -- Only if the above fails, try to drop functions one by one
  -- Drop all functions in vectors schema if schema exists
  IF EXISTS (SELECT FROM pg_namespace WHERE nspname = 'vectors') THEN
    FOR r IN SELECT proname, oidvectortypes(proargtypes) as argTypes 
        FROM pg_proc 
        WHERE pronamespace = 'vectors'::regnamespace
    LOOP
      EXECUTE 'DROP FUNCTION IF EXISTS vectors.' || r.proname || '(' || r.argTypes || ') CASCADE;';
    END LOOP;
  END IF;
  
  -- Drop all functions in user_data schema if schema exists
  IF EXISTS (SELECT FROM pg_namespace WHERE nspname = 'user_data') THEN
    FOR r IN SELECT proname, oidvectortypes(proargtypes) as argTypes 
        FROM pg_proc 
        WHERE pronamespace = 'user_data'::regnamespace
    LOOP
      EXECUTE 'DROP FUNCTION IF EXISTS user_data.' || r.proname || '(' || r.argTypes || ') CASCADE;';
    END LOOP;
  END IF;
  
  -- Drop select public functions
  EXECUTE 'DROP FUNCTION IF EXISTS public.handle_updated_at() CASCADE';
  EXECUTE 'DROP FUNCTION IF EXISTS public.handle_new_user() CASCADE';
END $$;

-- Recreate empty schemas
CREATE SCHEMA IF NOT EXISTS user_data;
CREATE SCHEMA IF NOT EXISTS vectors;
