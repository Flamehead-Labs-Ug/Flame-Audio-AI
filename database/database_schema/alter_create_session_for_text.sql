-- This migration alters the user_data.create_session function to accept TEXT for p_user_id,
-- to match the argument type used by your backend. Run this in Supabase SQL editor or psql.

-- Drop the old function if it exists (UUID signature)
DROP FUNCTION IF EXISTS user_data.create_session(UUID, TEXT);

-- Recreate the function with TEXT for p_user_id
CREATE OR REPLACE FUNCTION user_data.create_session(
    p_user_id TEXT,
    p_expires_interval TEXT
)
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

  -- Insert into sessions table, casting p_user_id to UUID
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
    p_user_id::UUID,
    v_token,
    now() + p_expires_interval::interval,
    current_setting('request.headers', true)::json->>'user-agent',
    current_setting('request.headers', true)::json->>'x-forwarded-for'
  );

  RETURN v_session_id;
END;
$$;

-- Optionally, grant execute permission (if needed)
-- GRANT EXECUTE ON FUNCTION user_data.create_session(TEXT, TEXT) TO your_role;
