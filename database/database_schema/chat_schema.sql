-- Chat history and persistent memory schema

-- 1. Create chat sessions table
CREATE TABLE IF NOT EXISTS user_data.chat_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users (id) ON DELETE CASCADE,
    agent_id UUID REFERENCES user_data.user_agents (id) ON DELETE SET NULL,
    document_id UUID REFERENCES user_data.audio_jobs (id) ON DELETE SET NULL,
    title TEXT NOT NULL DEFAULT 'New Chat',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb,
    chat_parameters JSONB DEFAULT '{}'::jsonb
);

-- Apply RLS policies to chat_sessions
ALTER TABLE user_data.chat_sessions ENABLE ROW LEVEL SECURITY;

-- Policies for chat_sessions
CREATE POLICY "Users can view their own chat sessions" 
    ON user_data.chat_sessions FOR SELECT 
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own chat sessions" 
    ON user_data.chat_sessions FOR INSERT 
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own chat sessions" 
    ON user_data.chat_sessions FOR UPDATE 
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own chat sessions" 
    ON user_data.chat_sessions FOR DELETE 
    USING (auth.uid() = user_id);

-- 2. Create chat messages table
CREATE TABLE IF NOT EXISTS user_data.chat_messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID NOT NULL REFERENCES user_data.chat_sessions (id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'function', 'tool')),
    content TEXT NOT NULL,
    message_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(), -- Renamed column
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding VECTOR(384)
);

-- Apply RLS policies to chat_messages
ALTER TABLE user_data.chat_messages ENABLE ROW LEVEL SECURITY;

-- Policies for chat_messages
CREATE POLICY "Users can view messages from their own chat sessions" 
    ON user_data.chat_messages FOR SELECT 
    USING (EXISTS (
        SELECT 1 FROM user_data.chat_sessions cs 
        WHERE cs.id = session_id AND cs.user_id = auth.uid()
    ));

CREATE POLICY "Users can insert messages into their own chat sessions" 
    ON user_data.chat_messages FOR INSERT 
    WITH CHECK (EXISTS (
        SELECT 1 FROM user_data.chat_sessions cs 
        WHERE cs.id = session_id AND cs.user_id = auth.uid()
    ));

CREATE POLICY "Users can update messages from their own chat sessions" 
    ON user_data.chat_messages FOR UPDATE 
    USING (EXISTS (
        SELECT 1 FROM user_data.chat_sessions cs 
        WHERE cs.id = session_id AND cs.user_id = auth.uid()
    ));

CREATE POLICY "Users can delete messages from their own chat sessions" 
    ON user_data.chat_messages FOR DELETE 
    USING (EXISTS (
        SELECT 1 FROM user_data.chat_sessions cs 
        WHERE cs.id = session_id AND cs.user_id = auth.uid()
    ));

-- 3. Create a function to retrieve chat messages by session
CREATE OR REPLACE FUNCTION user_data.get_chat_messages(
    p_session_id UUID
)
RETURNS TABLE (
    id UUID,
    role TEXT,
    content TEXT,
    message_timestamp TIMESTAMPTZ, -- Updated to match the new column name
    metadata JSONB
)
SECURITY DEFINER
LANGUAGE plpgsql
AS $$
BEGIN
    -- Check if the session belongs to the current user
    IF NOT EXISTS (SELECT 1 FROM user_data.chat_sessions cs 
                  WHERE cs.id = p_session_id AND cs.user_id = auth.uid()) THEN
        RAISE EXCEPTION 'Access denied to chat session';
    END IF;

    RETURN QUERY
    SELECT cm.id, cm.role, cm.content, cm.message_timestamp, cm.metadata -- Updated to match the new column name
    FROM user_data.chat_messages cm
    WHERE cm.session_id = p_session_id
    ORDER BY cm.message_timestamp ASC; -- Updated to match the new column name
END;
$$;

-- 4. Create a function to search through chat history with similarity
CREATE OR REPLACE FUNCTION user_data.search_chat_history(
    query_text TEXT,
    agent_id UUID DEFAULT NULL,
    similarity_threshold FLOAT DEFAULT 0.5,
    limit_results INT DEFAULT 10
)
RETURNS TABLE (
    message_id UUID,
    session_id UUID,
    role TEXT,
    content TEXT,
    message_timestamp TIMESTAMPTZ, -- Updated to match the new column name
    similarity FLOAT
)
SECURITY DEFINER
LANGUAGE plpgsql
AS $$
DECLARE
    query_embedding VECTOR(384);
BEGIN
    -- Generate embedding for the query text
    SELECT embedding INTO query_embedding FROM public.generate_embedding(query_text);
    
    RETURN QUERY
    SELECT 
        cm.id AS message_id,
        cm.session_id,
        cm.role,
        cm.content,
        cm.message_timestamp, -- Updated to match the new column name
        1 - (cm.embedding <=> query_embedding) AS similarity
    FROM user_data.chat_messages cm
    JOIN user_data.chat_sessions cs ON cm.session_id = cs.id
    WHERE cs.user_id = auth.uid()
    AND (agent_id IS NULL OR cs.agent_id = agent_id)
    AND cm.embedding IS NOT NULL
    AND 1 - (cm.embedding <=> query_embedding) > similarity_threshold
    ORDER BY similarity DESC
    LIMIT limit_results;
END;
$$;