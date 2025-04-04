-- Add vector_store_settings table to user_data schema
CREATE TABLE IF NOT EXISTS user_data.vector_store_settings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    settings JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT unique_user_settings UNIQUE (user_id)
);

-- Add RLS policies
ALTER TABLE user_data.vector_store_settings ENABLE ROW LEVEL SECURITY;

-- Policy for selecting own settings
CREATE POLICY select_own_vector_settings ON user_data.vector_store_settings 
    FOR SELECT USING (auth.uid() = user_id);

-- Policy for inserting own settings
CREATE POLICY insert_own_vector_settings ON user_data.vector_store_settings 
    FOR INSERT WITH CHECK (auth.uid() = user_id);

-- Policy for updating own settings
CREATE POLICY update_own_vector_settings ON user_data.vector_store_settings 
    FOR UPDATE USING (auth.uid() = user_id);

-- Policy for deleting own settings
CREATE POLICY delete_own_vector_settings ON user_data.vector_store_settings 
    FOR DELETE USING (auth.uid() = user_id);

-- Function to apply new embedding settings (will be used in the future)
CREATE OR REPLACE FUNCTION user_data.apply_embedding_settings()
RETURNS TRIGGER AS $$
BEGIN
    -- In the future, this function could trigger re-embedding of documents
    -- or update vector store configuration based on new settings
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to apply settings when they change
CREATE TRIGGER apply_vector_settings_trigger
    AFTER UPDATE ON user_data.vector_store_settings
    FOR EACH ROW
    EXECUTE FUNCTION user_data.apply_embedding_settings();
