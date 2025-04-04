import os
import streamlit as st
from supabase import create_client
import logging

logger = logging.getLogger("chat_realtime")

def initialize_realtime_chat(agent_id=None, session_id=None):
    """Initialize Realtime subscription for chat functionality"""
    if not st.session_state.get("authenticated", False):
        logger.warning("Cannot initialize Realtime: User not authenticated")
        return False
        
    try:
        # Create Supabase client with user's auth token
        supabase_client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_ANON_KEY"),
            headers={
                "Authorization": f"Bearer {st.session_state.get('_auth_token_')}"
            }
        )
        
        # Define handler for new messages
        def handle_new_message(payload):
            # Only process messages for the current chat session
            if session_id and payload['new'].get('session_id') == session_id:
                # Add the new message to session state messages list
                if 'chat_messages' not in st.session_state:
                    st.session_state.chat_messages = []
                
                st.session_state.chat_messages.append(payload['new'])
                # Trigger rerun to update the UI
                st.experimental_rerun()
        
        # Set up filter for messages table
        # Filter based on session_id if provided
        filters = {}
        if session_id:
            filters["session_id"] = session_id
            
        # Subscribe to changes based on filters
        chat_subscription = supabase_client.table('user_data.chat_messages')
        
        # Apply filters if any exist
        if filters:
            for key, value in filters.items():
                chat_subscription = chat_subscription.eq(key, value)
                
        # Finalize subscription with handler
        st.session_state.realtime_subscription = chat_subscription.on('INSERT', handle_new_message).subscribe()
        
        logger.info(f"Realtime subscription initialized for chat session: {session_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing Realtime: {e}")
        return False
        
def cleanup_realtime():
    """Remove Realtime subscription"""
    if st.session_state.get("realtime_subscription"):
        try:
            st.session_state.realtime_subscription.unsubscribe()
            del st.session_state.realtime_subscription
            logger.info("Realtime subscription removed")
        except Exception as e:
            logger.error(f"Error removing Realtime subscription: {e}")
