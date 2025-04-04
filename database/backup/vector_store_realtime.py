import json
import os
import logging
from typing import Optional, Callable, Dict, Any

import streamlit as st
from supabase import create_client

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store callback handlers
vector_settings_callbacks = {}

def initialize_vector_store_realtime(user_id: str) -> bool:
    """
    Initialize Supabase Realtime subscription for vector store settings changes
    
    Args:
        user_id: The user ID to subscribe to settings for
        
    Returns:
        bool: True if subscription was successful, False otherwise
    """
    try:
        logger.info(f"Initializing vector store settings realtime for user {user_id}")
        
        # Create Supabase client with user's auth token
        supabase_client = create_client(
            os.getenv("SUPABASE_URL"),
            os.getenv("SUPABASE_ANON_KEY"),
            headers={
                "Authorization": f"Bearer {st.session_state.get('_auth_token_')}"
            }
        )
        
        # Channel name based on user ID
        channel_name = f"vector-settings-{user_id}"
        
        # Subscribe to vector store settings table for this user
        channel = supabase_client.channel(channel_name)
        channel.on(
            'postgres_changes',
            event='UPDATE',
            schema='user_data',
            table='vector_store_settings',
            filter=f"user_id=eq.{user_id}",
            callback=lambda payload: handle_settings_update(payload)
        ).on(
            'postgres_changes',
            event='INSERT',
            schema='user_data',
            table='vector_store_settings',
            filter=f"user_id=eq.{user_id}",
            callback=lambda payload: handle_settings_update(payload)
        ).subscribe()
        
        # Store the channel in session state for later reference
        if "realtime_channels" not in st.session_state:
            st.session_state.realtime_channels = {}
            
        st.session_state.realtime_channels[channel_name] = channel
        logger.info(f"Successfully subscribed to vector store settings updates")
        
        return True
    
    except Exception as e:
        logger.error(f"Error initializing vector store settings realtime: {e}")
        return False

def handle_settings_update(payload: Dict[str, Any]):
    """
    Handle realtime updates to vector store settings
    
    Args:
        payload: The payload from Supabase Realtime
    """
    try:
        logger.info(f"Received vector store settings update: {json.dumps(payload)}")
        
        # Extract settings from payload
        new_record = payload.get("new", {})
        settings = new_record.get("settings", {})
        
        # Update session state with new settings
        if "vector_store_settings" not in st.session_state:
            st.session_state.vector_store_settings = {}
            
        st.session_state.vector_store_settings["settings"] = settings
        
        # Trigger any registered callbacks
        user_id = new_record.get("user_id")
        if user_id in vector_settings_callbacks:
            for callback in vector_settings_callbacks[user_id]:
                callback(settings)
        
        logger.info("Vector store settings updated in session state")
    
    except Exception as e:
        logger.error(f"Error handling vector store settings update: {e}")

def register_settings_callback(user_id: str, callback: Callable[[Dict[str, Any]], None]) -> None:
    """
    Register a callback for when vector store settings are updated
    
    Args:
        user_id: The user ID to register the callback for
        callback: Callback function that takes settings dict as parameter
    """
    if user_id not in vector_settings_callbacks:
        vector_settings_callbacks[user_id] = []
        
    vector_settings_callbacks[user_id].append(callback)
    logger.info(f"Registered vector store settings callback for user {user_id}")

def unregister_settings_callback(user_id: str, callback: Callable[[Dict[str, Any]], None]) -> None:
    """
    Unregister a previously registered callback
    
    Args:
        user_id: The user ID the callback was registered for
        callback: The callback function to unregister
    """
    if user_id in vector_settings_callbacks and callback in vector_settings_callbacks[user_id]:
        vector_settings_callbacks[user_id].remove(callback)
        logger.info(f"Unregistered vector store settings callback for user {user_id}")

def cleanup_realtime_subscriptions():
    """
    Cleanup all realtime subscriptions
    """
    if "realtime_channels" in st.session_state:
        for channel_name, channel in st.session_state.realtime_channels.items():
            if "vector-settings" in channel_name:
                try:
                    channel.unsubscribe()
                    logger.info(f"Unsubscribed from channel {channel_name}")
                except Exception as e:
                    logger.error(f"Error unsubscribing from channel {channel_name}: {e}")
