import streamlit as st

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Flame Audio AI - Home",
    page_icon="logos/flame logo.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import sys
import streamlit_antd_components as sac
from dotenv import load_dotenv

from authentication.auth_forms import auth_forms, handle_auth_callback, init_auth_session

# Load environment variables
load_dotenv()

# Check if authentication is enabled
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "true").lower() == "true"

# Initialize authentication session and handle callback
init_auth_session()
handle_auth_callback()

# Sidebar configuration
# Add logo at the top of the sidebar
st.sidebar.image("logos/flame logo.jpg", width=250)

# Title after the logo
st.sidebar.title("Flame Audio")

# Add Ant Design menu in sidebar
with st.sidebar:
    menu_selection = sac.menu([
        sac.MenuItem('Home', icon='house-fill', href='/'),
        sac.MenuItem('Playground', icon='mic-fill', href='/flameaudio'),
        sac.MenuItem('Documents', icon='file-text-fill', href='/documents'),
        sac.MenuItem('Chat', icon='chat-fill', href='/chat'),
        sac.MenuItem(type='divider'),
        sac.MenuItem('Connect', type='group', children=[
            sac.MenuItem('About FlameheadLabs', icon='info-circle', href='http://flameheadlabs.tech/'),
            sac.MenuItem('Github', icon='github', href='https://github.com/Flamehead-Labs-Ug/flame-audio'),
            sac.MenuItem('Twitter', icon='twitter', href='https://x.com/flameheadlabsug'),
            sac.MenuItem('LinkedIn', icon='linkedin', href='https://www.linkedin.com/in/flamehead-labs-919910285'),
            sac.MenuItem('Email', icon='envelope-fill', href='mailto:Flameheadlabs256@gmail.com'),
        ]),
    ], format_func='title', open_all=True, key="home_page_menu")

    # Handle menu navigation
    if menu_selection == 'documents':
        st.switch_page('pages/documents.py')
    elif menu_selection == 'save_document':
        st.switch_page('pages/02_Save_Document.py')
    elif menu_selection == 'chat':
        st.switch_page('pages/chat.py')

# Show authentication forms if not authenticated
if AUTH_ENABLED and not st.session_state.get("authenticated", False):
    with st.sidebar:
        auth_forms()

# User Profile Container (only show when authenticated)
if not AUTH_ENABLED or st.session_state.get("authenticated", False):
    st.sidebar.markdown("## User Profile")
    user_profile_container = st.sidebar.container(border=True)
    with user_profile_container:
        if "user" in st.session_state:
            email = st.session_state['user'].get('email', '')
            st.markdown(f"**Signed in as:**")
            st.info(email)
            if st.button("Sign Out", key="sign_out_btn", use_container_width=True):
                # Use the proper logout function from auth_forms.py
                from authentication.auth_forms import logout
                logout()

# Sign out is now handled by the logout() function from auth_forms.py

# Main content area - Landing Page
st.title("Welcome to Flame Audio AI")

# Hero section
hero_col1, hero_col2 = st.columns([3, 2])

with hero_col1:
    st.markdown("""
    ## Transform Speech to Text with Advanced AI

    Flame Audio AI is a powerful speech transcription and translation application built by FlameheadLabs.
    Convert spoken language to text through recorded audio or uploaded files with support for multiple languages
    and advanced processing options.

    ### Key Features:

    - **High-Quality Transcription**: Convert speech to text with high accuracy
    - **Translation**: Translate audio from any language to English
    - **Document Management**: Save, organize, and search your transcriptions
    - **AI Chat**: Interact with your transcribed content through natural language
    - **Agent System**: Create specialized agents for different types of content
    """)

    # Call-to-action buttons
    cta_col1, cta_col2, cta_col3 = st.columns(3)
    with cta_col1:
        if st.button("Start Transcribing", type="primary", use_container_width=True):
            st.switch_page("pages/flameaudio.py")
    with cta_col2:
        if st.button("View Documents", use_container_width=True):
            st.switch_page("pages/documents.py")
    with cta_col3:
        if st.button("Chat with AI", use_container_width=True):
            st.switch_page("pages/chat.py")

with hero_col2:
    st.image("logos/flame logo.jpg", width=300)

# Features section
st.markdown("---")
st.header("How It Works")

# Feature cards
feat_col1, feat_col2, feat_col3 = st.columns(3)

with feat_col1:
    st.container(border=True, height=300).markdown("""
    ### 1. Record or Upload

    Record audio directly in your browser or upload audio files in various formats.

    Supported formats include MP3, WAV, M4A, and more.
    """)

with feat_col2:
    st.container(border=True, height=300).markdown("""
    ### 2. Transcribe or Translate

    Process your audio with state-of-the-art AI models.

    Choose between transcription in the original language or translation to English.
    """)

with feat_col3:
    st.container(border=True, height=300).markdown("""
    ### 3. Save and Interact

    Save your transcriptions to the database and interact with them through:

    - Document search
    - AI-powered chat
    - Export options
    """)

# Testimonials section
st.markdown("---")
st.header("What Our Users Say")

test_col1, test_col2 = st.columns(2)

with test_col1:
    st.container(border=True).markdown("""
    > "Flame Audio has revolutionized how we handle meeting transcriptions. The accuracy is impressive and the AI chat feature helps us quickly extract insights."

    **- Sarah J., Product Manager**
    """)

with test_col2:
    st.container(border=True).markdown("""
    > "As a researcher, I need reliable transcription for interviews. Flame Audio not only provides accurate transcripts but also helps me analyze them with its AI capabilities."

    **- Dr. Michael T., Academic Researcher**
    """)

# Getting Started section
st.markdown("---")
st.header("Getting Started")

st.markdown("""
1. **Sign Up**: Create an account to save your transcriptions
2. **Set Up**: Configure your API key in the sidebar
3. **Record or Upload**: Start with your first audio file
4. **Process**: Transcribe or translate your audio
5. **Explore**: Use the AI chat to interact with your transcriptions
""")

# Call to action
st.markdown("---")
st.subheader("Ready to transform your audio content?")

final_cta_col1, final_cta_col2, final_cta_col3 = st.columns([2, 1, 2])
with final_cta_col2:
    if st.button("Get Started Now", type="primary", use_container_width=True):
        st.switch_page("pages/flameaudio.py")

# Define footer HTML
footer = """
<footer style="margin-top: 5rem; padding: 2.5rem 0; border-top: 1px solid rgba(0,0,0,0.05); width: 100%;">
    <div style="max-width: 1200px; margin: 0 auto; padding: 0 1.5rem; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
        <div style="display: flex; align-items: center;">
            <div style="font-size: 0.9rem; color: #6B7280; margin-right: 1rem;">
                2025 FlameheadLabs
            </div>
        </div>
        <div style="font-size: 0.85rem; color: #9CA3AF;">
            <a href="privacy.html" target="_blank" style="color: #6B7280; margin-right: 1rem; text-decoration: none;">Privacy Policy</a>
            <a href="terms.html" target="_blank" style="color: #6B7280; text-decoration: none;">Terms of Service</a>
        </div>
    </div>
</footer>
"""

st.markdown(footer, unsafe_allow_html=True)
