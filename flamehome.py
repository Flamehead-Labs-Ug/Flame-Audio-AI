import streamlit as st

# Page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Flame Audio AI - Home",
    page_icon="logos/flame logo.jpg",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
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
        sac.MenuItem('Agents', icon='person-fill', href='/agents'),
        sac.MenuItem('Documents', icon='file-text-fill', href='/documents'),
        #sac.MenuItem('Chat', icon='chat-fill', href='/chat'),
        sac.MenuItem('MCP', icon='gear-fill', href='/flame_mcp'),
        sac.MenuItem('Flame Audio Chat', icon='chat-dots-fill', href='/mcp_chat'),
        sac.MenuItem(type='divider'),
        sac.MenuItem('Connect', type='group', children=[
            sac.MenuItem('About FlameheadLabs', icon='info-circle', href='http://flameheadlabs.tech/'),
            sac.MenuItem('Github', icon='github', href='https://github.com/Flamehead-Labs-Ug/flame-audio'),
            sac.MenuItem('Twitter', icon='twitter', href='https://x.com/flameheadlabsug'),
            sac.MenuItem('LinkedIn', icon='linkedin', href='https://www.linkedin.com/in/flamehead-labs-919910285'),
            sac.MenuItem('Email', icon='envelope-fill', href='mailto:Flameheadlabs256@gmail.com'),
        ]),
    ], format_func='title', open_all=True, key="home_page_menu")

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
# === Futuristic Hero Section ===
st.markdown("""
<style>
.hero-bg {
    background: linear-gradient(90deg, #0f2027 0%, #2c5364 100%);
    border-radius: 1.5rem;
    padding: 2.5rem 2rem 2rem 2rem;
    margin-bottom: 2rem;
    color: #fff;
}
.speechwave {
    background: linear-gradient(90deg, #1CB5E0 0%, #000851 100%);
    border-radius: 1rem;
    padding: 1.5rem;
    color: #fff;
    box-shadow: 0 8px 32px 0 rgba(31,38,135,0.37);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-bg">
    <h1 style="font-size: 2.6rem; font-weight: 800; letter-spacing: -2px; margin-bottom: 0.5rem;">
        <span style="background: linear-gradient(90deg, #1CB5E0 0%, #000851 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Flame Audio AI</span>
    </h1>
    <h3 style="font-weight: 500; margin-bottom: 1.2rem;">AI-Powered Speech-to-Text & Conversational Intelligence</h3>
    <div style="display: flex; align-items: center; margin-bottom: 1.2rem;">
        <img src="https://cdn-icons-png.flaticon.com/512/4712/4712035.png" width="64" style="margin-right: 1.2rem; border-radius: 1rem; box-shadow: 0 4px 16px #1CB5E0;">
        <div style="font-size: 1.2rem;">
            Experience next-gen transcription, translation, and AI chat.<br>
            <b>Transform audio into actionable insights.</b>
        </div>
    </div>
    <div style="display: flex; gap: 1rem; margin-top: 1.5rem;">
        <a href="/flameaudio" style="background: #1CB5E0; color: #fff; padding: 0.85rem 2.2rem; border-radius: 2rem; font-size: 1.2rem; font-weight: 600; text-decoration: none; box-shadow: 0 2px 8px #1CB5E0;">🎙️ Start Transcribing</a>
        <a href="/documents" style="background: #2c5364; color: #fff; padding: 0.85rem 2.2rem; border-radius: 2rem; font-size: 1.2rem; font-weight: 600; text-decoration: none; box-shadow: 0 2px 8px #2c5364;">📄 View Documents</a>
        <a href="/mcp_chat" style="background: linear-gradient(90deg, #1CB5E0 0%, #000851 100%); color: #fff; padding: 0.85rem 2.2rem; border-radius: 2rem; font-size: 1.2rem; font-weight: 600; text-decoration: none; box-shadow: 0 2px 8px #1CB5E0;">🤖 Flame Audio Chat</a>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Highlight Flame Audio Chat Section ---
st.markdown("""
<div class="speechwave">
    <h2 style="margin-bottom: 0.5rem; font-size: 2rem; font-weight: 700;">🤖 Flame Audio Chat</h2>
    <p style="font-size: 1.1rem; margin-bottom: 1rem;">
        Our new <b>Flame Audio Chat</b> lets you interact with your transcriptions and documents using advanced conversational AI.<br>
        <b>Ask questions, summarize, extract insights, and command your audio data with natural language!</b>
    </p>
</div>
""", unsafe_allow_html=True)

features = [
    ("🎙️", "Record or Upload", "Capture audio with your mic or upload files (MP3, WAV, M4A, and more)."),
    ("📝", "Transcribe & Translate", "AI models convert speech to text and translate to English or other languages."),
    ("🤖", "Flame Audio Chat", "Chat with your documents using advanced conversational AI. Summarize, extract insights, and more!"),
    ("🧑‍💼", "Agents", "Create custom AI agents with unique personalities and knowledge for specialized tasks."),
]

feat_cols = st.columns(len(features))
for idx, (icon, title, desc) in enumerate(features):
    with feat_cols[idx]:
        st.markdown(f"""
        <div style='background: rgba(44,83,100,0.13); border-radius: 1.1rem; padding: 1.1rem 0.7rem 1.1rem 0.7rem; margin-bottom: 0.5rem; min-height: 145px;'>
            <div style='font-size:2.1rem; margin-bottom:0.3rem;'>{icon}</div>
            <div style='font-weight:700; font-size:1.15rem; margin-bottom:0.3rem;'>{title}</div>
            <div style='font-size:0.99rem;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# --- How It Works ---
st.markdown("---")
st.header("How It Works")
steps = [
    ("1", "Sign Up or Log In", "Create an account to save and access your transcriptions."),
    ("2", "Record or Upload", "Start with your first audio file on the Playground page."),
    ("3", "Transcribe, Chat, Organize", "Use Flame Audio Chat and Agents to analyze and manage your results."),
]
step_cols = st.columns(len(steps))
for idx, (num, title, desc) in enumerate(steps):
    with step_cols[idx]:
        st.markdown(f"""
        <div style='background: rgba(44,83,100,0.10); border-radius: 1.1rem; padding: 1.1rem 0.7rem; margin-bottom: 0.5rem; min-height: 110px; text-align:center;'>
            <div style='font-size:1.5rem; font-weight:700; color:#1CB5E0; margin-bottom:0.2rem;'>{num}</div>
            <div style='font-weight:600; font-size:1.07rem; margin-bottom:0.2rem;'>{title}</div>
            <div style='font-size:0.97rem;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

# --- Bottom CTA Bar ---
st.markdown("---")
cta_cols = st.columns(3)
with cta_cols[0]:
    if st.button("🎙️ Start Transcribing", type="primary", use_container_width=True, key="cta_transcribe_btn"):
        st.switch_page("pages/flameaudio.py")
with cta_cols[1]:
    if st.button("📄 View Documents", use_container_width=True, key="cta_documents_btn"):
        st.switch_page("pages/documents.py")
with cta_cols[2]:
    if st.button("🤖 Flame Audio Chat", use_container_width=True, key="cta_chat_btn"):
        st.switch_page("pages/mcp_chat.py")

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
