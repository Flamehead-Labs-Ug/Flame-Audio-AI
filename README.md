# Flame Audio

<p align="center">
  <img src="logos/flame logo.jpg" alt="Flame Audio Logo" width="200"/>
</p>

## Overview

Flame Audio is a powerful speech transcription, translation, and AI chat application built by FlameheadLabs. It allows users to easily convert spoken language to text through recorded audio or uploaded files, with optional account creation for saving transcription history and chatting with AI agents about your documents.

## Features

### Audio Processing
- **Live Audio Recording**: Record audio directly in the browser with high-quality settings
- **File Upload**: Support for multiple audio formats (flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm)
- **Multiple Models**: Integration with Groq API's advanced audio transcription models
- **Language Support**: Automatic language detection or manual language selection
- **Task Options**: Transcribe in original language or translate to English

### Document Management
- **Vector Store Integration**: Store and retrieve documents using advanced vector embeddings
- **Multiple Embedding Models**: Select from various embedding models for optimal semantic search
- **Customizable Chunking**: Configure chunk size and overlap to optimize document retrieval
- **Document Organization**: Associate documents with specific AI agents

### Chat Capabilities
- **AI Chat Agents**: Create customizable AI agents with different personalities and expertise
- **Document-Aware Chat**: Chat about your transcribed documents with contextual understanding
- **Chat Session Management**: Save and reload chat sessions for continued conversations
- **Realtime Updates**: Vector store configurations update in real-time across the application

### User Experience
- **Intuitive Navigation**: Easily move between transcription, document storage, and chat interfaces
- **Responsive Design**: Clean, modern UI that works across devices
- **Flexible Authentication**: Optional user authentication that can be enabled/disabled via configuration
- **Persistent Settings**: User preferences persist between sessions

## Prerequisites

- Python 3.9+ 
- FastAPI backend server
- Groq API key
- PostgreSQL database
- Supabase account (for authentication features, optional if AUTH_ENABLED=false)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Flamehead-Labs-Ug/flame-audio.git
cd flame-audio
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the `.env.template` file to `.env` and fill in your API keys:

```bash
cp .env.template .env
# Edit the .env file with your credentials
```

Required environment variables:
- `GROQ_API_KEY`: Your Groq API key
- `BACKEND_URL`: URL for the FastAPI backend (e.g., http://localhost:8000)
- `FRONTEND_URL`: URL for the Streamlit frontend (e.g., http://localhost:8501)
- `SUPABASE_URL` and `SUPABASE_ANON_KEY`: Only required if AUTH_ENABLED=true
- `DATABASE_URL`: PostgreSQL connection string

Optional configuration:
- `AUTH_ENABLED`: Set to "true" to enable authentication or "false" to disable it (default: "true")
- `API_KEY_INPUT_ENABLED`: Set to "true" to show the API key input field or "false" to hide it and use only the key from .env (default: "true")
- `EMBEDDING_PROVIDER`: Choose embedding provider ("local", "openai", or "hf-inference")
- `HUGGINGFACE_API_KEY`: Required if using HuggingFace Inference API for embeddings

Here's an example of what your `.env` file might look like:
```
# Groq API Configuration
GROQ_API_KEY=your_groq_api_key
API_KEY_INPUT_ENABLED=true  # Set to false to hide the API key input field

# Backend URL Configuration (required)
BACKEND_URL=http://localhost:8000  # Required
FRONTEND_URL=http://localhost:8501  # URL for the Streamlit frontend

# Supabase Configuration for Authentication
SUPABASE_URL=your_supabase_url_here
SUPABASE_ANON_KEY=your_supabase_anon_key_here
AUTH_ENABLED=true  # Set to false to disable authentication

# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/flame_audio

# Vector Store Configuration
EMBEDDING_PROVIDER=hf-inference  # Options: local, openai, hf-inference
HUGGINGFACE_API_KEY=your_huggingface_api_key  # Only needed for hf-inference
```

## Running the Application

### 1. Start the FastAPI backend

```bash
python -m uvicorn main:app --reload
```

### 2. Launch the Streamlit frontend

In a new terminal window:

```bash
streamlit run streamlit_app.py
```

### 3. Access the application

Open your web browser and navigate to:
```
http://localhost:8501
```

## Usage Guide

### Transcription and Translation

1. **Authentication**: Log in or create an account (if authentication is enabled)
2. **API Key**: Enter your Groq API key in the sidebar
3. **Select Model**: Choose a speech transcription model from the dropdown
4. **Create Audio**: Either record audio directly with the microphone button or upload an audio file
5. **Task Selection**: Choose between transcription (original language) or translation (to English)
6. **Process Audio**: Click the Transcribe/Translate button to process your audio
7. **Save Document**: Save transcribed content to your vector store for future reference

### Document Management

1. Access the Documents page to view all your saved transcriptions
2. Filter documents by agent or search for specific content
3. Configure vector store settings for optimal document retrieval
4. Delete documents you no longer need

### Chat Interface

1. **Create or Select an Agent**: Choose an existing agent or create a new one
2. **Configure Settings**: Adjust model settings, system prompt, and vector store parameters
3. **Choose Documents**: Select which documents the agent can access for context
4. **Start Chatting**: Engage in conversation with the AI about your transcribed content
5. **Manage Sessions**: Save multiple chat sessions and switch between them

### Advanced Options

- **Vector Store Configuration**: Fine-tune embedding model, chunk size, and similarity thresholds
- **Language Selection**: Manually select the source language or use auto-detection
- **Chunk Length**: Control how audio is divided for processing (in seconds)
- **Overlap**: Set overlap between chunks for smoother transitions
- **Temperature**: Adjust randomness in model output

## Technical Details

### API Endpoints

The application uses separate Groq API endpoints for different tasks:
- Transcription: https://api.groq.com/openai/v1/audio/transcriptions
- Translation: https://api.groq.com/openai/v1/audio/translations

### Authentication

Authentication is handled by Supabase and can be enabled/disabled through the `AUTH_ENABLED` environment variable.

## Deploying on AWS EC2

This guide will help you deploy the Flame Audio application on an AWS EC2 instance with Nginx as a reverse proxy.

### 1. Launch an EC2 Instance

Launch an Ubuntu EC2 instance with at least 2GB RAM. Make sure to allow HTTP (port 80) and HTTPS (port 443) traffic in your security group settings.

### 2. Connect to Your Instance

```bash
ssh -i your-key.pem ubuntu@your-ec2-public-ip
```

### 3. Install Required Packages

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv nginx certbot python3-certbot-nginx ffmpeg
```

### 4. Clone the Repository

```bash
git clone https://github.com/Flamehead-Labs-Ug/flame-audio.git
cd flame-audio
```

### 5. Set Up a Python Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 6. Configure Environment Variables

```bash
cp .env.template .env
nano .env
```

Edit the `.env` file to set your configuration:

```
GROQ_API_KEY=your_groq_api_key
API_KEY_INPUT_ENABLED=false
AUTH_ENABLED=true
BACKEND_URL=https://your-domain-name/api  # Use your domain name if configured
FRONTEND_URL=https://your-domain-name     # Must match your domain name for authentication to work
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
DATABASE_URL=postgresql://username:password@localhost:5432/flame_audio
```

> **Important**: When using a custom domain, make sure both `BACKEND_URL` and `FRONTEND_URL` use your domain name rather than the EC2 IP address. This is crucial for authentication workflows and API requests to function correctly.

### 7. Set Up Systemd Services

#### Create FastAPI Backend Service

```bash
sudo nano /etc/systemd/system/flame-fastapi.service
```

Add the following content:

```
[Unit]
Description=Flame Audio FastAPI Backend
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/flame-audio
EnvironmentFile=/home/ubuntu/flame-audio/.env
ExecStart=/home/ubuntu/flame-audio/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

#### Create Streamlit Frontend Service

```bash
sudo nano /etc/systemd/system/flame-streamlit.service
```

Add the following content:

```
[Unit]
Description=Flame Audio Streamlit Frontend
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/flame-audio
EnvironmentFile=/home/ubuntu/flame-audio/.env
ExecStart=/home/ubuntu/flame-audio/venv/bin/streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501
Restart=always

[Install]
WantedBy=multi-user.target
```

### 8. Start and Enable the Services

```bash
sudo systemctl start flame-fastapi
sudo systemctl enable flame-fastapi
sudo systemctl start flame-streamlit
sudo systemctl enable flame-streamlit
```

### 9. Configure Nginx

```bash
sudo nano /etc/nginx/sites-available/flame-audio
```

Add the following content, replacing `your-domain-name` with your actual domain and `your-ec2-public-ip` with your EC2 instance's public IP address:

```
server {
    server_name your-domain-name your-ec2-public-ip;
    
    client_max_body_size 100M;
    
    # FastAPI Backend
    location /api/ {
        rewrite ^/api/(.*)$ /$1 break;
        proxy_pass http://your-ec2-public-ip:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        client_max_body_size 100M;
    }
    
    # Streamlit Frontend
    location / {
        proxy_pass http://your-ec2-public-ip:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 86400;
    }
}
```

#### Enable the Nginx Configuration

```bash
sudo ln -s /etc/nginx/sites-available/flame-audio /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 10. Set Up SSL with Let's Encrypt

```bash
sudo certbot --nginx -d your-domain-name
```

### 11. Verify the Installation

Access your application by visiting your domain name in a browser. You should see the Flame Audio interface.

## Troubleshooting Deployment Issues

### 502 Bad Gateway Errors

If you see "502 Bad Gateway" errors when accessing your application:

1. **Check if services are running:**
   ```bash
   sudo systemctl status flame-fastapi
   sudo systemctl status flame-streamlit
   ```

2. **Verify both services are accessible directly:**
   ```bash
   curl http://localhost:8000/languages
   curl http://localhost:8501 -I
   ```

3. **Install ffmpeg if it's missing:**
   ```bash
   sudo apt install -y ffmpeg
   ```

4. **Check Nginx error logs:**
   ```bash
   sudo tail -n 50 /var/log/nginx/error.log
   ```

5. **If you see "Connection refused" errors**, update your Nginx config to use your EC2 public IP directly in the proxy_pass directives.

### Authentication Issues

If users can see each other's logins or authentication isn't working properly:

1. **Make sure FRONTEND_URL and BACKEND_URL are correctly set** in your .env file.

2. **Consider disabling authentication** for testing by setting `AUTH_ENABLED=false` in your .env file.

3. **Check for Supabase version compatibility issues**. If you see proxy-related errors:
   ```bash
   pip uninstall -y supabase gotrue httpx postgrest realtime storage3 supafunc
   pip install supabase==1.0.3 gotrue==1.0.1
   ```

### Updating Your Deployment

To update your deployment with the latest code:

```bash
cd ~/flame-audio
git pull
sudo systemctl restart flame-fastapi
sudo systemctl restart flame-streamlit
sudo systemctl restart nginx
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

- Website: [http://flameheadlabs.tech/](http://flameheadlabs.tech/)
- GitHub: [https://github.com/Flamehead-Labs-Ug/flame-audio](https://github.com/Flamehead-Labs-Ug/flame-audio)
- Twitter: [https://x.com/flameheadlabsug](https://x.com/flameheadlabsug)
- LinkedIn: [https://www.linkedin.com/in/flamehead-labs-919910285](https://www.linkedin.com/in/flamehead-labs-919910285)

---

<p align="center">
  <b>Powered by FlameheadLabs AI</b>
</p>
