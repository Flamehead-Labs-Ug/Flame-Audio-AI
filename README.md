# Flame Speech to Text

<p align="center">
  <img src="logos/flame logo.jpg" alt="Flame Speech to Text Logo" width="200"/>
</p>

## Overview

Flame Speech to Text is a powerful speech transcription and translation application built by FlameheadLabs. It allows users to easily convert spoken language to text through recorded audio or uploaded files, with optional account creation for saving transcription history.

## Features

- **Live Audio Recording**: Record audio directly in the browser with high-quality settings
- **File Upload**: Support for multiple audio formats (flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm)
- **Multiple Models**: Integration with Groq API's advanced audio transcription models
- **Language Support**: Automatic language detection or manual language selection
- **Task Options**: Transcribe in original language or translate to English
- **Advanced Parameters**: Customize chunk length, overlap, and temperature settings
- **Result Formats**: View results as plain text, detailed segments, or raw JSON
- **Flexible Authentication**: Optional user authentication that can be enabled/disabled via configuration
- **User-Friendly Validation**: Clear error messages when required inputs are missing
- **Persistent Settings**: User preferences like model selection persist between sessions
- **Responsive Design**: Clean, modern UI that works across devices

## Prerequisites

- Python 3.9+ 
- FastAPI backend server
- Groq API key
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

Optional configuration:
- `AUTH_ENABLED`: Set to "true" to enable authentication or "false" to disable it (default: "true")
- `API_KEY_INPUT_ENABLED`: Set to "true" to show the API key input field or "false" to hide it and use only the key from .env (default: "true")

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

### Basic Usage

1. **Authentication**: Log in or create an account (if authentication is enabled)
2. **API Key**: Enter your Groq API key in the sidebar
3. **Select Model**: Choose a speech transcription model from the dropdown
4. **Create Audio**: Either record audio directly with the microphone button or upload an audio file
5. **Task Selection**: Choose between transcription (original language) or translation (to English)
6. **Process Audio**: Click the Transcribe/Translate button to process your audio
7. **View Results**: See the processed text and use the buttons to copy, download or clear results

### Advanced Options

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

## Troubleshooting

- **Model Selection**: You must select a model before transcribing or translating
- **Authentication**: When enabled, you must be logged in to use the application
- **API Key**: A valid Groq API key is required for all operations
- **Backend Connection**: Ensure the FastAPI backend is running and BACKEND_URL is properly configured

## License

This project is licensed under the [MIT License](LICENSE).

## AWS EC2 Deployment Guide

This guide provides step-by-step instructions for deploying Flame Audio on an AWS EC2 instance.

### 1. Set Up an EC2 Instance

1. Login to your AWS Management Console and navigate to EC2 Dashboard
2. Click "Launch instance"
3. Choose Ubuntu Server 22.04 LTS (or newer)
4. Select an instance type (t2.micro for testing, t2.small or better for production)
5. Configure security groups to allow:
   - SSH (Port 22) - For administration
   - HTTP (Port 80) - For web access
   - HTTPS (Port 443) - For secure web access (optional)
6. Launch the instance and create/select a key pair

### 2. Connect to Your EC2 Instance

```bash
ssh -i /path/to/your-key.pem ubuntu@your-ec2-public-ip
```

### 3. Install Required Dependencies

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python and other dependencies
sudo apt install -y python3 python3-pip python3-venv nginx

# Install development tools
sudo apt install -y build-essential python3-dev
```

### 4. Clone the Repository

```bash
git clone https://github.com/Flamehead-Labs-Ug/flame-audio.git
cd flame-audio
```

### 5. Set Up Python Environment

```bash
python3 -m venv venv
source venv/bin/activate
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
BACKEND_URL=http://your-domain-name/api  # Use your domain name if configured
FRONTEND_URL=http://your-domain-name     # Must match your domain name for authentication to work
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key
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
ExecStart=/home/ubuntu/flame-audio/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
Environment="PATH=/home/ubuntu/flame-audio/venv/bin"
EnvironmentFile=/home/ubuntu/flame-audio/.env

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
ExecStart=/home/ubuntu/flame-audio/venv/bin/streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
Restart=always
Environment="PATH=/home/ubuntu/flame-audio/venv/bin"
EnvironmentFile=/home/ubuntu/flame-audio/.env

[Install]
WantedBy=multi-user.target
```

### 8. Enable and Start Services

```bash
sudo systemctl daemon-reload
sudo systemctl enable flame-fastapi
sudo systemctl enable flame-streamlit
sudo systemctl start flame-fastapi
sudo systemctl start flame-streamlit
```

### 9. Configure Nginx as Reverse Proxy

```bash
sudo nano /etc/nginx/sites-available/flame-audio
```

Add the following configuration:

```
server {
    listen 80;
    server_name your-ec2-public-ip;
    
    client_max_body_size 100M;
    
    # FastAPI Backend
    location /api/ {
        proxy_pass http://127.0.0.1:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        client_max_body_size 100M;
    }
    
    # Streamlit Frontend
    location / {
        proxy_pass http://127.0.0.1:8501/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
        client_max_body_size 100M;
    }
}
```

Create a symbolic link and restart Nginx:

```bash
sudo ln -s /etc/nginx/sites-available/flame-audio /etc/nginx/sites-enabled/
sudo rm /etc/nginx/sites-enabled/default  # Optional: remove default site
sudo nginx -t  # Test configuration
sudo systemctl restart nginx
```

### 10. Verify the Deployment

Open your browser and navigate to your EC2 instance's public IP address:

```
http://your-ec2-public-ip
```

### 11. Troubleshooting

- **Check service status:**
  ```bash
  sudo systemctl status flame-fastapi
  sudo systemctl status flame-streamlit
  sudo systemctl status nginx
  ```

- **View service logs:**
  ```bash
  sudo journalctl -u flame-fastapi -n 100
  sudo journalctl -u flame-streamlit -n 100
  sudo cat /var/log/nginx/error.log
  ```

- **Common Issues:**
  - 502 Bad Gateway: FastAPI service not running or port conflict
  - 413 Request Entity Too Large: Increase client_max_body_size in Nginx config
  - Connection refused: Check security group settings in AWS console

### 12. (Optional) Set Up SSL with Let's Encrypt

For a production environment, you should secure your application with HTTPS:

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

Follow the prompts to complete the SSL certificate installation.

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
