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
- **User Authentication**: Optional account creation to save transcription history (powered by Supabase)
- **Responsive Design**: Clean, modern UI that works across devices

## Prerequisites

- Python 3.9+ 
- FastAPI backend server
- Groq API key
- Supabase account (for authentication features)

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

1. **API Key**: Enter your Groq API key in the sidebar (or create an account to save it)
2. **Select Model**: Choose a speech transcription model from the dropdown
3. **Create Audio**: Either record audio directly with the microphone button or upload an audio file
4. **Transcribe**: Click the Transcribe button to process your audio
5. **View Results**: See the transcribed text and use the buttons to copy, download or clear results

### Advanced Options

- **System Messages**: Enter system prompts to guide the transcription model
- **Task Selection**: Choose between transcription or translation to English
- **Language Selection**: Manually select the source language or use auto-detection
- **Parameter Adjustment**: Fine-tune chunk length, overlap and temperature for better results
- **View Options**: Switch between Text, Segments, and JSON views for different detail levels

## Configuration

The application is configured through environment variables in the `.env` file:

- `GROQ_API_KEY`: Your API key for the Groq speech-to-text service
- `SUPABASE_URL`: URL of your Supabase project for authentication
- `SUPABASE_ANON_KEY`: Anonymous key for your Supabase project

## Security Notes

- User authentication tokens are securely stored in an encrypted file
- API keys are never exposed in the frontend code
- HTTPS is recommended for production deployments

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- Website: [http://flameheadlabs.tech/](http://flameheadlabs.tech/)
- GitHub: [https://github.com/Flamehead-Labs-Ug/flame-audio](https://github.com/Flamehead-Labs-Ug/flame-audio)
- Twitter: [https://x.com/flameheadlabsug](https://x.com/flameheadlabsug)
- LinkedIn: [https://www.linkedin.com/in/flamehead-labs-919910285](https://www.linkedin.com/in/flamehead-labs-919910285)

---

<p align="center">
  <b>Powered by FlameheadLabs AI</b>
</p>
