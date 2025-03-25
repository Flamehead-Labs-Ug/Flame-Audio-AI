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
- `SUPABASE_URL` and `SUPABASE_ANON_KEY`: Only required if AUTH_ENABLED=true

Optional configuration:
- `AUTH_ENABLED`: Set to "true" to enable authentication or "false" to disable it (default: "true")
- `API_KEY_INPUT_ENABLED`: Set to "true" to show the API key input field or "false" to hide it and use only the key from .env (default: "true")

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
