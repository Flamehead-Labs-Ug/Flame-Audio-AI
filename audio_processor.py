import httpx
import time
import subprocess
from datetime import datetime

class AudioProcessingError(Exception):
    """Base class for audio processing exceptions."""
    pass

class APICallError(AudioProcessingError):
    """Exception raised for API communication errors."""
    pass

class FileIOError(AudioProcessingError):
    """Exception raised for file reading/writing errors."""
    pass

import logging
import os
import tempfile
from pathlib import Path
import json

import os
from io import BytesIO
from pydub import AudioSegment

logger = logging.getLogger(__name__)

from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transcribe_audio_in_chunks(audio_path: Path, chunk_length: int = 600, overlap: int = 10, model: str = None) -> dict:
    """Transcribe audio file in chunks with overlap using Groq API directly.
    
    Args:
        audio_path (Path): Path to the audio file
        chunk_length (int): Length of each chunk in seconds
        overlap (int): Overlap between chunks in seconds
        model (str): Model to use for transcription
        
    Returns:
        dict: Transcription results including text and segments
    """
    try:
        # Get Groq API key from environment
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
            
        processor = AudioProcessor(
            api_key=groq_api_key,
            model=model,
            chunk_length_ms=chunk_length * 1000  # Convert seconds to milliseconds
        )
        
        logger.info(f"Starting transcription of {audio_path} using {model}")
        logger.info(f"Chunk length: {chunk_length}s, Overlap: {overlap}s")
        
        logger.debug(f"Processor config - chunk_length_ms: {processor.chunk_length_ms}, overlap_ms: {processor.overlap_ms}")
        logger.debug(f"Audio file path: {audio_path}")
        
        results = processor.process_audio(str(audio_path))
        logger.info("Transcription completed successfully")
        
        return results
    except ValueError as e:
        logger.error(f"Configuration error: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Error in transcribe_audio_in_chunks: {str(e)}")
        logger.error(f"Failed processing audio file: {audio_path}")
        logger.error(f"Model used: {model}")
        raise

class AudioProcessor:
    def __init__(self, api_key: str, model: str, chunk_length_ms: int = 600000, overlap_ms: int = 10000, temperature: float = 0.0, language: Optional[str] = None):
        self.language = language
        self.api_key = api_key
        if not model:
            raise ValueError("Model must be provided")
        self.model = model  # Model is now required
        self.api_url = "https://api.groq.com/openai/v1/audio/transcriptions"
        self.chunk_length_ms = chunk_length_ms  # Default 10 minutes in milliseconds
        self.overlap_ms = overlap_ms  # Default 10 seconds in milliseconds
        self.temperature = temperature  # Add temperature parameter from frontend
        self.transcriptions_dir = Path("transcriptions")
        self.transcriptions_dir.mkdir(exist_ok=True)
        # Create a dedicated temp directory within the project
        self.temp_dir = Path("temp")
        self.temp_dir.mkdir(exist_ok=True)
        # Configure httpx client with timeouts and authentication
        self.client = httpx.Client(
            timeout=httpx.Timeout(connect=30.0, read=300.0, write=120.0, pool=300.0),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Accept": "application/json"
            }
        )
        self.max_retries = 5  # Increased retries
        self.initial_backoff = 2  # seconds

    def process_audio(self, file_path: str) -> dict:
        """Process audio file into chunks and transcribe each chunk

        Args:
            file_path (str): Path to the audio file

        Returns:
            dict: Transcription results including text and segments

        Raises:
            AudioProcessingError: If there's an error processing the audio
            APICallError: If there's an error communicating with the API
            FileIOError: If there's an error reading/writing files
        """
        # Generate unique identifier for this transcription
        temp_prefix = next(tempfile._get_candidate_names())
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{temp_prefix}_{timestamp}"
        
        logger.debug(f"Processing audio file: {file_path}")
        logger.debug(f"Base filename: {base_filename}")
        logger.debug(f"Temp directory: {self.temp_dir}")
        logger.debug(f"Transcriptions directory: {self.transcriptions_dir}")
        
        logger.debug(f"Processing audio file: {file_path}")
        logger.debug(f"Base filename: {base_filename}")
        logger.debug(f"Temp directory: {self.temp_dir}")
        logger.debug(f"Transcriptions directory: {self.transcriptions_dir}")

        try:
            # Load audio file using pydub
            try:
                audio = AudioSegment.from_file(file_path)
                duration_ms = len(audio)
                logger.info(f"Audio duration: {duration_ms/1000:.2f} seconds")
            except Exception as e:
                logger.error(f"Error loading audio file: {str(e)}")
                # Fallback to getting duration using mediainfo
                try:
                    from pydub.utils import mediainfo
                    info = mediainfo(file_path)
                    duration_ms = float(info.get('duration', 0)) * 1000  # Convert to milliseconds
                    
                    if duration_ms == 0:
                        # If mediainfo fails to get duration, try using ffmpeg directly
                        logger.warning(f"Could not determine audio duration using mediainfo for {file_path}, trying ffmpeg")
                        
                        # Use ffmpeg to get duration
                        ffmpeg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "ffmpeg-7.1-full_build", "bin", "ffmpeg.exe")
                        if not os.path.exists(ffmpeg_path):
                            ffmpeg_path = "ffmpeg"
                            
                        cmd = [ffmpeg_path, "-i", file_path]
                        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        
                        # Parse duration from ffmpeg output
                        for line in process.stderr.split('\n'):
                            if 'Duration:' in line:
                                time_str = line.split('Duration:')[1].split(',')[0].strip()
                                h, m, s = time_str.split(':')
                                s, ms = s.split('.')
                                duration_ms = (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms) * 10
                                break
                        
                        if duration_ms == 0:
                            raise Exception(f"Could not determine audio duration for {file_path}")
                except Exception as inner_e:
                    logger.error(f"Error getting audio duration: {str(inner_e)}")
                    # Fallback to a conservative estimate based on file size
                    file_size = os.path.getsize(file_path)
                    # Rough estimate: 1MB ≈ 1 minute of audio at moderate quality
                    estimated_duration_sec = (file_size / (1024 * 1024)) * 60
                    duration_ms = estimated_duration_sec * 1000
                    logger.warning(f"Using estimated duration based on file size: {duration_ms/1000:.2f} seconds")

            
            # Calculate number of chunks
            total_chunks = max(1, int((duration_ms - self.overlap_ms) / (self.chunk_length_ms - self.overlap_ms)) + 1)
            logger.info(f"Processing audio in {total_chunks} chunks with {self.overlap_ms}ms overlap")
            
            # Process audio in chunks without loading the entire file
            all_segments = []
            for i in range(total_chunks):
                # Calculate chunk start and end times
                start_time_ms = i * (self.chunk_length_ms - self.overlap_ms)
                end_time_ms = min(start_time_ms + self.chunk_length_ms, duration_ms)
                
                logger.debug(f"Processing chunk {i+1}/{total_chunks} ({start_time_ms/1000:.2f}s to {end_time_ms/1000:.2f}s)")
                
                # Create temporary file for this chunk
                temp_filename = self.temp_dir / f"chunk_{i}_{base_filename}.wav"
                
                # Use ffmpeg to extract just this chunk directly to a file
                # This avoids loading the entire audio file into memory
                start_time_sec = start_time_ms / 1000
                duration_sec = (end_time_ms - start_time_ms) / 1000
                
                try:
                    # Use ffmpeg to extract the chunk
                    # Use the ffmpeg executable from the tools directory
                    ffmpeg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "ffmpeg-7.1-full_build", "bin", "ffmpeg.exe")
                    
                    # Check if ffmpeg exists at the specified path, otherwise use the command directly
                    if not os.path.exists(ffmpeg_path):
                        ffmpeg_path = "ffmpeg"
                        
                    ffmpeg_cmd = [
                        ffmpeg_path, "-y", "-i", file_path, 
                        "-ss", str(start_time_sec), 
                        "-t", str(duration_sec),
                        "-acodec", "pcm_s16le", 
                        "-ar", "16000", 
                        "-ac", "1",
                        str(temp_filename)
                    ]
                    
                    # Run ffmpeg command
                    process = subprocess.run(
                        ffmpeg_cmd, 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    if process.returncode != 0:
                        logger.warning(f"FFmpeg warning for chunk {i+1}: {process.stderr}")
                    
                    # Send chunk to API with retry mechanism
                    logger.debug(f"Sending chunk {i+1}/{total_chunks} to API")
                    
                    if os.path.exists(temp_filename) and os.path.getsize(temp_filename) > 0:
                        result = self._send_chunk_with_retry(temp_filename)
                        logger.debug(f"API response for chunk {i+1}/{total_chunks}: {json.dumps(result, indent=2)}")
                        
                        # Keep the original segments from API response
                        if 'segments' in result:
                            logger.debug(f"Received {len(result['segments'])} segments from API")
                            
                            # Adjust timing for each segment based on chunk position
                            for segment in result['segments']:
                                segment['start'] += start_time_sec  # Add chunk start time
                                segment['end'] += start_time_sec
                            all_segments.extend(result['segments'])
                        else:
                            logger.warning(f"No segments in API response for chunk {i+1}/{total_chunks}")
                            
                            # Fallback if API doesn't return segments
                            segment = {
                                'text': result.get('text', ''),
                                'start': start_time_sec,
                                'end': start_time_sec + duration_sec
                            }
                            all_segments.append(segment)
                    else:
                        logger.error(f"Failed to create chunk file or file is empty: {temp_filename}")
                        
                finally:
                    # Clean up temporary chunk file
                    try:
                        if os.path.exists(temp_filename):
                            os.unlink(temp_filename)
                    except Exception as e:
                        logger.error(f"Error deleting temporary file: {e}")
            
            # Combine all transcriptions
            combined_text = ' '.join(segment['text'] for segment in all_segments)
            
            # Save results
            # Use the specified language or 'auto' if not specified
            detected_language = self.language if self.language else 'auto'
            
            results = {
                'task': 'transcribe',
                'language': detected_language,
                'duration': duration_ms / 1000.0,  # Convert ms to seconds
                'text': combined_text,
                'segments': all_segments,
                'metadata': {
                    'filename': os.path.basename(file_path),
                    'duration_ms': duration_ms,
                    'num_chunks': total_chunks,
                    'chunk_size_ms': self.chunk_length_ms
                }
            }
            
            # Save transcription files
            self._save_transcription_files(base_filename, results)
            
            return results
            
        except Exception as e:
            raise Exception(f"Error processing audio: {str(e)}")
    
    def _send_chunk_with_retry(self, temp_filename: str) -> dict:
        retries = 0
        while retries < self.max_retries:
            try:
                with open(temp_filename, 'rb') as audio_file:
                    files = {'file': (os.path.basename(temp_filename), audio_file, 'audio/wav')}
                    # Check file size before sending
                    file_size = os.path.getsize(temp_filename)
                    free_tier_limit = 40 * 1024 * 1024  # 40MB limit for free tier
                    dev_tier_limit = 100 * 1024 * 1024  # 100MB limit for dev tier
                    
                    if file_size > dev_tier_limit:
                        raise Exception(f"File size {file_size/1024/1024:.2f}MB exceeds the maximum limit of 100MB")
                    elif file_size > free_tier_limit:
                        logger.warning(f"File size {file_size/1024/1024:.2f}MB exceeds free tier limit (40MB). Ensure you're on dev tier.")

                    # Use the specified model for transcription
                    # Prepare request data with validated language
                    data = {
                        'model': self.model,  # Use the model specified during initialization
                        'response_format': 'verbose_json',
                        'temperature': self.temperature
                    }
                    
                    # Only include language if it's specified and valid
                    if self.language:
                        # List of supported languages
                        supported_languages = ['pt', 'sv', 'hi', 'sr', 'pa', 'sd', 'tl', 'de', 'ja', 'th', 'ka', 'uz', 'haw', 'ha', 'no', 'kk', 'gl', 'ps', 'mg', 'tt', 'pl', 'cs', 'mi', 'br', 'yo', 'ln', 'en', 'id', 'el', 'yi', 'zh', 'ar', 'bn', 'sl', 'mr', 'oc', 'es', 'ur', 'km', 'mt', 'my', 'it', 'vi', 'lt', 'bs', 'af', 'be', 'gu', 'lo', 'te', 'az', 'et', 'sq', 'si', 'bo', 'yue', 'ru', 'tr', 'ro', 'da', 'lv', 'ne', 'as', 'mk', 'eu', 'so', 'lb', 'jv', 'ca', 'hr', 'cy', 'fa', 'is', 'am', 'ba', 'fi', 'bg', 'la', 'ml', 'hy', 'fo', 'he', 'ta', 'sn', 'tk', 'sa', 'ko', 'fr', 'ms', 'hu', 'sk', 'kn', 'sw', 'ht', 'nl', 'uk', 'mn', 'tg', 'nn', 'su']
                        
                        if self.language.lower() not in supported_languages:
                            raise ValueError(f"Unsupported language: {self.language}. Language must be one of: {supported_languages}")
                        
                        data['language'] = self.language.lower()
                    response = self.client.post(self.api_url, files=files, data=data)
                    
                    # Check for rate limits and other HTTP errors
                    if response.status_code == 429:
                        retry_after = int(response.headers.get('Retry-After', '1'))
                        logger.warning(f"Rate limit hit. Retrying after {retry_after} seconds...")
                        time.sleep(retry_after)
                        continue
                    
                    # Check for error response before raising for status
                    if response.status_code >= 400:
                        error_detail = "Unknown error"
                        try:
                            error_json = response.json()
                            if 'error' in error_json:
                                error_detail = error_json['error'].get('message', str(error_json['error']))
                            logger.error(f"API Error ({response.status_code}): {error_detail}")
                        except Exception:
                            # If we can't parse the JSON, use the text response
                            error_detail = response.text
                            logger.error(f"API Error ({response.status_code}): {error_detail}")
                        
                        raise Exception(f"API Error ({response.status_code}): {error_detail}")
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    # Ensure the response matches the expected format
                    if not isinstance(result, dict) or 'text' not in result:
                        logger.error(f"Invalid API response format: {json.dumps(result, indent=2)}")
                        raise Exception(f"Invalid API response format")
                    
                    if 'error' in result:
                        logger.error(f"API Error: {result['error']}")
                        raise Exception(f"API Error: {result['error']}")
                    
                    return result
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as e:
                retries += 1
                if retries == self.max_retries:
                    raise Exception(f"Failed after {self.max_retries} retries: {str(e)}")
                
                # Implement exponential backoff with minimum delay of 1 second
                wait_time = max(1, self.initial_backoff * (2 ** (retries - 1)))
                logger.warning(f"Attempt {retries} failed, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            except Exception as e:
                logger.error(f"Error in API request: {str(e)}")
                raise Exception(f"Error sending chunk to API: {str(e)}")

    def _save_transcription_files(self, base_filename: str, results: dict):
        # Save plain text transcription
        text_file = self.transcriptions_dir / f"{base_filename}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(results['text'])
        
        # Save full results including metadata
        full_file = self.transcriptions_dir / f"{base_filename}_full.json"
        with open(full_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        
        # Save segments separately
        segments_file = self.transcriptions_dir / f"{base_filename}_segments.json"
        with open(segments_file, 'w', encoding='utf-8') as f:
            json.dump(results['segments'], f, indent=2)