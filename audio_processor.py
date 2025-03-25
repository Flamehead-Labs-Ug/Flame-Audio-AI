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

def transcribe_audio_in_chunks(audio_path: Path, chunk_length: int = 600, overlap: int = 10, model: str = None, temperature: float = 0.0, task: str = "transcribe") -> dict:
    """Transcribe audio file in chunks with overlap using Groq API directly.
    
    Args:
        audio_path (Path): Path to the audio file
        chunk_length (int): Length of each chunk in seconds
        overlap (int): Overlap between chunks in seconds
        model (str): Model to use for transcription
        temperature (float): Temperature for transcription (controls randomness)
        task (str): Task to perform - "transcribe" or "translate"
        
    Returns:
        dict: Transcription results including text and segments
    """
    try:
        # Get Groq API key from environment
        groq_api_key = os.getenv("GROQ_API_KEY")
        
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")
        
        if not model:
            raise ValueError("Model must be provided")
            
        # Create an AudioProcessor instance and process the file
        processor = AudioProcessor(
            api_key=groq_api_key,
            model=model,
            chunk_length_ms=chunk_length * 1000,  # Convert to milliseconds
            overlap_ms=overlap * 1000,  # Convert to milliseconds
            temperature=temperature,
            task=task  # Pass the task parameter
        )
        
        # Use the absolute path to the audio file
        result = processor.process_audio(str(audio_path.absolute()))
        return result
    
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise Exception(f"Error processing audio: {str(e)}")

class AudioProcessor:
    def __init__(self, api_key: str, model: str, chunk_length_ms: int = 600000, overlap_ms: int = 10000, temperature: float = 0.0, language: Optional[str] = None, task: str = "transcribe"):
        self.language = language
        self.api_key = api_key
        self.task = task
        if not model:
            raise ValueError("Model must be provided")
        self.model = model  # Model is now required
        
        # Use different endpoints based on task
        if self.task == "translate":
            self.api_url = "https://api.groq.com/openai/v1/audio/translations"
        else:  # Default to transcription
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
        try:
            # Generate unique identifier for this transcription
            temp_prefix = next(tempfile._get_candidate_names())
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"{temp_prefix}_{timestamp}"
            
            logger.info(f"Processing audio file: {file_path}")
            logger.info(f"Base filename: {base_filename}")
            logger.info(f"Temp directory: {self.temp_dir}")
            logger.info(f"Transcriptions directory: {self.transcriptions_dir}")
            
            # Ensure directories exist
            os.makedirs(self.temp_dir, exist_ok=True)
            os.makedirs(self.transcriptions_dir, exist_ok=True)

            # Check if file exists
            if not os.path.exists(file_path):
                error_msg = f"Audio file not found: {file_path}"
                logger.error(error_msg)
                raise FileIOError(error_msg)
            
            # Get file size for estimation if needed
            file_size = os.path.getsize(file_path)
            logger.info(f"File size: {file_size / 1024:.2f} KB")
            
            # Default to file size estimation for duration
            # Rough estimate: 1MB ≈ 1 minute of audio at moderate quality
            estimated_duration_sec = (file_size / (1024 * 1024)) * 60
            duration_ms = estimated_duration_sec * 1000
            logger.info(f"Estimated duration based on file size: {duration_ms/1000:.2f} seconds")
            
            # Try to load with pydub if available
            try:
                audio = AudioSegment.from_file(file_path)
                duration_ms = len(audio)
                logger.info(f"Audio duration from pydub: {duration_ms/1000:.2f} seconds")
            except Exception as e:
                logger.warning(f"Could not load audio with pydub: {str(e)}")
                # Continue with our file size estimation
                pass
            
            # Calculate number of chunks
            total_chunks = max(1, int((duration_ms - self.overlap_ms) / (self.chunk_length_ms - self.overlap_ms)) + 1)
            logger.info(f"Processing audio in {total_chunks} chunks with {self.overlap_ms}ms overlap")
            
            # Initialize results
            all_transcriptions = []
            full_text = ""
            all_segments = []
            
            # Skip chunking if audio is short enough
            if duration_ms <= self.chunk_length_ms:
                logger.info(f"Audio is short enough to process in one go ({duration_ms/1000:.2f}s <= {self.chunk_length_ms/1000:.2f}s)")
                result = self._send_chunk_with_retry(file_path)
                return result
            
            # Process audio in chunks
            for i in range(total_chunks):
                # Calculate start and end times for this chunk
                start_ms = max(0, i * (self.chunk_length_ms - self.overlap_ms))
                end_ms = min(duration_ms, start_ms + self.chunk_length_ms)
                
                # Skip processing if we've reached the end of the audio
                if start_ms >= duration_ms:
                    break
                
                logger.info(f"Processing chunk {i+1}/{total_chunks}: {start_ms/1000:.2f}s to {end_ms/1000:.2f}s")
                
                # For non-chunked audio, send directly
                if total_chunks == 1:
                    return self._send_chunk_with_retry(file_path)
                
                # For chunked audio, create temporary chunk file
                temp_filename = os.path.join(self.temp_dir, f"{base_filename}_chunk_{i+1}.mp3")
                
                # Try direct API call with original file for first chunk
                if i == 0 and start_ms == 0 and end_ms >= duration_ms * 0.8:  # If first chunk covers 80% or more
                    logger.info("First chunk covers most of audio, using direct processing")
                    try:
                        result = self._send_chunk_with_retry(file_path)
                        return result
                    except Exception as e:
                        logger.warning(f"Direct processing failed, falling back to chunking: {str(e)}")
                
                # Use direct file instead of chunking if we can't find ffmpeg
                try:
                    # Check if ffmpeg is available
                    import shutil
                    ffmpeg_available = shutil.which("ffmpeg") is not None
                    if not ffmpeg_available:
                        logger.warning("FFmpeg not found in PATH, trying direct processing without chunking")
                        result = self._send_chunk_with_retry(file_path)
                        return result
                        
                    # Extract the chunk using ffmpeg
                    ffmpeg_cmd = [
                        "ffmpeg", "-y", "-i", file_path, 
                        "-ss", f"{start_ms/1000:.2f}", 
                        "-to", f"{end_ms/1000:.2f}", 
                        "-c", "copy", 
                        temp_filename
                    ]
                    
                    logger.info(f"Running FFmpeg command: {' '.join(ffmpeg_cmd)}")
                    process = subprocess.run(
                        ffmpeg_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    if process.returncode != 0:
                        logger.error(f"FFmpeg error: {process.stderr}")
                        raise Exception(f"FFmpeg error: {process.stderr}")
                    
                    # Send the chunk for transcription
                    result = self._send_chunk_with_retry(temp_filename)
                    
                    # Append results
                    all_transcriptions.append(result)
                    full_text += result["text"] + " "
                    
                    # Adjust segment timestamps
                    if "segments" in result:
                        for segment in result["segments"]:
                            segment["start"] += start_ms / 1000  # Convert back to seconds
                            segment["end"] += start_ms / 1000
                            all_segments.append(segment)
                    
                    # Clean up temporary chunk file
                    try:
                        os.remove(temp_filename)
                    except Exception as e:
                        logger.warning(f"Error removing temporary chunk file: {str(e)}")
                
                except FileNotFoundError as e:
                    # If ffmpeg is not found, try sending the whole file directly
                    logger.warning(f"FFmpeg not found, trying direct processing: {str(e)}")
                    try:
                        result = self._send_chunk_with_retry(file_path)
                        return result
                    except Exception as direct_e:
                        logger.error(f"Direct processing also failed: {str(direct_e)}")
                        raise Exception(f"Cannot process audio: FFmpeg not available and direct processing failed")
                
                except Exception as e:
                    logger.error(f"Error processing chunk {i+1}: {str(e)}")
                    # Continue with the next chunk if possible
                    continue
            
            # Return combined results
            if all_segments:
                # Sort segments by start time
                all_segments.sort(key=lambda x: x["start"])
            
            combined_result = {
                "text": full_text.strip(),
                "segments": all_segments
            }
            
            # Save the results to files
            self._save_transcription_files(base_filename, combined_result)
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Error processing audio: {str(e)}")
            raise Exception(f"Error processing audio: {str(e)}")

    def _send_chunk_with_retry(self, temp_filename: str) -> dict:
        retries = 0
        while retries < self.max_retries:
            try:
                # Prepare the file for API submission
                with open(temp_filename, "rb") as f:
                    file_content = f.read()
                
                # Prepare the form data with the correct parameters
                form_data = {
                    "file": (os.path.basename(temp_filename), file_content, "audio/mpeg"),
                    "model": self.model,
                    "response_format": "verbose_json",
                    "temperature": str(self.temperature)
                }
                
                # Add language parameter only if specified
                if self.language and self.task == "transcribe":
                    form_data["language"] = self.language
                
                logger.info(f"Sending request to {self.api_url} with model {self.model}")
                logger.info(f"Task: {self.task}")
                
                # Make API request with form data
                files = {'file': (os.path.basename(temp_filename), file_content, 'audio/mpeg')}
                data = {
                    'model': self.model,
                    'response_format': 'verbose_json',
                    'temperature': str(self.temperature)
                }
                
                # Add language parameter only for transcription
                if self.language and self.task == "transcribe":
                    data['language'] = self.language
                
                # Make the API call
                response = httpx.post(
                    self.api_url,
                    files=files,
                    data=data,
                    headers={
                        "Authorization": f"Bearer {self.api_key}"
                    },
                    timeout=300  # Longer timeout for large files
                )
                
                # Check for successful response
                if response.status_code == 200:
                    logger.info("Successful API response received")
                    return response.json()
                else:
                    error_msg = f"API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise APICallError(error_msg)
            
            except (httpx.TimeoutException, httpx.ConnectError, httpx.ReadTimeout) as e:
                retries += 1
                logger.warning(f"Connection error (attempt {retries}/{self.max_retries}): {str(e)}")
                sleep_time = self.initial_backoff * (2 ** (retries - 1))  # Exponential backoff
                logger.info(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
            
            except Exception as e:
                error_msg = f"Error sending chunk to API: {str(e)}"
                logger.error(error_msg)
                raise APICallError(error_msg)
        
        # If we've exhausted all retries
        raise APICallError(f"Failed to get a successful response after {self.max_retries} attempts")

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