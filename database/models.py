from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field

# Session Models
class SessionCreate(BaseModel):
    user_id: str
    expiry_days: int = 7

class Session(BaseModel):
    id: str
    user_id: str
    created_at: datetime
    expires_at: datetime
    is_valid: bool

# Audio Job Models
class AudioJobCreate(BaseModel):
    user_id: str
    session_id: str
    file_name: str
    task_type: str
    original_language: str
    target_language: Optional[str] = "en"
    file_size_bytes: Optional[int] = None
    duration_seconds: Optional[float] = None
    settings: Dict[str, Any] = Field(default_factory=dict)

class AudioJob(BaseModel):
    id: str
    user_id: str
    session_id: Optional[str]
    created_at: datetime
    file_name: str
    task_type: str
    original_language: str
    target_language: str
    file_size_bytes: Optional[int]
    duration_seconds: Optional[float]
    status: str
    settings: Dict[str, Any]

# Transcription Models
class TranscriptionCreate(BaseModel):
    job_id: str
    chunk_index: int
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    text: str
    language: str

class Transcription(BaseModel):
    id: str
    job_id: str
    chunk_index: int
    start_time: Optional[float]
    end_time: Optional[float]
    text: str
    language: str

# Translation Models
class TranslationCreate(BaseModel):
    job_id: str
    transcription_id: Optional[str] = None
    chunk_index: int
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    original_text: Optional[str] = None
    translated_text: str
    source_language: str
    target_language: str = "en"

class Translation(BaseModel):
    id: str
    job_id: str
    transcription_id: Optional[str]
    chunk_index: int
    start_time: Optional[float]
    end_time: Optional[float]
    original_text: Optional[str]
    translated_text: str
    source_language: str
    target_language: str

# Document Processing Models
class DocumentData(BaseModel):
    """Model for document processing in a single API call"""
    filename: str
    task_type: str = "transcribe"
    original_language: str = "auto"
    model: str
    duration: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    agent_id: Optional[str] = None
    segments: List[Dict[str, Any]]
    embedding_settings: Optional[Dict[str, Any]] = None
    file_type: Optional[str] = "audio"

# Vector Search Models
class SearchQuery(BaseModel):
    query: str
    match_count: int = 5
    similarity_threshold: float = 0.7
    filter_metadata: Optional[Dict[str, Any]] = None

class SearchResult(BaseModel):
    transcription_id: str
    chunk_id: str
    similarity: float
    text: str
    file_name: str
