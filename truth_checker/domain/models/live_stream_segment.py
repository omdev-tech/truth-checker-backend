"""Domain model for live stream segments."""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class SegmentStatus(Enum):
    """Status of a live stream segment."""
    PENDING = "pending"
    RECORDING = "recording"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class LiveStreamSegment:
    """Represents a segment of a live stream for processing."""
    
    segment_id: int
    stream_url: str
    start_time: float  # seconds from stream start
    duration: float    # segment duration in seconds
    status: SegmentStatus
    created_at: datetime
    
    # Optional processing results
    audio_file_path: Optional[str] = None
    transcription_text: Optional[str] = None
    transcription_confidence: Optional[float] = None
    fact_check_result: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    
    # Metadata
    overlap_duration: float = 5.0  # overlap with previous segment
    metadata: Optional[Dict[str, Any]] = None
    
    @property
    def end_time(self) -> float:
        """Calculate end time of the segment."""
        return self.start_time + self.duration
    
    @property
    def is_completed(self) -> bool:
        """Check if segment processing is completed."""
        return self.status == SegmentStatus.COMPLETED
    
    @property
    def has_error(self) -> bool:
        """Check if segment has an error."""
        return self.status == SegmentStatus.ERROR
    
    @property
    def is_processing(self) -> bool:
        """Check if segment is currently being processed."""
        return self.status in [SegmentStatus.RECORDING, SegmentStatus.PROCESSING]
    
    def mark_recording(self, audio_file_path: str) -> None:
        """Mark segment as recording with audio file path."""
        self.status = SegmentStatus.RECORDING
        self.audio_file_path = audio_file_path
    
    def mark_processing(self) -> None:
        """Mark segment as processing."""
        self.status = SegmentStatus.PROCESSING
    
    def mark_completed(
        self, 
        transcription_text: str,
        transcription_confidence: float,
        fact_check_result: Dict[str, Any],
        processing_time: float
    ) -> None:
        """Mark segment as completed with results."""
        self.status = SegmentStatus.COMPLETED
        self.transcription_text = transcription_text
        self.transcription_confidence = transcription_confidence
        self.fact_check_result = fact_check_result
        self.processing_time = processing_time
    
    def mark_error(self, error_message: str) -> None:
        """Mark segment as error with message."""
        self.status = SegmentStatus.ERROR
        self.error_message = error_message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert segment to dictionary for API responses."""
        return {
            'segment_id': self.segment_id,
            'stream_url': self.stream_url,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'transcription': {
                'text': self.transcription_text,
                'confidence': self.transcription_confidence
            } if self.transcription_text else None,
            'fact_check': self.fact_check_result,
            'processing_time': self.processing_time,
            'error_message': self.error_message,
            'metadata': self.metadata or {}
        }


@dataclass
class LiveStreamSession:
    """Represents a live stream processing session."""
    
    session_id: str
    stream_url: str
    stream_type: str
    started_at: datetime
    is_active: bool = True
    
    # Configuration
    segment_duration: float = 30.0
    overlap_duration: float = 5.0
    segment_interval: float = 25.0  # create new segment every 25s
    
    # State
    current_segment_id: int = 0
    total_segments: int = 0
    completed_segments: int = 0
    error_segments: int = 0
    
    # Optional metadata
    stream_metadata: Optional[Dict[str, Any]] = None
    stopped_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def next_segment_id(self) -> int:
        """Get the next segment ID and increment counter."""
        segment_id = self.current_segment_id
        self.current_segment_id += 1
        self.total_segments += 1
        return segment_id
    
    def calculate_segment_start_time(self, segment_id: int) -> float:
        """Calculate start time for a segment with overlap."""
        return segment_id * self.segment_interval
    
    def mark_segment_completed(self) -> None:
        """Mark a segment as completed."""
        self.completed_segments += 1
    
    def mark_segment_error(self) -> None:
        """Mark a segment as error."""
        self.error_segments += 1
    
    def stop_session(self, error_message: Optional[str] = None) -> None:
        """Stop the live stream session."""
        self.is_active = False
        self.stopped_at = datetime.now()
        if error_message:
            self.error_message = error_message
    
    @property
    def processing_segments(self) -> int:
        """Calculate number of segments currently processing."""
        return self.total_segments - self.completed_segments - self.error_segments
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of segment processing."""
        if self.total_segments == 0:
            return 0.0
        return self.completed_segments / self.total_segments
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session to dictionary for API responses."""
        return {
            'session_id': self.session_id,
            'stream_url': self.stream_url,
            'stream_type': self.stream_type,
            'started_at': self.started_at.isoformat(),
            'stopped_at': self.stopped_at.isoformat() if self.stopped_at else None,
            'is_active': self.is_active,
            'configuration': {
                'segment_duration': self.segment_duration,
                'overlap_duration': self.overlap_duration,
                'segment_interval': self.segment_interval
            },
            'statistics': {
                'total_segments': self.total_segments,
                'completed_segments': self.completed_segments,
                'processing_segments': self.processing_segments,
                'error_segments': self.error_segments,
                'success_rate': self.success_rate
            },
            'stream_metadata': self.stream_metadata,
            'error_message': self.error_message
        } 