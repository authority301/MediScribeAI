from sqlalchemy import BigInteger, Column, DateTime, ForeignKey, Integer, Numeric, Text, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.database.base import Base


class AudioRecord(Base):
    __tablename__ = "audio_records"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    consultation_id = Column(
        UUID(as_uuid=True), ForeignKey("consultations.id"), nullable=False, index=True
    )
    storage_path = Column(Text, nullable=False)
    original_filename = Column(Text, nullable=False)
    content_type = Column(Text, nullable=False)
    format = Column(Text, nullable=True)
    duration_seconds = Column(Numeric, nullable=True)
    sample_rate_hz = Column(Integer, nullable=True)
    file_size_bytes = Column(BigInteger, nullable=True)
    checksum = Column(Text, nullable=True)
    recorded_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # Step 8B: local Fog audio-processing metadata (pending/processing/completed/failed)
    processing_status = Column(Text, nullable=False, server_default=text("'pending'"))
    processed_storage_path = Column(Text, nullable=True)
    processed_content_type = Column(Text, nullable=True)
    processed_file_size_bytes = Column(BigInteger, nullable=True)
    processed_sample_rate_hz = Column(Integer, nullable=True)
    processed_channels = Column(Integer, nullable=True)
    processed_at = Column(DateTime(timezone=True), nullable=True)
    processing_error = Column(Text, nullable=True)

    # Step 10: local PyAnnote diarization metadata (pending/processing/completed/failed)
    diarization_status = Column(Text, nullable=False, server_default=text("'pending'"))
    diarization_error = Column(Text, nullable=True)

    consultation = relationship("Consultation", back_populates="audio_records")
    transcripts = relationship("Transcript", back_populates="audio_record")
    speaker_segments = relationship("SpeakerSegment", back_populates="audio_record")
