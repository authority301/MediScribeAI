from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, Numeric, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.database.base import Base


class SpeakerSegment(Base):
    __tablename__ = "speaker_segments"
    __table_args__ = (
        Index("ix_speaker_segments_transcript_sequence", "transcript_id", "sequence_index"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    transcript_id = Column(UUID(as_uuid=True), ForeignKey("transcripts.id"), nullable=False)
    sequence_index = Column(Integer, nullable=False)
    speaker_label = Column(Text, nullable=False)
    inferred_role = Column(Text, nullable=True)
    start_time_ms = Column(Integer, nullable=False)
    end_time_ms = Column(Integer, nullable=False)
    segment_text = Column(Text, nullable=False)
    diarization_confidence = Column(Numeric, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    transcript = relationship("Transcript", back_populates="speaker_segments")
    medical_entities = relationship("MedicalEntity", back_populates="speaker_segment")
    evidence_links = relationship("EvidenceLink", back_populates="speaker_segment")
