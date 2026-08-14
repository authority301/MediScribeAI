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
    # Nullable: Step 10 diarization can produce a speaker interval with no
    # aligned ASR transcript yet (or ASR hasn't run at all for this audio).
    transcript_id = Column(UUID(as_uuid=True), ForeignKey("transcripts.id"), nullable=True)
    # Diarization operates on the audio directly, independent of any transcript.
    audio_record_id = Column(UUID(as_uuid=True), ForeignKey("audio_records.id"), nullable=True)
    sequence_index = Column(Integer, nullable=False)
    # Nullable: Step 9A writes raw ASR segments with no speaker identity yet.
    # Diarization (Step 10) is what will populate this -- never fabricated here.
    speaker_label = Column(Text, nullable=True)
    inferred_role = Column(Text, nullable=True)
    start_time_ms = Column(Integer, nullable=False)
    end_time_ms = Column(Integer, nullable=False)
    # Nullable: a pure diarization interval may have no confidently-aligned
    # ASR text (weak/no temporal overlap) -- left empty rather than fabricated.
    segment_text = Column(Text, nullable=True)
    diarization_confidence = Column(Numeric, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    transcript = relationship("Transcript", back_populates="speaker_segments")
    audio_record = relationship("AudioRecord", back_populates="speaker_segments")
    medical_entities = relationship("MedicalEntity", back_populates="speaker_segment")
    evidence_links = relationship("EvidenceLink", back_populates="speaker_segment")
