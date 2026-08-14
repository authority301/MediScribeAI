from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Index, Text, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.database.base import Base


class Transcript(Base):
    __tablename__ = "transcripts"
    __table_args__ = (
        Index(
            "ix_transcripts_final_consultation",
            "consultation_id",
            postgresql_where=text("is_final = true"),
        ),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    consultation_id = Column(
        UUID(as_uuid=True), ForeignKey("consultations.id"), nullable=False, index=True
    )
    audio_record_id = Column(UUID(as_uuid=True), ForeignKey("audio_records.id"), nullable=True)
    full_text = Column(Text, nullable=False)
    asr_model = Column(Text, nullable=True)
    is_final = Column(Boolean, nullable=False, server_default=text("false"))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    consultation = relationship("Consultation", back_populates="transcripts")
    audio_record = relationship("AudioRecord", back_populates="transcripts")
    speaker_segments = relationship(
        "SpeakerSegment", back_populates="transcript", order_by="SpeakerSegment.sequence_index"
    )
