from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Index, Integer, Numeric, Text, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.database.base import Base


class MedicalEntity(Base):
    __tablename__ = "medical_entities"
    __table_args__ = (
        Index("ix_medical_entities_consultation_type", "consultation_id", "entity_type"),
        Index("ix_medical_entities_transcript_id", "transcript_id"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    consultation_id = Column(UUID(as_uuid=True), ForeignKey("consultations.id"), nullable=False)
    # Step 12: which transcript this entity was extracted from. Reusing this
    # table for both clinical entities (SYMPTOM, MEDICATION, ...) and PHI
    # detections (entity_type prefixed "PHI_") -- see app/medical_entities/.
    transcript_id = Column(UUID(as_uuid=True), ForeignKey("transcripts.id"), nullable=True)
    speaker_segment_id = Column(
        UUID(as_uuid=True), ForeignKey("speaker_segments.id"), nullable=True, index=True
    )
    entity_type = Column(Text, nullable=False)
    # For PHI entity_types, this holds a redaction placeholder (e.g.
    # "[PERSON_NAME]"), never the raw PHI value -- see app/medical_entities/phi.py.
    entity_text = Column(Text, nullable=False)
    normalized_value = Column(Text, nullable=True)
    start_char = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)
    # Heuristic extraction confidence (deterministic rule/pattern match
    # strength) -- NOT clinical certainty and not a calibrated probability.
    confidence_score = Column(Numeric, nullable=True)
    negated = Column(Boolean, nullable=False, server_default=text("false"))
    historical = Column(Boolean, nullable=False, server_default=text("false"))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    consultation = relationship("Consultation", back_populates="medical_entities")
    transcript = relationship("Transcript", back_populates="medical_entities")
    speaker_segment = relationship("SpeakerSegment", back_populates="medical_entities")
