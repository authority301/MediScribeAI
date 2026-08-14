from sqlalchemy import Column, DateTime, ForeignKey, Index, Integer, Numeric, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.database.base import Base


class MedicalEntity(Base):
    __tablename__ = "medical_entities"
    __table_args__ = (
        Index("ix_medical_entities_consultation_type", "consultation_id", "entity_type"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    consultation_id = Column(UUID(as_uuid=True), ForeignKey("consultations.id"), nullable=False)
    speaker_segment_id = Column(
        UUID(as_uuid=True), ForeignKey("speaker_segments.id"), nullable=True, index=True
    )
    entity_type = Column(Text, nullable=False)
    entity_text = Column(Text, nullable=False)
    normalized_value = Column(Text, nullable=True)
    start_char = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)
    confidence_score = Column(Numeric, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    consultation = relationship("Consultation", back_populates="medical_entities")
    speaker_segment = relationship("SpeakerSegment", back_populates="medical_entities")
