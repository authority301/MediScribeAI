from sqlalchemy import Column, Text, DateTime, ForeignKey, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.database.base import Base


class Consultation(Base):
    __tablename__ = "consultations"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    doctor_id = Column(UUID(as_uuid=True), ForeignKey("doctors.id"), nullable=False, index=True)
    patient_reference = Column(Text, nullable=True)
    consultation_date = Column(DateTime(timezone=True), nullable=False)
    language_mode = Column(Text, nullable=True)
    status = Column(Text, nullable=False, server_default=text("'scheduled'"), index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    doctor = relationship("Doctor", back_populates="consultations")
    audio_records = relationship("AudioRecord", back_populates="consultation")
    transcripts = relationship("Transcript", back_populates="consultation")
    medical_entities = relationship("MedicalEntity", back_populates="consultation")
    soap_notes = relationship("SOAPNote", back_populates="consultation")
