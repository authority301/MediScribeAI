from sqlalchemy import Column, Text, DateTime, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.database.base import Base


class Doctor(Base):
    __tablename__ = "doctors"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    full_name = Column(Text, nullable=False)
    email = Column(Text, nullable=False, unique=True)
    specialization = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    voice_profiles = relationship("VoiceProfile", back_populates="doctor")
    consultations = relationship("Consultation", back_populates="doctor")
    reviewed_soap_notes = relationship("SOAPNote", back_populates="reviewed_by_doctor")
