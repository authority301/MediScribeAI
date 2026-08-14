from sqlalchemy import Column, Text, DateTime, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.database.base import Base


class VoiceProfile(Base):
    __tablename__ = "voice_profiles"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    doctor_id = Column(UUID(as_uuid=True), ForeignKey("doctors.id"), nullable=False, index=True)
    embedding_reference = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    doctor = relationship("Doctor", back_populates="voice_profiles")
