from sqlalchemy import Column, DateTime, ForeignKey, Integer, Text, UniqueConstraint, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.database.base import Base


class SOAPNote(Base):
    __tablename__ = "soap_notes"
    __table_args__ = (
        UniqueConstraint("consultation_id", "version", name="uq_soap_notes_consultation_version"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    consultation_id = Column(
        UUID(as_uuid=True), ForeignKey("consultations.id"), nullable=False, index=True
    )
    version = Column(Integer, nullable=False)
    subjective = Column(Text, nullable=True)
    objective = Column(Text, nullable=True)
    assessment = Column(Text, nullable=True)
    plan = Column(Text, nullable=True)
    status = Column(Text, nullable=False, server_default=text("'generated'"))
    generated_by = Column(Text, nullable=True)
    reviewed_by_doctor_id = Column(UUID(as_uuid=True), ForeignKey("doctors.id"), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    consultation = relationship("Consultation", back_populates="soap_notes")
    reviewed_by_doctor = relationship("Doctor", back_populates="reviewed_soap_notes")
    soap_claims = relationship(
        "SOAPClaim", back_populates="soap_note", order_by="SOAPClaim.sequence_index"
    )
