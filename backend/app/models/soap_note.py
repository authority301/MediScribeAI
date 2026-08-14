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
    # Step 13: source transcript this note was generated from.
    transcript_id = Column(UUID(as_uuid=True), ForeignKey("transcripts.id"), nullable=True)
    version = Column(Integer, nullable=False)
    subjective = Column(Text, nullable=True)
    objective = Column(Text, nullable=True)
    assessment = Column(Text, nullable=True)
    plan = Column(Text, nullable=True)
    # Doctor review workflow (Step 7 design): generated / edited / approved / rejected.
    status = Column(Text, nullable=False, server_default=text("'generated'"))
    generated_by = Column(Text, nullable=True)
    reviewed_by_doctor_id = Column(UUID(as_uuid=True), ForeignKey("doctors.id"), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Step 13: SOAP generation pipeline status -- a SEPARATE concept from the
    # doctor-review `status` above. pending/processing/completed/failed track
    # whether the generator itself ran successfully; an incomplete note
    # (little evidence in the transcript) is still "completed", never "failed".
    generation_status = Column(Text, nullable=False, server_default=text("'pending'"))
    generation_error = Column(Text, nullable=True)

    # Step 14B: OPERATIONAL status of the most recent NLI verification run
    # (pending/processing/completed/failed) -- deliberately named apart from
    # evidence_links.verification_status, which is the SEMANTIC research
    # outcome (SUPPORTED/CONTRADICTED/UNGROUNDED). Never conflate the two.
    evidence_verification_status = Column(Text, nullable=False, server_default=text("'pending'"))
    evidence_verification_error = Column(Text, nullable=True)

    consultation = relationship("Consultation", back_populates="soap_notes")
    transcript = relationship("Transcript", back_populates="soap_notes")
    reviewed_by_doctor = relationship("Doctor", back_populates="reviewed_soap_notes")
    soap_claims = relationship(
        "SOAPClaim", back_populates="soap_note", order_by="SOAPClaim.sequence_index"
    )
