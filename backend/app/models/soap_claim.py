from sqlalchemy import Column, DateTime, ForeignKey, Integer, Numeric, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.database.base import Base


class SOAPClaim(Base):
    __tablename__ = "soap_claims"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    soap_note_id = Column(UUID(as_uuid=True), ForeignKey("soap_notes.id"), nullable=False, index=True)
    section = Column(Text, nullable=False)
    claim_text = Column(Text, nullable=False)
    sequence_index = Column(Integer, nullable=False)
    # Step 13: generation/evidence-availability confidence for the deterministic
    # baseline -- NOT clinical correctness, NOT a calibrated probability.
    generation_confidence = Column(Numeric, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    soap_note = relationship("SOAPNote", back_populates="soap_claims")
    evidence_links = relationship("EvidenceLink", back_populates="soap_claim")
    verification_result = relationship(
        "VerificationResult", back_populates="soap_claim", uselist=False
    )
