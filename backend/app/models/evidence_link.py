from sqlalchemy import Column, DateTime, ForeignKey, Numeric, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.database.base import Base


class EvidenceLink(Base):
    __tablename__ = "evidence_links"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    soap_claim_id = Column(UUID(as_uuid=True), ForeignKey("soap_claims.id"), nullable=False, index=True)
    speaker_segment_id = Column(
        UUID(as_uuid=True), ForeignKey("speaker_segments.id"), nullable=False, index=True
    )
    relationship_type = Column(Text, nullable=False)
    alignment_score = Column(Numeric, nullable=True)
    evidence_snippet = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    soap_claim = relationship("SOAPClaim", back_populates="evidence_links")
    speaker_segment = relationship("SpeakerSegment", back_populates="evidence_links")
