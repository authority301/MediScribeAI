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
    # Step 14A default: "candidate" (unclassified retrieval). Step 14B updates
    # this PER-LINK to supports/contradicts/insufficient based on that link's
    # own NLI label -- it is never the sole research result on its own.
    relationship_type = Column(Text, nullable=False)
    # Step 14A retrieval score -- preserved, never overwritten by Step 14B.
    alignment_score = Column(Numeric, nullable=True)
    evidence_snippet = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    # Step 14B: raw per-evidence-link NLI probabilities (premise = this
    # segment's text, hypothesis = the SOAP claim text -- never reversed).
    # NOT clinical certainty; see app/nli/model.py and aggregation.py.
    entailment_score = Column(Numeric, nullable=True)
    contradiction_score = Column(Numeric, nullable=True)
    neutral_score = Column(Numeric, nullable=True)
    nli_label = Column(Text, nullable=True)  # ENTAILMENT / CONTRADICTION / NEUTRAL (this link only)
    # CLAIM-LEVEL aggregated research outcome (SUPPORTED/CONTRADICTED/
    # UNGROUNDED), denormalized onto every evidence_link belonging to that
    # claim so it's queryable here without a join to a separate table.
    verification_status = Column(Text, nullable=True)

    soap_claim = relationship("SOAPClaim", back_populates="evidence_links")
    speaker_segment = relationship("SpeakerSegment", back_populates="evidence_links")
