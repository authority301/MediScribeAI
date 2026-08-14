from sqlalchemy import Column, DateTime, ForeignKey, Numeric, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.database.base import Base


class VerificationResult(Base):
    __tablename__ = "verification_results"

    id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    soap_claim_id = Column(
        UUID(as_uuid=True), ForeignKey("soap_claims.id"), nullable=False, unique=True
    )
    status = Column(Text, nullable=False, index=True)
    confidence_score = Column(Numeric, nullable=False)
    verifier_model = Column(Text, nullable=True)
    verified_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    soap_claim = relationship("SOAPClaim", back_populates="verification_result")
