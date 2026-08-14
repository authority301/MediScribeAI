from sqlalchemy.orm import Session

from app.evidence.config import EVIDENCE_MIN_SCORE, EVIDENCE_TOP_K
from app.evidence.retrieval import retrieve_evidence
from app.models import Consultation, EvidenceLink, SOAPClaim, SOAPNote, SpeakerSegment, Transcript

# Step 14A never classifies supports/contradicts/insufficient -- that
# judgement (NLI verification) is Step 14B. "candidate" honestly marks these
# rows as unclassified retrieval results, never a verdict on claim truth.
CANDIDATE_RELATIONSHIP_TYPE = "candidate"


class EvidenceRetrievalError(Exception):
    """Carries the HTTP status/detail the route should surface for a retrieval failure."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def run_evidence_retrieval(consultation: Consultation, soap_note: SOAPNote, db: Session) -> dict:
    """Retrieve candidate transcript evidence for every claim on one SOAP note
    and persist the results as evidence_links. Never runs ASR, diarization,
    role identification, entity extraction, or SOAP generation itself -- all
    must already have completed.

    IDEMPOTENCY (documented choice): rebuild strategy -- all existing
    evidence_links for this note's claims are deleted before the fresh
    retrieval result is inserted, so repeated calls are deterministic and
    never accumulate duplicates.
    """
    if soap_note.generation_status != "completed":
        raise EvidenceRetrievalError(
            409, "SOAP note generation must be completed before evidence retrieval"
        )

    claims = (
        db.query(SOAPClaim)
        .filter(SOAPClaim.soap_note_id == soap_note.id)
        .order_by(SOAPClaim.section, SOAPClaim.sequence_index)
        .all()
    )

    if claims:
        claim_ids = [c.id for c in claims]
        db.query(EvidenceLink).filter(EvidenceLink.soap_claim_id.in_(claim_ids)).delete(
            synchronize_session=False
        )
        db.commit()

    if not claims:
        # A SOAP note with no claims is a safe, completed zero-result -- not
        # a failure or a precondition violation.
        return {"claims_processed": 0, "evidence_links_created": 0, "per_claim": []}

    try:
        segments = (
            db.query(SpeakerSegment)
            .join(Transcript, SpeakerSegment.transcript_id == Transcript.id)
            .filter(Transcript.consultation_id == consultation.id)
            .order_by(SpeakerSegment.sequence_index)
            .all()
        )

        total_links = 0
        per_claim = []
        for claim in claims:
            candidates = retrieve_evidence(
                claim_section=claim.section,
                claim_text=claim.claim_text,
                segments=segments,
                top_k=EVIDENCE_TOP_K,
                min_score=EVIDENCE_MIN_SCORE,
            )
            for candidate in candidates:
                db.add(
                    EvidenceLink(
                        soap_claim_id=claim.id,
                        speaker_segment_id=candidate.speaker_segment_id,
                        relationship_type=CANDIDATE_RELATIONSHIP_TYPE,
                        alignment_score=candidate.retrieval_score,
                    )
                )
            total_links += len(candidates)
            per_claim.append({"claim_id": claim.id, "evidence": candidates})

        db.commit()
        return {
            "claims_processed": len(claims),
            "evidence_links_created": total_links,
            "per_claim": per_claim,
        }
    except Exception as exc:
        db.rollback()
        raise EvidenceRetrievalError(500, "Evidence retrieval failed unexpectedly.") from exc
