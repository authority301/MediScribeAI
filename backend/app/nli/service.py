from sqlalchemy.orm import Session

from app.models import Consultation, EvidenceLink, SOAPClaim, SOAPNote, SpeakerSegment
from app.nli.aggregation import EvidenceNliResult, aggregate_claim_verification
from app.nli.config import NLI_CONTRADICTION_THRESHOLD, NLI_ENTAILMENT_THRESHOLD
from app.nli.model import run_nli

# Per-EVIDENCE-LINK classification, distinct from the claim-level aggregate
# stored in verification_status. Never the sole research result on its own.
RELATIONSHIP_TYPE_BY_LABEL = {
    "ENTAILMENT": "supports",
    "CONTRADICTION": "contradicts",
    "NEUTRAL": "insufficient",
}


class EvidenceVerificationError(Exception):
    """Carries the HTTP status/detail the route should surface for a verification failure."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def _mark_failed(soap_note: SOAPNote, db: Session, error_message: str) -> None:
    soap_note.evidence_verification_status = "failed"
    soap_note.evidence_verification_error = error_message
    db.commit()


def run_evidence_verification(consultation: Consultation, soap_note: SOAPNote, db: Session) -> dict:
    """Run local NLI verification over every Step 14A evidence link for one
    SOAP note's claims. Never runs ASR, diarization, role identification,
    entity extraction, SOAP generation, or Step 14A retrieval itself -- all
    must already have completed, and evidence links must already exist.

    IDEMPOTENCY (documented choice): existing evidence_links are UPDATED in
    place (NLI fields + relationship_type only) -- never deleted or
    duplicated. alignment_score (Step 14A) is never touched.
    """
    if soap_note.generation_status != "completed":
        raise EvidenceVerificationError(
            409, "SOAP note generation must be completed before evidence verification"
        )

    claims = (
        db.query(SOAPClaim)
        .filter(SOAPClaim.soap_note_id == soap_note.id)
        .order_by(SOAPClaim.section, SOAPClaim.sequence_index)
        .all()
    )
    if not claims:
        raise EvidenceVerificationError(409, "No SOAP claims exist for this note")

    claim_ids = [c.id for c in claims]
    links = db.query(EvidenceLink).filter(EvidenceLink.soap_claim_id.in_(claim_ids)).all()
    if not links:
        raise EvidenceVerificationError(
            409,
            "No evidence links found; run evidence retrieval (Step 14A) before verification",
        )

    soap_note.evidence_verification_status = "processing"
    db.commit()

    try:
        links_by_claim: dict = {}
        for link in links:
            links_by_claim.setdefault(link.soap_claim_id, []).append(link)

        segment_cache: dict = {}
        supported_count = 0
        contradicted_count = 0
        ungrounded_count = 0
        per_claim = []

        for claim in claims:
            claim_links = links_by_claim.get(claim.id, [])
            if not claim_links:
                continue

            evidence_results: list[EvidenceNliResult] = []
            for link in claim_links:
                segment = segment_cache.get(link.speaker_segment_id)
                if segment is None:
                    segment = (
                        db.query(SpeakerSegment)
                        .filter(SpeakerSegment.id == link.speaker_segment_id)
                        .first()
                    )
                    segment_cache[link.speaker_segment_id] = segment
                # Premise is ALWAYS the transcript evidence segment; hypothesis
                # is ALWAYS the SOAP claim. Never reversed. The original
                # (possibly negated) segment text is passed as-is -- never
                # filtered or rewritten based on Step 12 negation metadata.
                premise = segment.segment_text if segment and segment.segment_text else ""

                result = run_nli(premise=premise, hypothesis=claim.claim_text)

                link.entailment_score = result.entailment
                link.contradiction_score = result.contradiction
                link.neutral_score = result.neutral
                link.nli_label = result.label
                link.relationship_type = RELATIONSHIP_TYPE_BY_LABEL[result.label]

                evidence_results.append(
                    EvidenceNliResult(
                        speaker_segment_id=link.speaker_segment_id,
                        nli_label=result.label,
                        entailment_score=result.entailment,
                        contradiction_score=result.contradiction,
                        neutral_score=result.neutral,
                    )
                )

            claim_status = aggregate_claim_verification(
                evidence_results, NLI_ENTAILMENT_THRESHOLD, NLI_CONTRADICTION_THRESHOLD
            )
            for link in claim_links:
                link.verification_status = claim_status

            if claim_status == "SUPPORTED":
                supported_count += 1
            elif claim_status == "CONTRADICTED":
                contradicted_count += 1
            else:
                ungrounded_count += 1

            per_claim.append(
                {
                    "claim_id": claim.id,
                    "claim_text": claim.claim_text,
                    "section": claim.section,
                    "verification_status": claim_status,
                    "links": claim_links,
                }
            )

        soap_note.evidence_verification_status = "completed"
        soap_note.evidence_verification_error = None
        db.commit()

        return {
            "claims_processed": len(claims),
            "evidence_links_verified": len(links),
            "supported": supported_count,
            "contradicted": contradicted_count,
            "ungrounded": ungrounded_count,
            "per_claim": per_claim,
        }
    except EvidenceVerificationError:
        raise
    except Exception as exc:
        db.rollback()
        try:
            _mark_failed(soap_note, db, "Unexpected evidence verification failure")
        except Exception:
            db.rollback()
        raise EvidenceVerificationError(500, "Evidence verification failed unexpectedly.") from exc
