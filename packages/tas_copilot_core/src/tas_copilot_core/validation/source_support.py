from __future__ import annotations

from tas_copilot_contracts import Claim


def require_claim_support(claim: Claim) -> tuple[bool, str | None]:
    if not claim.support:
        return False, "claim has no provenance support"
    return True, None

