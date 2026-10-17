import re
from typing import List, Optional
from ..schema import CriterionVerdict

class SycophancyMitigator:
    """
    Mitigates sycophancy and superficial style bias by:
    1. Forcing the judge to provide direct evidence citations from the text.
    2. Verifying that cited evidence actually exists verbatim in the candidate output.
    3. Neutralizing flattering or sycophantic rhetoric that lacks factual grounding.
    """

    EVIDENCE_GROUNDING_INSTRUCTION = (
        "CRITICAL INSTRUCTION - COMBAT SYCOPHANCY & STYLE BIAS:\n"
        "- Do NOT be swayed by confident, eloquent, or flatteringly polite language.\n"
        "- For any score >= 4.0, you MUST provide an 'evidence_quote' citing a verbatim substring "
        "from the model output that directly proves the claim.\n"
        "- If an answer sounds authoritative but provides false facts or fabricated specifications, "
        "you MUST fail it regardless of how elegant its writing style appears."
    )

    @staticmethod
    def verify_evidence_grounding(verdicts: List[CriterionVerdict], candidate_text: str) -> float:
        """
        Verifies whether evidence quotes actually appear verbatim in the candidate text.
        If a judge assigned a high score (>=4.0) with an ungrounded or hallucinated quote,
        returns an adjustment penalty.
        """
        penalty = 0.0
        candidate_clean = " ".join(candidate_text.lower().split())

        for v in verdicts:
            if v.score >= 4.0:
                if v.evidence_quote:
                    quote_clean = " ".join(v.evidence_quote.lower().split())
                    # Check if substring exists
                    if quote_clean not in candidate_clean:
                        # The quote was hallucinated or altered by the judge
                        penalty += 0.2
                else:
                    # High score awarded without evidence quote
                    penalty += 0.1

        return round(min(penalty, 0.8), 2)
