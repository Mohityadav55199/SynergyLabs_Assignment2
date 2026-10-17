from typing import Literal, Tuple
from ..schema import SinglePairEvaluation

class PositionBiasMitigator:
    """
    Mitigates position bias (order effect) by executing pairwise evaluations
    in both forward (A, B) and reverse (B, A) orders.
    Calculates flip rate and determines final consensus verdict.
    """

    @staticmethod
    def map_reverse_winner(reverse_winner: Literal["A", "B", "TIE"]) -> Literal["A", "B", "TIE"]:
        """
        In the reverse evaluation, candidate A was actually the original candidate B,
        and candidate B was the original candidate A.
        Therefore:
        - If reverse judge chose "A", the original candidate B won.
        - If reverse judge chose "B", the original candidate A won.
        - If "TIE", remains "TIE".
        """
        if reverse_winner == "A":
            return "B"
        elif reverse_winner == "B":
            return "A"
        return "TIE"

    @classmethod
    def resolve_verdict(
        cls,
        forward_eval: SinglePairEvaluation,
        reverse_eval: SinglePairEvaluation,
        strategy: Literal["strict_agreement", "score_average"] = "score_average"
    ) -> Tuple[Literal["A", "B", "TIE"], bool, bool, float, float, str]:
        """
        Returns:
            (final_winner, position_agreement, flip_detected, avg_score_a, avg_score_b, rationale)
        """
        forward_winner = forward_eval.winner
        # Reverse winner mapped back to original identities
        mapped_reverse_winner = cls.map_reverse_winner(reverse_eval.winner)

        # In reverse eval:
        # reverse_eval.score_a belongs to original B!
        # reverse_eval.score_b belongs to original A!
        avg_score_a = round((forward_eval.score_a + reverse_eval.score_b) / 2.0, 2)
        avg_score_b = round((forward_eval.score_b + reverse_eval.score_a) / 2.0, 2)

        # Agreement check:
        # A flip occurs if the judge picked one candidate in forward, but the opposite in reverse
        flip_detected = False
        if forward_winner != "TIE" and mapped_reverse_winner != "TIE":
            if forward_winner != mapped_reverse_winner:
                flip_detected = True

        agreement = (forward_winner == mapped_reverse_winner)

        if strategy == "strict_agreement":
            if agreement:
                final_winner = forward_winner
                rationale = f"Bidirectional agreement achieved: both forward and reverse orders favored {final_winner}."
            else:
                final_winner = "TIE"
                rationale = (
                    f"Order-dependent flip detected: Forward order favored {forward_winner}, but reverse "
                    f"order favored {mapped_reverse_winner}. Declared TIE to neutralize position bias."
                )
        else: # score_average
            score_diff = avg_score_a - avg_score_b
            if abs(score_diff) < 0.25:
                final_winner = "TIE"
                rationale = f"Averaged scores across both presentation orders are within margin ({avg_score_a} vs {avg_score_b}). Declared TIE."
            elif score_diff > 0:
                final_winner = "A"
                rationale = f"Across both presentation orders, Candidate A achieved higher averaged score ({avg_score_a} vs {avg_score_b})."
            else:
                final_winner = "B"
                rationale = f"Across both presentation orders, Candidate B achieved higher averaged score ({avg_score_b} vs {avg_score_a})."

        return final_winner, agreement, flip_detected, avg_score_a, avg_score_b, rationale
