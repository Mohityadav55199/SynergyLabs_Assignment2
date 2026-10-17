from typing import Dict, Any

class ScoreCalibrator:
    """
    Combats score clustering / compression by injecting calibrated few-shot anchor exemplars.
    Prevents judges from clustering all scores in the narrow 3.5 - 4.5 band.
    """

    FEW_SHOT_ANCHOR_EXAMPLES = (
        "CALIBRATED SCORE SCALE (Use the FULL 1.0 to 5.0 range):\n"
        "• Score 1.0 - Catastrophic Failure:\n"
        "  Example: 'Tuples prevent mutating lists inside them via ImmutableMutationError.' (Pure hallucination)\n"
        "• Score 3.0 - Mediocre / Partial Quality:\n"
        "  Example: 'QuickSort is O(n log n). Dual pivot exists and uses two pivots.' (Factually true but superficial, misses worst case and details)\n"
        "• Score 5.0 - Exceptional Gold Standard:\n"
        "  Example: 'Average time O(n log n), auxiliary space O(log n); worst case O(n^2), space O(n). Dual-pivot keeps same asymptotic bounds while reducing comparisons by ~20% and improving cache locality.' (Completely accurate, rigorous, crisp)"
    )

    @classmethod
    def get_calibration_prompt(cls) -> str:
        return cls.FEW_SHOT_ANCHOR_EXAMPLES
