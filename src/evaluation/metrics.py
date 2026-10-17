import math
from typing import List, Tuple, Dict, Any

class StatisticalMetrics:
    """
    Pure-Python implementation of core validation & IR metrics.
    Zero external C-extension or binary compilation requirements.
    """

    @staticmethod
    def agreement_rate(preds: List[bool], golds: List[bool]) -> float:
        if not preds or len(preds) != len(golds):
            return 0.0
        matches = sum(1 for p, g in zip(preds, golds) if p == g)
        return round(matches / len(preds), 4)

    @staticmethod
    def cohens_kappa(preds: List[bool], golds: List[bool]) -> float:
        """
        Calculates Cohen's Kappa coefficient (inter-rater agreement for binary pass/fail).
        kappa = (P_observed - P_expected) / (1 - P_expected)
        """
        if not preds or len(preds) != len(golds):
            return 0.0
        n = len(preds)

        # Contingency table:
        # a: pred=True, gold=True
        # b: pred=True, gold=False
        # c: pred=False, gold=True
        # d: pred=False, gold=False
        a = sum(1 for p, g in zip(preds, golds) if p is True and g is True)
        b = sum(1 for p, g in zip(preds, golds) if p is True and g is False)
        c = sum(1 for p, g in zip(preds, golds) if p is False and g is True)
        d = sum(1 for p, g in zip(preds, golds) if p is False and g is False)

        p_observed = (a + d) / n

        p_pred_true = (a + b) / n
        p_pred_false = (c + d) / n
        p_gold_true = (a + c) / n
        p_gold_false = (b + d) / n

        p_expected = (p_pred_true * p_gold_true) + (p_pred_false * p_gold_false)

        if math.isclose(1.0, p_expected):
            return 1.0

        kappa = (p_observed - p_expected) / (1.0 - p_expected)
        return round(kappa, 4)

    @staticmethod
    def pearson_correlation(x: List[float], y: List[float]) -> float:
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        n = len(x)
        mean_x = sum(x) / n
        mean_y = sum(y) / n

        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        den_x = sum((xi - mean_x) ** 2 for xi in x)
        den_y = sum((yi - mean_y) ** 2 for yi in y)

        if den_x == 0 or den_y == 0:
            return 0.0

        r = num / (math.sqrt(den_x) * math.sqrt(den_y))
        return round(r, 4)

    @staticmethod
    def _rank(arr: List[float]) -> List[float]:
        # Handle ties with average ranks
        indexed = sorted(enumerate(arr), key=lambda x: x[1])
        ranks = [0.0] * len(arr)
        i = 0
        while i < len(indexed):
            val = indexed[i][1]
            j = i
            while j < len(indexed) and math.isclose(indexed[j][1], val):
                j += 1
            avg_rank = (i + j - 1) / 2.0 + 1.0
            for k in range(i, j):
                ranks[indexed[k][0]] = avg_rank
            i = j
        return ranks

    @classmethod
    def spearman_rank_correlation(cls, x: List[float], y: List[float]) -> float:
        if len(x) != len(y) or len(x) < 2:
            return 0.0
        rank_x = cls._rank(x)
        rank_y = cls._rank(y)
        return cls.pearson_correlation(rank_x, rank_y)

    @staticmethod
    def mean_absolute_error(preds: List[float], golds: List[float]) -> float:
        if not preds or len(preds) != len(golds):
            return 0.0
        return round(sum(abs(p - g) for p, g in zip(preds, golds)) / len(preds), 4)

    @staticmethod
    def root_mean_squared_error(preds: List[float], golds: List[float]) -> float:
        if not preds or len(preds) != len(golds):
            return 0.0
        mse = sum((p - g) ** 2 for p, g in zip(preds, golds)) / len(preds)
        return round(math.sqrt(mse), 4)

    @staticmethod
    def test_retest_consistency(run1: List[Any], run2: List[Any]) -> float:
        if not run1 or len(run1) != len(run2):
            return 0.0
        consistent = sum(1 for r1, r2 in zip(run1, run2) if r1 == r2)
        return round(consistent / len(run1), 4)
