"""Pure-Python statistics used by the reports (no numpy/scipy dependency)."""
import math
import random
from collections import Counter
from typing import List, Optional, Sequence, Tuple


def mean(xs: Sequence[float]) -> Optional[float]:
    return round(sum(xs) / len(xs), 4) if xs else None


def stdev(xs: Sequence[float]) -> Optional[float]:
    """Population standard deviation."""
    if not xs:
        return None
    m = sum(xs) / len(xs)
    return round(math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)), 4)


def agreement_rate(a: Sequence, b: Sequence) -> Optional[float]:
    if not a or len(a) != len(b):
        return None
    return round(sum(1 for x, y in zip(a, b) if x == y) / len(a), 4)


def cohen_kappa(a: Sequence, b: Sequence) -> Optional[float]:
    """Cohen's kappa for two raters over any categorical labels. None if undefined (empty input); 1.0 when both rate a single identical class."""
    n = len(a)
    if n == 0 or n != len(b):
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in set(ca) | set(cb))
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return round((po - pe) / (1 - pe), 4)


def kappa_label(k: Optional[float]) -> str:
    """Landis & Koch bands."""
    if k is None:
        return "n/a"
    for lim, name in ((0.0, "poor"), (0.2, "slight"), (0.4, "fair"), (0.6, "moderate"), (0.8, "substantial")):
        if k <= lim:
            return name
    return "almost perfect"


def wilson_ci(k: int, n: int, z: float = 1.96) -> Optional[Tuple[float, float]]:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)


def sign_test_p(wins_a: int, wins_b: int) -> Optional[float]:
    """Exact two-sided binomial sign test on decisive outcomes (ties excluded), H0: P(A beats B) = 0.5."""
    n = wins_a + wins_b
    if n == 0:
        return None
    k = min(wins_a, wins_b)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return round(min(1.0, 2 * p), 4)


def bootstrap_mean_diff_ci(a: Sequence[float], b: Sequence[float], iters: int = 5000, seed: int = 7) -> Optional[Tuple[float, float]]:
    """95% bootstrap CI of mean(a - b) over paired samples (same length)."""
    if not a or len(a) != len(b):
        return None
    rng = random.Random(seed)
    d = [x - y for x, y in zip(a, b)]
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(iters))
    return round(means[int(0.025 * iters)], 4), round(means[int(0.975 * iters) - 1], 4)


def modal_band_share(xs: Sequence[float], width: float = 1.0) -> Optional[float]:
    """Share of scores falling in the single most populated band of the given width (clustering indicator)."""
    if not xs:
        return None
    bins = Counter(math.floor(x / width) for x in xs)
    return round(max(bins.values()) / len(xs), 4)
