"""Contains the formulas from
- Saperstein, Bernard. "The generalized birthday problem.", 1972.
"""

import mpmath as mp

from src.utils.debug import *


mp.dps = 100


def _I_(k: int, n: int, m: int, c: int, e: int, a: int) -> float:
    sigma = sum(
        mp.binomial(n - m + k - a, k - (a - c - e) - i) * mp.binomial(a - k, i)
        for i in range(max(0, c - (n - m - e)), a - k + 1)
    )

    return (
        mp.binomial(n - m, c) * mp.binomial(n - m, e)
        - mp.binomial(n - m, a - k) * sigma
    )


def prob_kstar_leq_k(k: int, n: int, m: int, a: int) -> float:
    log(DEBUG, "", k=k, n=n, m=m, a=a)

    if m >= n / 2:
        if k <= (a + 1) / 2:
            return 0

        return (
            1 / mp.binomial(n, a)
            * sum(
                mp.binomial(2 * m - n, a - c - e) * _I_(k, n, m, c, e, a)
                for c in range(a - k + 1, min(k - 1, n - m) + 1)
                for e in range(a - k + 1, min(a - c, n - m) + 1)
            )
        )

    # m < n / 2
    if k <= a / 2:
        return 0

    A = (
        sum(
            mp.binomial(s, k - 1) * mp.binomial(n - s - 1, a - k)
            for s in range(k - 1, m)
        )
        + sum(
            mp.binomial(m - 1, k - 1) * mp.binomial(n - s - 1, a - k)
            for s in range(m, n - a + k)
        )
    )

    B = sum(
        (
            mp.binomial(n - x, a - r) * mp.binomial(x - 2 * m, r - k - y)
            * _I_(k, 2 * (m - 1), m - 1, k - 1, y, k + y - 1)
        )
        for r in range(k + 1, a + 1)
        for x in range(2 * m, n - a + r + 1)
        for y in range(max(0, r - k - x + 2 * m), r - k + 1)
    )

    C = sum(
        (
            mp.binomial(n - x, a - r) * mp.binomial(2 * m - x, k - y - 1)
            * _I_(k, x - 2, m - 1, y, r - k, r - 1)
        )
        for r in range(k + 1, a + 1)
        for x in range(r - k + m + 1, min(2 * m - 1, n - a + r) + 1)
        for y in range(max(r - k, k + x - 2 * m - 1), min(k - 1, x - m - 1) + 1)
    )

    return 1 - 1 / mp.binomial(n, a) * (A + B + C)
