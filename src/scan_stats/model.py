"""Contains the implementation of the results in [1]

Refs:
[1] Glaz, Joseph, Joseph I. Naus, and Sylvan Wallenstein. "Scan statistics.", 2000
[2] Naus, Joseph I. "Approximations for distributions of scan statistics.", 1982.
"""

import math
import scipy


from src.utils.debug import *


def H(theta: float, p: float) -> float:
    check(0 < theta < 1 and 0 < p < 1, "", theta=theta, p=p)

    return (
        theta * math.log(theta / p)
        + (1 - theta) * math.log((1 - theta) / (1 - p))
    )


def h(theta: float, p: float) -> float:
    check(0 < theta < 1 and 0 < p < 1, "")

    return math.log((theta * (1 - p)) / (p * (1 - theta)))


def scan_stats_approx_1(n: int, p: float, k: int, r: int):
    """Implements the approximation presented at the end of page 67 of [1],
    which assumes p < r/k != 1.
    """

    log(DEBUG, "Started", n=n, p=p, k=k, r=r)

    r_over_k = r / k
    check(r_over_k > p, "Approximation requires r / k > p", r_over_k=r_over_k, p=p)

    b_r_k_p = scipy.stats.binom.pmf(r, k, p)
    lambda_ = (n - k + 1) * (r_over_k - p) * b_r_k_p

    return math.exp(-lambda_)


def scan_stats_approx_2(n: int, p: float, k: int, r: int):
    """Implements the approximation presented at the end of page 69 of [1],
    which assumes p < r/k != 1.
    """

    log(DEBUG, "Started", n=n, p=p, k=k, r=r)

    r_over_k = r / k
    check(r_over_k > p, "Approximation requires r / k > p", r_over_k=r_over_k, p=p)

    H_value = H(r_over_k, p)
    # h_value = h(r_over_k, p)

    numerator = (r - k * p) * math.exp(-k * H_value)  # - ro * h_value
    denominator = math.sqrt(2 * math.pi * r * k * (k - r))

    return math.exp(-n * numerator / denominator)


def scan_stats_cdf_approx_by_naus(n: int, m: int, p: float, k: int):
    """
    - Theorem 2 in Section 4 in [2].
    - (8.25) on page 185 in [1].

    Pr{max of `m`-size scans over `n` Bernoulli trials <= k}.
    """

    def Q_l(l: int, n: int, m: int, p: float, k: int) -> float:
        b_k = scipy.stats.binom.pmf(k, m, p)

        if l == 2:
            return (
                scipy.stats.binom.cdf(k - 1, m, p) ** 2
                - (k - 1) * b_k * scipy.stats.binom.cdf(k - 2, m, p)
                + m * p * b_k * scipy.stats.binom.cdf(k - 3, m - 1, p)
            )

        elif l == 3:
            A1 = (
                2 * b_k * scipy.stats.binom.cdf(k - 1, m, p)
                * (
                    (k - 1) * scipy.stats.binom.cdf(k - 2, m, p)
                    - m * p * scipy.stats.binom.cdf(k - 3, m - 1, p)
                )
            )

            A2 = (
                0.5 * (b_k ** 2)
                * (
                    (k - 1) * (k - 2) * scipy.stats.binom.cdf(k - 3, m, p)
                    - 2 * (k - 2) * m * p * scipy.stats.binom.cdf(k - 4, m - 1, p)
                    + m * (m - 1) * (p ** 2) * scipy.stats.binom.cdf(k - 5, m - 2, p)
                )
            )

            A3 = sum(
                scipy.stats.binom.pmf(2 * k - r, m, p) * scipy.stats.binom.cdf(r - 1, m, p) ** 2
                for r in range(1, k)
            )

            A4 = sum(
                scipy.stats.binom.pmf(2 * k - r, m, p) * scipy.stats.binom.pmf(r, m, p)
                * (
                    (r - 1) * scipy.stats.binom.cdf(r - 2, m, p)
                    - m * p * scipy.stats.binom.cdf(r - 3, m - 1, p)
                )
                for r in range(2, k)
            )

            return (
                scipy.stats.binom.cdf(k - 1, m, p) ** 3 - A1 + A2 + A3 - A4
            )

    L = n // m
    Q_2 = Q_l(l=2, n=n, m=m, p=p, k=k + 1)
    Q_3 = Q_l(l=3, n=n, m=m, p=p, k=k + 1)
    return Q_2 * (Q_3 / Q_2) ** (L - 2)
