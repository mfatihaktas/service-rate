"""Contains the implementation of the results in [1]

Refs:
[1] Glaz, Joseph, Joseph I. Naus, and Sylvan Wallenstein. "Scan statistics."
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
