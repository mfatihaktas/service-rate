import functools
import math
import scipy

from mpmath import mp

from src.utils.debug import *


mp.dps = 100


def prob_num_cells_w_r_particles_eq_c(n: int, m: int, d: int, r: int, c: int) -> float:
    """Denoted as `Pr{mu_r'(n, N) = c}` in [1]."""

    if r == 0:
        return prob_num_empty_cells_eq_c(n=n, m=m, d=d, c=c)

    else:
        assert_("Defined only for r=0", r=r)


def prob_num_empty_cells_eq_c(n: int, m: int, d: int, c: int) -> float:
    """Denoted as `Pr{mu_0'(n, N) = c}` in [1]."""
    # log(DEBUG, "Started", n=n, m=m, d=d, c=c)

    if n <= d:
        if c == 0:
            return 1
        else:
            return 0

    n_choose_d = scipy.special.comb(n, d)
    # log(DEBUG, "", n_choose_d=n_choose_d)

    if c == 0:
        s = 0
        for l in range(n + 1):
            # n_choose_l = scipy.special.comb(n, l)
            # n_minus_l_choose_d = scipy.special.comb(n - l, d)
            # s += (-1)**l * n_choose_l * n_minus_l_choose_d**m

            n_choose_l = scipy.special.comb(n, l)
            n_minus_l_choose_d = scipy.special.comb(n - l, d)
            term = (-1)**l * n_choose_l * (n_minus_l_choose_d / n_choose_d)**m
            # log(DEBUG, "", term=term)
            s += term

        # coeff = n_choose_d**(-m)
        # return coeff * s

        return s

    n_choose_c = scipy.special.comb(n, c)
    n_minus_c_choose_d = scipy.special.comb(n - c, d)
    # log(DEBUG, "", n_choose_c=n_choose_c, n_minus_c_choose_d=n_minus_c_choose_d)

    coeff = n_choose_c * (n_minus_c_choose_d / n_choose_d)**m
    return coeff * prob_num_empty_cells_eq_c(n=n - c, m=m, d=d, c=0)


def prob_num_empty_cells_eq_c_w_mpmath(n: int, m: int, d: int, c: int) -> float:
    """Denoted as `Pr{mu_0'(n, N) = c}` in [1]."""
    # log(DEBUG, "Started", n=n, m=m, d=d, c=c)

    num_nonempty_cells = n - c
    if not (d <= num_nonempty_cells <= m * d):
        return 0

    if n <= d:
        if c == 0:
            return 1
        else:
            return 0

    n_ = mp.mpf(f"{n}")
    d_ = mp.mpf(f"{d}")
    c_ = mp.mpf(f"{c}")

    n_choose_d = mp.binomial(n_, d_)
    # log(DEBUG, "", n_choose_d=n_choose_d)

    if c == 0:
        s = 0
        for l in range(n + 1):
            # n_choose_l = scipy.special.comb(n, l)
            # n_minus_l_choose_d = scipy.special.comb(n - l, d)
            # s += (-1)**l * n_choose_l * n_minus_l_choose_d**m

            n_choose_l = mp.binomial(n_, l)
            n_minus_l_choose_d = mp.binomial(n_ - l, d_)
            ratio = n_minus_l_choose_d / n_choose_d
            term = mp.power(mp.mpf("-1"), l) * n_choose_l * mp.power(ratio, m)
            # log(INFO, "", term=term)
            s += term

        # coeff = n_choose_d**(-m)
        # return coeff * s

        return s

    n_choose_c = mp.binomial(n_, c_)
    n_minus_c_choose_d = mp.binomial(n_ - c_, d_)
    # log(DEBUG, "", n_choose_c=n_choose_c, n_minus_c_choose_d=n_minus_c_choose_d)

    ratio = n_minus_c_choose_d / n_choose_d
    coeff = n_choose_c * mp.power(ratio, m)
    prob_c_0 = prob_num_empty_cells_eq_c_w_mpmath(n=n - c, m=m, d=d, c=0)
    # log(WARNING, "", coeff=coeff, prob_c_0=prob_c_0)

    return coeff * prob_c_0


def prob_num_empty_cells_eq_c_w_mpmath_deprecated(n: int, m: int, d: int, c: int) -> float:
    """Denoted as `Pr{mu_0'(n, N) = c}` in [1]."""
    # log(DEBUG, "Started", n=n, m=m, d=d, c=c)

    num_nonempty_cells = n - c
    if not (d <= num_nonempty_cells <= m * d):
        return 0

    if n <= d:
        if c == 0:
            return 1
        else:
            return 0

    n_ = mp.mpf(f"{n}")
    m_ = mp.mpf(f"{m}")
    d_ = mp.mpf(f"{d}")
    c_ = mp.mpf(f"{c}")

    n_minus_c_choose_d_over_n_choose_d = math.prod(
        [
            (n_ - d_ - i) / (n_ - i)
            for i in range(c)
        ]
    )

    n_choose_c = mp.binomial(n_, c_)

    s = 0
    for l in range(n + 1):
        n_choose_l = mp.binomial(n_, l)
        n_minus_l_choose_d_over_n_choose_d = math.prod(
            [
                (n_ - d_ - i) / (n_ - i)
                for i in range(l)
            ]
        )

        term = (
            mp.power(mp.mpf("-1"), l)
            * n_choose_c
            * n_choose_l
            * mp.power(
                n_minus_l_choose_d_over_n_choose_d * n_minus_c_choose_d_over_n_choose_d,
                m_
            )
        )
        # log(WARNING, "", term=term)
        s += term

    return s


def prob_num_nonempty_cells_eq_c(n: int, m: int, d: int, c: int) -> float:
    num_empty_cells = n - c
    return prob_num_empty_cells_eq_c_w_mpmath(n=n, m=m, d=d, c=num_empty_cells)


def prob_num_nonempty_cells_geq_c(n: int, m: int, d: int, c: int) -> float:
    return sum(
        prob_num_nonempty_cells_eq_c(n=n, m=m, d=d, c=c_)
        for c_ in range(c, n + 1)
    )


def prob_expand_span_by_e_with_each_complex(n: int, m: int, d: int, e: int) -> float:
    if e > d:
        log(WARNING, "Expansion at each step can NOT be greater than complex size", d=d, e=e)
        return 0
    elif d + (m - 1) * e > n:
        log(WARNING, "Cannot expand beyond the available nodes", d=d, e=e)
        return 0

    # n_ = mp.mpf(f"{n}")
    # d_ = mp.mpf(f"{d}")
    # e_ = mp.mpf(f"{e}")

    span = d
    prob = 1
    for step in range(m - 1):
        prob_ = mp.binomial(n - span, e) * mp.binomial(span, d - e) / mp.binomial(n, d)
        log(WARNING, f"> step= {step}", prob_=prob_)
        prob *= prob_

        span += e

    return prob


def prob_expand_span_by_at_least_e_with_each_complex(n: int, m: int, d: int, e: int) -> float:
    if e > d:
        log(WARNING, "Expansion at each step can NOT be greater than complex size", d=d, e=e)
        return 0
    elif d + (m - 1) * e > n:
        log(WARNING, "Cannot expand beyond the available nodes", d=d, e=e)
        return 0

    prob = 0
    step_prob_list = []

    def helper(cur_span: int):
        nonlocal prob
        # log(WARNING, "Started", cur_span=cur_span)

        if cur_span >= n:
            return

        elif len(step_prob_list) == m - 1:
            prob += math.prod(step_prob_list)
            return

        for e_ in range(e, d + 1):
            step_prob = mp.binomial(n - cur_span, e_) * mp.binomial(cur_span, d - e_) / mp.binomial(n, d)
            step_prob_list.append(step_prob)

            helper(cur_span=cur_span + e_)

            step_prob_list.pop()

    helper(cur_span=d)

    return prob


def prob_expand_span_as_necessary(n: int, m: int, d: int, lambda_: int) -> float:
    prob = 0
    step_prob_list = []

    def helper(cur_span: int):
        nonlocal prob
        # log(WARNING, "Started", cur_span=cur_span)

        if cur_span >= n:
            return

        elif len(step_prob_list) == m - 1:
            prob += math.prod(step_prob_list)
            return

        m_ = len(step_prob_list)
        min_e = max((m_ + 1) * lambda_ - cur_span, 0)
        for e_ in range(min_e, d + 1):
            step_prob = mp.binomial(n - cur_span, e_) * mp.binomial(cur_span, d - e_) / mp.binomial(n, d)
            step_prob_list.append(step_prob)

            helper(cur_span=cur_span + e_)

            step_prob_list.pop()

    helper(cur_span=d)

    return prob


def prob_expand_span_as_necessary_faster(n: int, m: int, d: int, lambda_: int) -> float:
    @functools.cache
    def helper(cur_span: int, cur_m: int):
        # log(WARNING, "Started", cur_span=cur_span)

        if cur_m == m:
            return 1
        elif cur_span > n:
            return 0

        prob = 0
        min_e = max((cur_m + 1) * lambda_ - cur_span, 0)
        for e_ in range(min_e, d + 1):
            step_prob = mp.binomial(n - cur_span, e_) * mp.binomial(cur_span, d - e_) / mp.binomial(n, d)
            prob_ = helper(cur_span=(cur_span + e_), cur_m=(cur_m + 1))

            prob += step_prob * prob_

        return prob

    return helper(cur_span=d, cur_m=1)


def prob_expand_span_as_necessary_alternative(n: int, m: int, d: int, lambda_: int) -> float:
    return math.prod(
        [
            prob_num_nonempty_cells_geq_c(
                n=n, m=m_, d=d, c=(m_ * lambda_)
            )
            for m_ in range(1, m + 1)
        ]
    )
