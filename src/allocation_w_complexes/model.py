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
