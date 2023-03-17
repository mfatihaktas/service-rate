"""Let there be `n` (N) cells into which `m` (n') complexes of particles are thrown
independently of each other, `d` (m) particles in each complex. Thus, `t = m * d`
(n = n' * m) is the total number of particles thrown into the cells.


Parameters inside `()` refer to the notation used in [1].

Refs:
[1] Kolchin, Valentin Fedorovich, Boris A. Sevast'janov, and Vladimir P. Christ'Yakov.
Random allocations. Vh Winston, 1978.
"""

import scipy

from src.utils.debug import *


def prob_num_cells_w_r_particles_eq_c(n: int, m: int, d: int, r: int, c: int) -> float:
    """Denoted as `Pr{mu_r'(n, N) = c}` in [1]."""

    if r == 0:
        return prob_num_cells_w_zero_particles_eq_c(n=n, m=m, d=d, c=c)

    else:
        assert_("Defined only for r=0", r=r)


def prob_num_cells_w_zero_particles_eq_c(n: int, m: int, d: int, c: int) -> float:
    """Denoted as `Pr{mu_0'(n, N) = c}` in [1]."""

    n_choose_d = scipy.special.comb(n, d)

    if c == 0:
        s = 0
        for l in range(n + 1):
            n_choose_l = scipy.special.comb(n, l)
            n_minus_l_choose_d = scipy.special.comb(n - l, d)
            s += (-1)**l * n_choose_l * n_minus_l_choose_d**m

        coeff = n_choose_d**(-m)
        return coeff * s

    n_choose_c = scipy.special.comb(n, c)
    n_minus_c_choose_d = scipy.special.comb(n - c, d)

    coeff = n_choose_c * (n_minus_c_choose_d / n_choose_d)**m
    return coeff * prob_num_cells_w_zero_particles_eq_c(n=n - c, m=m, d=d, c=0)
