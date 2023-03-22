import pytest

from typing import Tuple

from src.model import allocation_w_complexes

from src.utils.debug import *


@pytest.fixture(
    scope="session",
    params=[
        # (5, 2, 2),
        # (10, 5, 2),
        # (20, 8, 4),
        # (30, 12, 5),
        # (30, 1, 4),
        # (40, 1, 4),
        # (40, 12, 5),
        # (60, 12, 5),
        # (70, 12, 5),
        # (80, 12, 5),
        # (100, 5, 4),
        # (100, 10, 5),
        # (120, 5, 4),
        # (120, 10, 5),
        (120, 20, 5),
    ],
)
def n_m_d(request) -> Tuple[int, int, int]:
    return request.param


def test_prob_num_empty_cells_eq_c(
    n_m_d: Tuple[int, int, int],
):
    n, m, d = n_m_d

    cum_prob = 0
    for c in range(n):
        # prob_num_empty_cells_eq_c = allocation_w_complexes.prob_num_empty_cells_eq_c(n=n, m=m, d=d, c=c)
        prob_num_empty_cells_eq_c = allocation_w_complexes.prob_num_empty_cells_eq_c_w_mpmath(n=n, m=m, d=d, c=c)
        # prob_num_empty_cells_eq_c = allocation_w_complexes.prob_num_empty_cells_eq_c_w_mpmath_2(n=n, m=m, d=d, c=c)
        log(INFO, "",
            prob_num_empty_cells_eq_c=prob_num_empty_cells_eq_c,
            n=n, m=m, d=d, c=c,
        )

        cum_prob += prob_num_empty_cells_eq_c

    log(INFO, "", cum_prob=cum_prob)
