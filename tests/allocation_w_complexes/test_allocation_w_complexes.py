import pytest

from typing import Tuple

from src.allocation_w_complexes import model, sim

from src.utils.debug import *


@pytest.fixture(
    scope="session",
    params=[
        # (40, 10, 3),
        # (100, 10, 3),
        (100, 20, 5),
    ],
)
def n_m_d(request) -> Tuple[int, int, int]:
    return request.param


def test_prob_num_empty_cells(n_m_d: Tuple[int, int, int]):
    n, m, d = n_m_d

    log(DEBUG, "Started", n=n, m=m, d=d)

    num_empty_cells_to_prob_map = sim.sim_num_empty_cells_to_prob_map(n=n, m=m, d=d, num_sample=10**5)

    cum_prob_model = 0
    cum_diff_between_model_and_sim = 0
    for num_nonempty_cells in range(d, m * d):
        num_empty_cells = n - num_nonempty_cells

        prob_sim = num_empty_cells_to_prob_map[num_empty_cells] if num_empty_cells in num_empty_cells_to_prob_map else 0
        prob_model = model.prob_num_empty_cells_eq_c_w_mpmath(n=n, m=m, d=d, c=num_empty_cells)
        cum_prob_model += prob_model

        diff_between_model_and_sim = abs(prob_model - prob_sim)
        log(DEBUG, f"> num_nonempty_cells= {num_nonempty_cells}, diff_between_model_and_sim= {diff_between_model_and_sim}",
            prob_sim=prob_sim, prob_model=prob_model
        )

        cum_diff_between_model_and_sim += diff_between_model_and_sim

    log(INFO, "Done",
        cum_diff_between_model_and_sim=cum_diff_between_model_and_sim,
        cum_prob_model=cum_prob_model,
    )
