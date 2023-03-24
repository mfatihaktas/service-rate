import pytest

from typing import Tuple

from src.allocation_w_complexes import (
    model,
    sim as allocation_w_complexes_sim,
)
from src.storage_overlap import (
    design,
    sim as storage_overlap_sim,
)

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


def test_prob_num_nonempty_cells_eq(n_m_d: Tuple[int, int, int]):
    n, m, d = n_m_d
    num_samples = 10**4

    log(DEBUG, "Started", n=n, m=m, d=d)

    num_nonempty_cells_to_prob_map = allocation_w_complexes_sim.sim_num_nonempty_cells_to_prob_map(n=n, m=m, d=d, num_samples=num_samples)

    storage_design = design.RandomExpanderDesign(k=n, n=n, d=d, use_cvxpy=False)
    object_span_to_prob_map = storage_overlap_sim.sim_object_span_to_prob_map(storage_design=storage_design, m=m, num_samples=num_samples)

    cum_prob_model = 0
    cum_diff_between_model_and_sim = 0
    cum_diff_between_model_and_sim_w_storage_design = 0
    for num_nonempty_cells in range(d, m * d):
        prob_sim = num_nonempty_cells_to_prob_map[num_nonempty_cells] if num_nonempty_cells in num_nonempty_cells_to_prob_map else 0
        prob_sim_w_storage_design = object_span_to_prob_map[num_nonempty_cells] if num_nonempty_cells in object_span_to_prob_map else 0
        prob_model = model.prob_num_nonempty_cells_eq_c(n=n, m=m, d=d, c=num_nonempty_cells)
        cum_prob_model += prob_model

        diff_between_model_and_sim = abs(prob_model - prob_sim)
        diff_between_model_and_sim_w_storage_design = abs(prob_model - prob_sim_w_storage_design)
        log(DEBUG, f"> num_nonempty_cells= {num_nonempty_cells}",
            prob_sim=prob_sim,
            prob_sim_w_storage_design=prob_sim_w_storage_design,
            prob_model=prob_model,
            diff_between_model_and_sim=diff_between_model_and_sim,
            diff_between_model_and_sim_w_storage_design=diff_between_model_and_sim_w_storage_design,
        )

        cum_diff_between_model_and_sim += diff_between_model_and_sim
        cum_diff_between_model_and_sim_w_storage_design += diff_between_model_and_sim_w_storage_design

    log(INFO, "Done",
        cum_diff_between_model_and_sim=cum_diff_between_model_and_sim,
        cum_diff_between_model_and_sim_w_storage_design=cum_diff_between_model_and_sim_w_storage_design,
        cum_prob_model=cum_prob_model,
    )


def test_prob_num_nonempty_cells_geq(n_m_d: Tuple[int, int, int]):
    n, m, d = n_m_d

    log(DEBUG, "Started", n=n, m=m, d=d)

    num_nonempty_cells_to_tail_prob_map = allocation_w_complexes_sim.sim_num_nonempty_cells_to_tail_prob_map(n=n, m=m, d=d, num_samples=10**5)

    cum_diff_between_model_and_sim = 0
    for num_nonempty_cells in range(d, m * d):
        tail_prob_sim = num_nonempty_cells_to_tail_prob_map[num_nonempty_cells] if num_nonempty_cells in num_nonempty_cells_to_tail_prob_map else 0
        tail_prob_model = model.prob_num_nonempty_cells_geq_c(n=n, m=m, d=d, c=num_nonempty_cells)

        diff_between_model_and_sim = abs(tail_prob_model - tail_prob_sim)
        log(DEBUG, f"> num_nonempty_cells= {num_nonempty_cells}",
            tail_prob_sim=tail_prob_sim,
            tail_prob_model=tail_prob_model,
            diff_between_model_and_sim=diff_between_model_and_sim,
        )

        cum_diff_between_model_and_sim += diff_between_model_and_sim

    log(INFO, "Done",
        cum_diff_between_model_and_sim=cum_diff_between_model_and_sim,
    )


def test_prob_expand_span_as_necessary():
    n, d = 100, 5
    lambda_ = d - 1

    log(DEBUG, "Started", n=n, d=d, lambda_=lambda_)

    for m in range(2, 20):
        prob_expand_span_as_necessary = model.prob_expand_span_as_necessary(n=n, m=m, d=d, lambda_=lambda_)
        prob_expand_span_as_necessary_faster = model.prob_expand_span_as_necessary_faster(n=n, m=m, d=d, lambda_=lambda_)

        # prob_expand_span_as_necessary_alternative = model.prob_expand_span_as_necessary_alternative(n=n, m=m, d=d, lambda_=lambda_)

        log(INFO, f"> m= {m}",
            prob_expand_span_as_necessary=prob_expand_span_as_necessary,
            prob_expand_span_as_necessary_faster=prob_expand_span_as_necessary_faster,
            # prob_expand_span_as_necessary_alternative=prob_expand_span_as_necessary_alternative,
        )

    log(DEBUG, "Done", n=n, d=d)


@pytest.fixture(
    scope="session",
    params=[
        # (10, 4, 2, 2, 3),
        # (10, 4, 3, 2, 5),
        # (30, 10, 3, 2, 5),
        # (30, 10, 3, 2, 4),
        # (100, 20, 5, 2, 8),
        # (100, 20, 5, 2, 9),
        # (100, 20, 5, 2, 10),
        # (100, 30, 3, 2, 5),
        # (100, 20, 4, 3, 8),
        # (100, 25, 6, 4, 10),
        # (100, 15, 5, 3, 12),  # Diff
        # (100, 10, 5, 2, 7),
        # (100, 15, 5, 4, 15),  # Diff
        # (100, 15, 4, 3, 10),  # Diff
        # (100, 10, 4, 3, 9),
        (150, 15, 4, 3, 10),  # Diff
        # (150, 15, 4, 2, 7),
        # (150, 20, 7, 5, 25),
        # (150, 20, 7, 2, 12),
    ],
)
def n_m_d_t_u(request) -> Tuple[int, int, int, int, int]:
    return request.param


def test_prob_span_of_all_t_tuples_geq_u(n_m_d_t_u: Tuple[int, int, int, int, int]):
    n, m, d, t, u = n_m_d_t_u
    num_samples = 10**4

    sim_prob_span_of_all_t_tuples_geq_u = allocation_w_complexes_sim.sim_prob_span_of_every_t_complexes_geq_u(
        n=n, m=m, d=d, t=t, u=u, num_samples=num_samples
    )
    # sim_prob_span_of_all_t_tuples_geq_u_alternative = allocation_w_complexes_sim.sim_prob_span_of_every_t_complexes_geq_u_alternative(
    #     n=n, m=m, d=d, t=t, u=u, num_samples=num_samples
    # )

    prob_span_of_all_t_tuples_geq_u_alternative = model.prob_span_of_every_t_complexes_geq_u_alternative(
        n=n, m=m, d=d, t=t, u=u
    )

    log(INFO, "",
        sim_prob_span_of_all_t_tuples_geq_u=sim_prob_span_of_all_t_tuples_geq_u,
        # sim_prob_span_of_all_t_tuples_geq_u_alternative=sim_prob_span_of_all_t_tuples_geq_u_alternative,
        prob_span_of_all_t_tuples_geq_u_alternative=prob_span_of_all_t_tuples_geq_u_alternative,
    )
