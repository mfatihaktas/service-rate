import pytest

from typing import Callable

from src.sim import random_variable
from src.storage_overlap import math_utils

from src.utils.debug import *


@pytest.fixture(
    scope="session",
    params=[
        random_variable.Exponential(mu=1),
        # random_variable.Pareto(loc=1, a=2.5),
    ],
)
def demand_rv(request) -> Callable[[float], float]:
    return request.param


def test_prob_cum_demand_leq_cum_supply(
    demand_rv: random_variable.RandomVariable,
):
    prob = math_utils.prob_cum_demand_leq_cum_supply_w_mpmath(
        num_demands=2,
        demand_pdf=demand_rv.pdf,
        d=2,
        span_size=3,
    )

    log(DEBUG, "", prob=prob)


def test_prob_cum_demand_leq_cum_supply_w_scipy(
    demand_rv: random_variable.RandomVariable,
):
    prob = math_utils.prob_cum_demand_leq_cum_supply_w_scipy(
        num_demands=1,
        demand_pdf=demand_rv.pdf,
        d=2,
        span_size=3,
    )

    log(DEBUG, "", prob=prob)


def test_prob_cum_demand_leq_cum_supply_w_scipy_and_numba(
    demand_rv: random_variable.RandomVariable,
):
    import time

    num_demands = 15
    d = 2
    span_size = 3

    start_time = time.time()
    prob = 0
    # prob = math_utils.prob_cum_demand_leq_cum_supply_w_scipy(
    #     num_demands=num_demands,
    #     demand_pdf=demand_rv.pdf,
    #     d=d,
    #     span_size=span_size,
    # )
    exec_time = time.time() - start_time

    start_time = time.time()
    prob_w_numba = math_utils.prob_cum_demand_leq_cum_supply_w_scipy_and_numba_for_exp_demand(
        num_demands=num_demands,
        mu=demand_rv.mu,
        d=d,
        span_size=span_size,
    )
    exec_time_w_numba = time.time() - start_time

    log(INFO, "",
        prob=prob,
        prob_w_numba=prob_w_numba,
        exec_time=exec_time,
        exec_time_w_numba=exec_time_w_numba,
    )
