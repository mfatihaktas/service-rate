import pytest

from typing import Callable

from src.sim import random_variable
from src.storage_overlap import math_utils

from src.utils.debug import *


@pytest.fixture(
    scope="session",
    params=[
        # random_variable.Exponential(mu=1),
        random_variable.Pareto(loc=1, a=2.5),
    ],
)
def obj_demand_pdf(request) -> Callable[[float], float]:
    rv = request.param
    return rv.pdf


def test_prob_cum_demand_leq_cum_supply(
    obj_demand_pdf: Callable[[float], float],
):
    prob = math_utils.prob_cum_demand_leq_cum_supply(
        num_objs=2,
        obj_demand_pdf=obj_demand_pdf,
        d=2,
        cum_supply=3,
    )

    log(DEBUG, "", prob=prob)
