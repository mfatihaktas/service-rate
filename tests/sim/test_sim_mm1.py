import pytest

from src.sim import (
    random_variable,
    sim as sim_module,
)
from src.utils.debug import *


@pytest.fixture(scope="module")
def inter_gen_time_rv() -> random_variable.RandomVariable:
    return random_variable.Exponential(mu=0.8)


@pytest.fixture(scope="module")
def service_time_rv() -> random_variable.RandomVariable:
    return random_variable.Exponential(mu=1)


def test_MM1(
    inter_gen_time_rv: random_variable.RandomVariable,
    service_time_rv: random_variable.RandomVariable,
):
    num_sim_runs = 2
    num_requests_to_serve = 10
    sim_result = sim_module.sim_single_server_w_joblib(
        inter_gen_time_rv=inter_gen_time_rv,
        service_time_rv=service_time_rv,
        num_requests_to_serve=num_requests_to_serve,
        num_sim_runs=num_sim_runs,
    )
    log(INFO, "", sim_result=sim_result)


def test_MM1_w_finite_queue(
    inter_gen_time_rv: random_variable.RandomVariable,
    service_time_rv: random_variable.RandomVariable,
):
    num_sim_runs = 1
    num_requests_to_serve = 100
    queue_length = 2
    sim_result = sim_module.sim_single_server_w_joblib(
        inter_gen_time_rv=inter_gen_time_rv,
        service_time_rv=service_time_rv,
        num_requests_to_serve=num_requests_to_serve,
        queue_length=queue_length,
        num_sim_runs=num_sim_runs,
    )
    log(INFO, "", sim_result=sim_result)
