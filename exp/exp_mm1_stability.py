from src.sim import (
    random_variable,
    sim as sim_module,
)
from src.debug_utils import *


def sim_MM1_w_finite_queue_stability(
    arrival_rate: int,
    num_requests_to_serve: int,
    queue_length: int,
    num_sim_runs: int = 1,
):
    log(
        DEBUG,
        "Started",
        arrival_rate=arrival_rate,
        num_requests_to_serve=num_requests_to_serve,
        queue_length=queue_length,
        num_sim_runs=num_sim_runs,
    )

    sim_result = sim_module.sim_single_server(
        inter_gen_time_rv=random_variable.Exponential(mu=arrival_rate),
        service_time_rv=random_variable.Exponential(mu=1),
        num_requests_to_serve=num_requests_to_serve,
        queue_length=queue_length,
        num_sim_runs=num_sim_runs,
    )
    log(INFO, "", sim_result=sim_result)
