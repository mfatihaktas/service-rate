import dataclasses
import joblib
import numpy
import simpy

from src.sim import (
    random_variable,
    server as server_module,
    sink as sink_module,
    source as source_module,
)

from src.debug_utils import *


@dataclasses.dataclass(repr=False)
class SimResult:
    t_l: list[float]

    ET: float = None
    std_T: float = None

    def __repr__(self):
        return "SimResult( \n" f"\t ET= {self.ET} \n" f"\t std_T= {self.std_T} \n" ")"

    def __post_init__(self):
        self.ET = numpy.mean(self.t_l)
        self.std_T = numpy.std(self.t_l)


def combine_sim_results(sim_result_list: list[SimResult]) -> SimResult:
    t_l = []
    for sim_result in sim_result_list:
        t_l.extend(sim_result.t_l)

    return SimResult(t_l=t_l)


def sim_single_server(
    inter_gen_time_rv: random_variable.RandomVariable,
    service_time_rv: random_variable.RandomVariable,
    num_requests_to_serve: int,
    queue_length: int = None,
    sim_result_list: list[SimResult] = None,
):
    log(
        DEBUG,
        "Started",
        inter_gen_time_rv=inter_gen_time_rv,
        service_time_rv=service_time_rv,
        num_requests_to_serve=num_requests_to_serve,
    )

    env = simpy.Environment()
    sink = sink_module.Sink(
        env=env, _id="sink", num_requests_to_recv=num_requests_to_serve
    )

    if queue_length:
        server = server_module.ServerWithFiniteQueue(
            env=env, _id="server", sink=sink, queue_length=queue_length
        )
    else:
        server = server_module.Server(env=env, _id="server", sink=sink)

    _ = source_module.Source(
        env=env,
        _id="source",
        inter_gen_time_rv=inter_gen_time_rv,
        service_time_rv=service_time_rv,
        next_hop=server,
    )

    env.run(until=sink.recv_requests_proc)

    sim_result = SimResult(t_l=sink.request_response_time_list)
    log(INFO, "Done", sim_result=sim_result)
    if sim_result_list is not None:
        sim_result_list.append(sim_result)


def sim_single_server_w_joblib(
    inter_gen_time_rv: random_variable.RandomVariable,
    service_time_rv: random_variable.RandomVariable,
    num_requests_to_serve: int,
    queue_length: int = None,
    num_sim_runs: int = 1,
) -> SimResult:
    log(
        DEBUG,
        "Started",
        inter_gen_time_rv=inter_gen_time_rv,
        service_time_rv=service_time_rv,
        num_requests_to_serve=num_requests_to_serve,
        num_sim_runs=num_sim_runs,
    )

    sim_result_list = []
    if num_sim_runs == 1:
        sim_single_server(
            inter_gen_time_rv=inter_gen_time_rv,
            service_time_rv=service_time_rv,
            num_requests_to_serve=num_requests_to_serve,
            queue_length=queue_length,
            sim_result_list=sim_result_list,
        )

    else:
        joblib.Parallel(n_jobs=-1, prefer="threads")(
            joblib.delayed(sim_single_server)(
                inter_gen_time_rv=inter_gen_time_rv,
                service_time_rv=service_time_rv,
                num_requests_to_serve=num_requests_to_serve,
                queue_length=queue_length,
                sim_result_list=sim_result_list,
            )
            for i in range(num_sim_runs)
        )

    log(INFO, "Done", sim_result_list=sim_result_list)

    if len(sim_result_list) > 1:
        sim_result = combine_sim_results(sim_result_list=sim_result_list)
    else:
        sim_result = sim_result_list[0]

    log(INFO, "Done", sim_result=sim_result)
    return sim_result
