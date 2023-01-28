from src.sim import (
    random_variable,
    sim as sim_module,
)
from src.debug_utils import *


SERVICE_TIME_RV = random_variable.Exponential(mu=1)


def sim_MM1_w_finite_queue_stability(
    arrival_rate: int,
    num_requests_to_serve: int,
    queue_length: int,
    num_sim_runs: int,
) -> sim_module.SimResult:
    return sim_module.sim_single_server_w_joblib(
        inter_gen_time_rv=random_variable.Exponential(mu=arrival_rate),
        service_time_rv=SERVICE_TIME_RV,
        num_requests_to_serve=num_requests_to_serve,
        queue_length=queue_length,
        num_sim_runs=num_sim_runs,
    )


def sim_frac_dropped_requests_vs_arrival_rate():
    num_sim_runs = 2
    num_requests_to_serve = 10  # 1000
    queue_length = 10

    log(DEBUG, "Started",
        num_requests_to_serve=num_requests_to_serve,
        queue_length=queue_length,
        num_sim_runs=num_sim_runs,
    )

    arrival_rate_list = []
    sim_result_list = []
    # for arrival_rate in [0.2, 0.5, 0.8]:
    for arrival_rate in [0.2]:
        log(DEBUG, f">> arrival_rate= {arrival_rate}")
        arrival_rate_list.append(arrival_rate)

        sim_result = sim_MM1_w_finite_queue_stability(
            arrival_rate=arrival_rate,
            num_requests_to_serve=num_requests_to_serve,
            queue_length=queue_length,
            num_sim_runs=num_sim_runs,
        )
        sim_result_list.append(sim_result)

    fontsize = 14
    plot.legend(fontsize=fontsize)
    plot.ylabel("Fraction of requests dropped", fontsize=fontsize)
    # plot.yscale("log")
    plot.xlabel("Arrival rate", fontsize=fontsize)

    plot.title(
        r"$X \sim \textrm{Exp}(\lambda)$, "
        fr"$S \sim {SERVICE_TIME_RV.to_latex()}$, "
        fr"$N_r= {num_requests_to_serve}$, "
        fr"$N_s= {num_sim_runs}$, "
        fr"$N_q= {queue_length}$"
    )

    # Save the plot
    plot.gcf().set_size_inches(4, 6)
    plot.savefig(f"plot_frac_dropped_requests_vs_arrival_rate.png", bbox_inches="tight")
    plot.gcf().clear()

    log(DEBUG, "Done")
