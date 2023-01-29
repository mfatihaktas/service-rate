from src.sim import (
    random_variable,
    sim as sim_module,
)
from src.debug_utils import *
from src.plot_utils import *


SERVICE_TIME_RV = random_variable.Exponential(mu=1)


def sim_MM1_w_finite_queue_stability(
    arrival_rate: int,
    num_requests_to_serve: int,
    queue_length: int,
    num_sim_runs: int,
) -> sim_module.SummarySimResult:
    return sim_module.sim_single_server_w_joblib(
        inter_gen_time_rv=random_variable.Exponential(mu=arrival_rate),
        service_time_rv=SERVICE_TIME_RV,
        num_requests_to_serve=num_requests_to_serve,
        queue_length=queue_length,
        num_sim_runs=num_sim_runs,
    )


def sim_frac_dropped_requests_vs_arrival_rate():
    num_sim_runs = 10
    num_requests_to_serve = 1000
    queue_length = 10

    log(INFO, "Started",
        num_requests_to_serve=num_requests_to_serve,
        queue_length=queue_length,
        num_sim_runs=num_sim_runs,
    )

    arrival_rate_list = []
    summary_sim_result_list = []
    for arrival_rate in [0.2, 0.5, 0.8]:
    # for arrival_rate in [0.2]:
        log(INFO, f">> arrival_rate= {arrival_rate}")
        arrival_rate_list.append(arrival_rate)

        summary_sim_result = sim_MM1_w_finite_queue_stability(
            arrival_rate=arrival_rate,
            num_requests_to_serve=num_requests_to_serve,
            queue_length=queue_length,
            num_sim_runs=num_sim_runs,
        )
        summary_sim_result_list.append(summary_sim_result)

    E_frac_dropped_requests_list = [summary_sim_result.E_frac_dropped_requests for summary_sim_result in summary_sim_result_list]
    std_frac_dropped_requests_list = [summary_sim_result.std_frac_dropped_requests for summary_sim_result in summary_sim_result_list]
    plot.errorbar(arrival_rate_list, E_frac_dropped_requests_list, yerr=std_frac_dropped_requests_list, color=next(dark_color_cycle), label=fr"$N_q= {queue_length}$", marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    fontsize = 14
    plot.legend(fontsize=fontsize)
    plot.ylabel("Fraction of requests dropped", fontsize=fontsize)
    # plot.yscale("log")
    plot.xlabel("Arrival rate", fontsize=fontsize)

    plot.title(
        r"$X \sim \textrm{Exp}(\lambda)$, "
        fr"$S \sim {SERVICE_TIME_RV.to_latex()}$, "
        r"$N_{\textrm{req}}$" + fr"$ = {num_requests_to_serve}$, "
        r"$N_{\textrm{sim}}$" + fr"$ = {num_sim_runs}$"
    )

    # Save the plot
    plot.gcf().set_size_inches(4, 6)
    plot.savefig("plot_frac_dropped_requests_vs_arrival_rate.png", bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")


if __name__ == "__main__":
    sim_frac_dropped_requests_vs_arrival_rate()
