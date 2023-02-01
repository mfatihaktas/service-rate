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


def sim_frac_dropped_requests_vs_arrival_rate_for_given_queue_length(
    num_requests_to_serve: int,
    num_sim_runs: int,
    queue_length: int
):
    log(INFO, "Started",
        num_requests_to_serve=num_requests_to_serve,
        num_sim_runs=num_sim_runs,
        queue_length=queue_length,
    )

    arrival_rate_list = []
    summary_sim_result_list = []
    # for arrival_rate in [0.5, 0.8, 0.9, 0.95]:
    # for arrival_rate in [0.2]:
    for arrival_rate in [0.7, 0.8, 0.9, 0.95]:
        log(INFO, f">>> arrival_rate= {arrival_rate}")
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
    log(INFO, "",
        queue_length=queue_length,
        arrival_rate_list=arrival_rate_list,
        E_frac_dropped_requests_list=E_frac_dropped_requests_list,
        std_frac_dropped_requests_list=std_frac_dropped_requests_list,
    )
    plot.errorbar(arrival_rate_list, E_frac_dropped_requests_list, yerr=std_frac_dropped_requests_list, color=next(dark_color_cycle), label=fr"$N_q= {queue_length}$", marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    log(INFO, "Done")


def sim_frac_dropped_requests_vs_arrival_rate():
    num_requests_to_serve = 10000
    num_sim_runs = 2 * 8
    log(INFO, "Started", num_requests_to_serve=num_requests_to_serve, num_sim_runs=num_sim_runs)

    for queue_length in [10, 20, 100]:
        log(INFO, f">> queue_length= {queue_length}")
        sim_frac_dropped_requests_vs_arrival_rate_for_given_queue_length(
            num_requests_to_serve=num_requests_to_serve,
            num_sim_runs=num_sim_runs,
            queue_length=queue_length,
        )

    fontsize = 14
    plot.legend(fontsize=fontsize)
    # plot.yscale("log")
    plot.ylabel("Fraction of requests dropped", fontsize=fontsize)
    plot.xlabel("Arrival rate", fontsize=fontsize)

    plot.title(
        r"$X \sim \textrm{Exp}(\lambda)$, "
        fr"$S \sim {SERVICE_TIME_RV.to_latex()}$, "
        r"$N_{\textrm{req}}$" + fr"$ = {num_requests_to_serve}$, "
        r"$N_{\textrm{sim}}$" + fr"$ = {num_sim_runs}$"
    )

    # Save the plot
    plot.gcf().set_size_inches(10, 6)
    plot_name = (
        "plot_frac_dropped_requests_vs_arrival_rate_w"
        f"_num_requests_to_serve_{num_requests_to_serve}"
        f"_num_sim_runs_{num_sim_runs}"
        ".png"
    )
    plot.savefig(plot_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")


def plot_frac_dropped_requests_vs_arrival_rate():
    num_requests_to_serve = 10000
    num_sim_runs = 16
    log(INFO, "Started", num_requests_to_serve=num_requests_to_serve, num_sim_runs=num_sim_runs)

    queue_length_to_sim_data_map_w_num_requests_to_serve_10000_num_sim_runs_16 = {
        10: {
            "arrival_rate_list": [0.7, 0.8, 0.9, 0.95],
            "E_frac_dropped_requests_list": [0.005818749999999999, 0.019131250000000002, 0.045675, 0.06887499999999999],
            "std_frac_dropped_requests_list": [0.0020359944591034624, 0.003591608057333094, 0.004929439623324339, 0.006434526012069576],
        },

        20: {
            "arrival_rate_list": [0.7, 0.8, 0.9, 0.95],
            "E_frac_dropped_requests_list": [6.875e-05, 0.0016125, 0.0114375, 0.024875],
            "std_frac_dropped_requests_list": [0.00012103072956898178, 0.0009499177595981665, 0.0029019120851604034, 0.005581610430691128],
        },

        100: {
            "arrival_rate_list": [0.7, 0.8, 0.9, 0.95],
            "E_frac_dropped_requests_list": [0.0, 0.0, 0.0, 0.0],
            "std_frac_dropped_requests_list": [0.0, 0.0, 0.0, 0.0],
        },
    }
    queue_length_to_sim_data_map = queue_length_to_sim_data_map_w_num_requests_to_serve_10000_num_sim_runs_16

    for queue_length in [10, 20, 100]:
        log(INFO, f">> queue_length= {queue_length}")

        sim_data_map = queue_length_to_sim_data_map[queue_length]
        plot.errorbar(sim_data_map["arrival_rate_list"], sim_data_map["E_frac_dropped_requests_list"], yerr=sim_data_map["std_frac_dropped_requests_list"], color=next(dark_color_cycle), label=r"$N_{\textrm{queue}}$" + fr"$ = {queue_length}$", marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    fontsize = 14
    plot.legend(fontsize=fontsize)
    plot.ylabel("Fraction of requests dropped", fontsize=fontsize)
    plot.xlabel(r"Arrival rate $\lambda$", fontsize=fontsize)

    plot.title(
        r"$X \sim \textrm{Exp}(\lambda)$, "
        fr"$S \sim {SERVICE_TIME_RV.to_latex()}$, "
        r"$N_{\textrm{req}}$" + fr"$ = {num_requests_to_serve}$, "
        r"$N_{\textrm{sim}}$" + fr"$ = {num_sim_runs}$"
    )

    # Save the plot
    plot.gcf().set_size_inches(10, 6)
    plot_name = (
        "plot_frac_dropped_requests_vs_arrival_rate_w_num"
        f"_requests_to_serve_{num_requests_to_serve}"
        f"_num_sim_runs_{num_sim_runs}"
        ".png"
    )
    plot.savefig(plot_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")



if __name__ == "__main__":
    # sim_frac_dropped_requests_vs_arrival_rate()
    plot_frac_dropped_requests_vs_arrival_rate()
