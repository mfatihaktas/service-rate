import math
import numpy

from typing import Callable

from src.model import (
    demand,
)
from src.storage_overlap import (
    design,
    model,
    sim,
)
from src.sim import random_variable

from src.utils.plot import *


STRATEGY_TO_CHECK_IF_DEMAND_COVERED = design.StrategyToCheckIfDemandCovered.demand_assigner


def plot_P_vs_d(
    n: int,
    maximal_load: float,
    num_samples: int,
    num_sim_run: int,
    plot_sim: bool = True,
    plot_model: bool = True,
):
    log(INFO, "Started",
        n=n,
        maximal_load=maximal_load,
        num_samples=num_samples,
        num_sim_run=num_sim_run,
        plot_sim=plot_sim,
        plot_model=plot_model,
    )

    min_P = float("Inf")

    def plot_(
        demand_rv: random_variable.RandomVariable,
        label: str,
    ):
        nonlocal min_P
        log(DEBUG, f"> demand_rv= {demand_rv}")

        d_list = []
        E_P_list, std_P_list = [], []
        P_upper_bound_list = []
        P_upper_bound_approx_list = []

        color = next(dark_color_cycle)

        # for d in range(1, math.ceil(math.log(n)) + 1 + 3):
        # for d in range(1, 5 * math.ceil(math.log(n))):
        # for d in range(1, 11):
        # for d in range(1, 21):
        # for d in numpy.linspace(1, 20, 10):
        for d in [1, 2, 3, 5, 8, 12, 20]:
        # for d in [1, 2, 3, 5, 8, 12, 20, 30]:
            d = int(d)
            log(DEBUG, f">> d= {d}")

            d_list.append(d)

            if plot_sim:
                storage_design = design.RandomExpanderDesign(
                    k=n,
                    n=n,
                    d=d,
                    use_cvxpy=USE_CVXPY,
                    use_demand_assigner=USE_DEMAND_ASSIGNER,
                )

                demand_vector_sampler = demand.DemandVectorSamplerWithGeneralObjDemands(
                    num_objs=n,
                    demand_rv=demand_rv,
                )
                P_list = sim.sim_frac_of_demand_vectors_covered(
                    demand_vector_sampler=demand_vector_sampler,
                    storage_design=storage_design,
                    num_samples=num_samples,
                    num_sim_run=num_sim_run,
                    maximal_load=maximal_load,
                )

                E_P = numpy.mean(P_list)
                E_P_list.append(E_P)
                std_P_list.append(numpy.std(P_list))

            if plot_model:
                if isinstance(demand_rv, random_variable.Exponential):
                    E_demand = 1 / demand_rv.mu

                    # Model with complexes
                    # storage_model = model.RandomDesignModelForExpDemand(k=n, n=n, d=d, average_object_demand=E_demand)
                    # P_upper_bound = storage_model.prob_serving_upper_bound_w_complexes(
                    #     maximal_load=maximal_load,
                    #     # max_num_objs=n // 3,
                    #     # max_num_objs=n,
                    #     max_num_objs=30,
                    #     # max_num_objs=2,
                    # )

                    # Approx
                    storage_model = model.RandomDesignModelForExpDemand_wApprox(k=n, n=n, d=d, average_object_demand=E_demand)
                    P_upper_bound_approx = storage_model.prob_serving_upper_bound_w_complexes(
                        maximal_load=maximal_load,
                        # max_num_objs=n // 3,
                        max_num_objs=n,
                        # max_num_objs=30,
                        # max_num_objs=2,
                    )

                elif isinstance(demand_rv, random_variable.Bernoulli):
                    pass

                else:
                    assert_("Demand RV should be either Exponential or Bernoulli")

                # P_upper_bound_list.append(P_upper_bound)
                P_upper_bound_approx_list.append(P_upper_bound_approx)

        log(INFO, f"demand_rv= {demand_rv}",
            d_list=d_list,
            E_P_list=E_P_list,
            std_P_list=std_P_list,
            P_upper_bound_list=P_upper_bound_list,
            P_upper_bound_approx_list=P_upper_bound_approx_list,
        )

        if plot_sim:
            plot.errorbar(d_list, E_P_list, yerr=std_P_list, label=f"{label}", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
            min_P = min(min_P, min(E_P_list))

        if plot_model:
            # plot.plot(d_list, P_upper_bound_list, label=f"{label}, UB", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
            plot.plot(d_list, P_upper_bound_approx_list, label=f"{label}, UB-approx", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
            # min_P = min(min_P, min(P_upper_bound_list))
            min_P = min(min_P, min(P_upper_bound_approx_list))

        plot.xticks(d_list)

    plot.axhline(y=1, color="k", linestyle="--", alpha=0.5)

    # rho ~ Exp
    demand_dist = "\mathrm{Exp}"
    for E_demand in numpy.linspace(0.1, 0.8, 5):
    # for E_demand in [0.2, 0.3, 0.4]:
        demand_rv = random_variable.Exponential(mu=1 / E_demand)
        # label = fr"$\mu = {mu}$"
        label = r"$\mathrm{E}[\rho]=$" + fr"${round(E_demand, 2)}$"
        plot_(demand_rv=demand_rv, label=label)

    fontsize = 16
    plot.legend(fontsize=14, framealpha=0.5, loc="upper left", bbox_to_anchor=(1, 1))
    plot.ylabel(r"$\mathcal{P}$", fontsize=fontsize)
    # plot.ylabel(r"$\mathcal{P}$ for clustering design", fontsize=fontsize)
    plot.xlabel(r"$d$", fontsize=fontsize)
    plot.xscale("log")
    # plot.yscale("log")
    plot.yticks([y for y in plot.yticks()[0] if min_P <= y <= 1] + [1])

    plot.title(
        (
            fr"$k= n= {n}$, "
            fr"$m= {maximal_load}$, "
            fr"$\rho \sim {demand_dist}$"
        ),
        fontsize=fontsize,
    )

    # Save the plot
    # plot.gcf().set_size_inches(8, 6)
    plot.gcf().set_size_inches(6, 4)
    file_name = (
        "plots/plot_P_vs_d_for_random"
        + f"_n_{n}"
        + ".pdf"
    )
    plot.savefig(file_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")


def plot_P_vs_d_as_n_gets_large(
    d_func: Callable[[int], int],
    d_func_label: str,
    maximal_load: float,
    plot_sim: bool,
    plot_model: bool,
    num_samples: int = None,
    num_sim_run: int = None,
):
    log(INFO, "Started",
        d_func=d_func,
        d_func_label=d_func_label,
        maximal_load=maximal_load,
        plot_sim=plot_sim,
        plot_model=plot_model,
        num_samples=num_samples,
        num_sim_run=num_sim_run,
    )

    min_P = float("Inf")

    def plot_(
        demand_rv: random_variable.RandomVariable,
        label: str,
    ):
        nonlocal min_P
        log(DEBUG, ">",
            demand_rv=demand_rv,
            average_demand=demand_rv.mean(),
        )

        n_list = []
        E_P_list, std_P_list = [], []
        P_upper_bound_list = []
        P_upper_bound_approx_list = []

        color = next(dark_color_cycle)

        # for n in [10, 100, 200, 500]:
        # for n in [10, 100, 200, 500, 1000, 2000, 5000, 8000, 10000]:
        for n in [10 ** p for p in range(2, 6)]:
            d = max(1, d_func(n))
            log(DEBUG, ">>", n=n, d=d)

            n_list.append(n)

            if plot_sim:
                # storage_design = design.RandomExpanderDesign(
                storage_design = design.RandomBlockDesign(
                    k=n,
                    n=n,
                    d=d,
                    strategy_to_check_if_demand_covered=STRATEGY_TO_CHECK_IF_DEMAND_COVERED,
                )

                demand_vector_sampler = demand.DemandVectorSamplerWithGeneralObjDemands(
                    num_objs=n,
                    demand_rv=demand_rv,
                )
                P_list = sim.sim_frac_of_demand_vectors_covered(
                    demand_vector_sampler=demand_vector_sampler,
                    storage_design=storage_design,
                    num_samples=num_samples,
                    num_sim_run=num_sim_run,
                    maximal_load=maximal_load,
                )

                E_P = numpy.mean(P_list)
                E_P_list.append(E_P)
                std_P_list.append(numpy.std(P_list))

            if plot_model is False:
                continue

            if isinstance(demand_rv, random_variable.Exponential):
                E_demand = 1 / demand_rv.mu

                # Model with complexes
                # storage_model = model.RandomDesignModelForExpDemand(k=n, n=n, d=d, average_object_demand=E_demand)
                storage_model = model.RandomDesignModelForExpDemand_w2objs(k=n, n=n, d=d, average_object_demand=E_demand)
                # storage_model = model.RandomDesignModelForExpDemand_wApprox_w2objs(k=n, n=n, d=d, average_object_demand=E_demand)

                P_upper_bound = storage_model.prob_serving_upper_bound_w_complexes(
                    maximal_load=maximal_load,
                    # max_num_objs=n // 3,
                    # max_num_objs=n,
                    # max_num_objs=30,
                )

                # Approx
                P_upper_bound_approx = 0
                # storage_model = model.RandomDesignModelForExpDemand_wApprox(k=n, n=n, d=d, average_object_demand=E_demand)
                # P_upper_bound_approx = storage_model.prob_serving_upper_bound_w_complexes(
                #     maximal_load=maximal_load,
                #     # max_num_objs=n // 3,
                #     max_num_objs=n,
                #     # max_num_objs=30,
                #     # max_num_objs=2,
                # )

            elif isinstance(demand_rv, random_variable.Bernoulli):
                pass

            else:
                assert_("Demand RV should be either Exponential or Bernoulli")

            P_upper_bound_list.append(P_upper_bound)
            P_upper_bound_approx_list.append(P_upper_bound_approx)

        log(INFO, f"demand_rv= {demand_rv}",
            n_list=n_list,
            E_P_list=E_P_list,
            P_upper_bound_list=P_upper_bound_list,
            P_upper_bound_approx_list=P_upper_bound_approx_list,
        )

        if plot_sim:
            plot.errorbar(n_list, E_P_list, yerr=std_P_list, label=f"{label}", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
            min_P = min(min_P, min(E_P_list))

        plot.plot(n_list, P_upper_bound_list, label=f"{label}, UB", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
        # plot.plot(n_list, P_upper_bound_approx_list, label=f"{label}, UB-approx", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
        min_P = min(min_P, min(P_upper_bound_list))
        # min_P = min(min_P, min(P_upper_bound_approx_list))

        plot.xticks(n_list)

    plot.axhline(y=1, color="k", linestyle="--", alpha=0.5)

    # rho ~ Exp
    demand_dist = "\mathrm{Exp}"
    for E_demand in numpy.linspace(0.1, maximal_load, 3):
    # for E_demand in numpy.linspace(maximal_load * 0.9, maximal_load, 3):
    # for E_demand in [0.2, 0.3, 0.4]:
        demand_rv = random_variable.Exponential(mu=1 / E_demand)
        # label = fr"$\mu = {mu}$"
        label = r"$\mathrm{E}[\rho]=$" + fr"${round(E_demand, 2)}$"
        plot_(demand_rv=demand_rv, label=label)

    fontsize = 16
    plot.legend(fontsize=14, framealpha=0.5, loc="upper left", bbox_to_anchor=(1, 1))
    plot.ylabel(r"$\mathcal{P}$", fontsize=fontsize)
    # plot.ylabel(r"$\mathcal{P}$ for clustering design", fontsize=fontsize)
    plot.xlabel(r"$n$", fontsize=fontsize)
    plot.xscale("log")
    # plot.yscale("log")
    plot.yticks([y for y in plot.yticks()[0] if min_P <= y <= 1] + [1])

    plot.title(
        (
            fr"$d= {d_func_label}$, "
            fr"$m= {maximal_load}$, "
            fr"$\rho \sim {demand_dist}$"
        ),
        fontsize=fontsize,
    )

    # Save the plot
    # plot.gcf().set_size_inches(8, 6)
    plot.gcf().set_size_inches(6, 4)
    file_name = (
        "plots/plot_P_vs_d_as_n_gets_large"
        # + f"_n_{n}"
        + ".pdf"
    )
    plot.savefig(file_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")


if __name__ == "__main__":
    # plot_P_vs_d(
    #     # n=30,
    #     # n=120,
    #     n=500,
    #     # n=1200,
    #     maximal_load=0.7,
    #     num_samples=3,
    #     num_sim_run=100,
    #     plot_sim=True,
    #     plot_model=True,
    # )

    # d_func = lambda n : math.ceil(math.log(math.log(math.log(n))))
    # d_func_label = r"\log(\log(\log(n)))"

    # d_func = lambda n : math.ceil(math.log(math.log(n)))
    # d_func_label = r"\log(\log(n))"

    # d_func = lambda n : math.ceil(math.sqrt(math.log(n)))
    # d_func_label = r"\log(n)^{1/2}"

    power = 0.8
    d_func = lambda n : math.floor(math.log(n) ** power)
    d_func_label = fr"\log(n)^{power}"

    # d_func = lambda n : math.ceil(math.log(n))
    # d_func_label = r"\log(n)"

    # d_func = lambda n : math.ceil(math.log(n)**2)
    # d_func_label = r"\log(n)^2"

    # d_func = lambda n : math.ceil(n / 2)
    # d_func_label = r"n / 2"

    plot_P_vs_d_as_n_gets_large(
        d_func=d_func,
        d_func_label=d_func_label,
        maximal_load=0.7,
        num_samples=3,
        num_sim_run=50,
        plot_sim=True,
        plot_model=False,
    )
