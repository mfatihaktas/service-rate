import math
import numpy

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


USE_CVXPY = True  # False


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

    def plot_(
        demand_rv: random_variable.RandomVariable,
        label: str,
    ):
        log(DEBUG, f"> demand_rv= {demand_rv}")

        d_list = []
        E_P_list, std_P_list = [], []
        P_upper_bound_list = []

        color = next(dark_color_cycle)

        # for d in range(1, math.ceil(math.log(n)) + 1 + 3):
        # for d in range(1, 5 * math.ceil(math.log(n))):
        for d in range(1, 11):
            if n % d != 0:
                continue

            log(DEBUG, f">> d= {d}")

            d_list.append(d)

            if plot_sim:
                storage_design = design.RandomExpanderDesign(k=n, n=n, d=d, use_cvxpy=USE_CVXPY)

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
                    storage_model = model.RandomDesignModelForExpDemand(k=n, n=n, d=d, average_object_demand=E_demand)
                    P_upper_bound = storage_model.prob_serving_upper_bound_w_complexes(maximal_load=maximal_load)

                elif isinstance(demand_rv, random_variable.Bernoulli):
                    pass

                else:
                    assert_("Demand RV should be either Exponential or Bernoulli")

                P_upper_bound_list.append(P_upper_bound)

        log(INFO, f"demand_rv= {demand_rv}",
            d_list=d_list,
            E_P_list=E_P_list,
            std_P_list=std_P_list,
            P_upper_bound_list=P_upper_bound_list,
        )

        if plot_sim:
            plot.errorbar(d_list, E_P_list, yerr=std_P_list, label=f"{label}", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

        if plot_model:
            plot.plot(d_list, P_upper_bound_list, label=f"{label}, UB", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

        plot.xticks(d_list)

    plot.axhline(y=1, color="k", linestyle="--", alpha=0.5)

    # rho ~ Exp
    demand_dist = "\mathrm{Exp}"
    # for E_demand in numpy.linspace(1, 1, 1):
    for E_demand in [0.2, 0.3, 0.4]:
        demand_rv = random_variable.Exponential(mu=1 / E_demand)
        # label = fr"$\mu = {mu}$"
        label = r"$\mathrm{E}[\rho]=$" + fr"${E_demand}$"
        plot_(demand_rv=demand_rv, label=label)

    fontsize = 16
    plot.legend(fontsize=14)
    plot.ylabel(r"$\mathcal{P}$", fontsize=fontsize)
    # plot.ylabel(r"$\mathcal{P}$ for clustering design", fontsize=fontsize)
    plot.xlabel(r"$d$", fontsize=fontsize)
    plot.yscale("log")
    plot.yticks([y for y in plot.yticks()[0] if y < 1] + [1])

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


if __name__ == "__main__":
    plot_P_vs_d(
        n=120,
        maximal_load=0.7,
        num_samples=3,
        num_sim_run=100,
        plot_sim=True,
        plot_model=True,
    )
