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
        P_model_list = []

        color = next(dark_color_cycle)

        # for d in range(1, math.ceil(math.log(n)) + 1 + 3):
        for d in range(1, 11):
            if n % d != 0:
                continue

            log(DEBUG, f">> d= {d}")

            d_list.append(d)

            if plot_sim:
                storage_design = design.ClusteringDesign(k=n, n=n, d=d, use_cvxpy=USE_CVXPY)

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
                check(
                    (
                        isinstance(demand_rv, random_variable.Exponential)
                        or isinstance(demand_rv, random_variable.Bernoulli)
                    ),
                    "Demand RV should be either Exponential or Bernoulli",
                    demand_rv=demand_rv,
                )

                # storage_model = model.ClusteringDesignModelForBernoulliObjDemands(k=n, n=n, b=1, d=d)
                storage_model = model.ClusteringDesignModelForExpObjDemands(k=n, n=n, b=1, d=d)

                # P_model = storage_model.prob_serving(mu=mu)
                P_model = storage_model.prob_serving_w_downscaling_mean_obj_demand_w_b(
                    mean_obj_demand_b_1=mu, maximal_load=maximal_load
                )
                P_model_list.append(P_model)

        log(INFO, f"demand_rv= {demand_rv}",
            d_list=d_list,
            E_P_list=E_P_list,
            std_P_list=std_P_list,
            P_model_list=P_model_list,
        )

        if plot_sim:
            plot.errorbar(d_list, E_P_list, yerr=std_P_list, label=f"{label}", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

        if plot_model:
            plot.plot(d_list, P_model_list, label=f"{label}, model", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    # rho ~ Exp
    demand_dist = "\mathrm{Exp}"
    # for E_demand in numpy.linspace(1, 1, 1):
    for E_demand in [0.2, 0.3, 0.4]:
        demand_rv = random_variable.Exponential(mu=1 / E_demand)
        # label = fr"$\mu = {mu}$"
        label = r"$\mathrm{E}[\rho_i]=$" + fr"${E_demand}$"
        plot_(demand_rv=demand_rv, label=label)

    fontsize = 16
    plot.legend(fontsize=14)
    plot.ylabel(r"$\mathcal{P}$ for clustering design", fontsize=fontsize)
    plot.xlabel(r"$d$", fontsize=fontsize)

    plot.title(
        (
            fr"$k= n= {n}$, "
            fr"$m= {maximal_load}$, "
            fr"$\rho \sim {demand_dist}$"
        ),
        fontsize=fontsize
    )

    # Save the plot
    # plot.gcf().set_size_inches(8, 6)
    plot.gcf().set_size_inches(6, 4)
    file_name = (
        "plots/plot_P_vs_d_for_clustering"
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
        num_sim_run=1000,
        plot_sim=True,
        plot_model=False,
    )
