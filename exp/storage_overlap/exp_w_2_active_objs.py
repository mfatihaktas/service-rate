import dataclasses
import joblib
import numpy

from typing import Callable

from src.storage_overlap import (
    design,
    model,
    sim,
)
from src.model import demand
from src.sim import random_variable

from src.utils.plot import *


def plot_P_for_given_params(
    k: int,
    d: int,
    active_obj_demand_rv_list: list[random_variable.RandomVariable],
    func_active_obj_demand_to_x: Callable,
    x_label: str,
    maximal_load: float,
    run_sim: bool = False,
    plot_model: bool = True,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    log(INFO, "Started",
        k=k,
        d=d,
        active_obj_demand_rv_list=active_obj_demand_rv_list,
        x_label=x_label,
        maximal_load=maximal_load,
        run_sim=run_sim,
        plot_model=plot_model,
        num_samples=num_samples,
        num_sim_run=num_sim_run,
    )

    num_active_objs = 2

    def plot_(
        storage_design: design.StorageDesign,
        storage_model: model.StorageDesignModel,
    ):
        log(INFO, f">> storage_design= {storage_design}")

        x_list = []

        E_frac_of_demand_vectors_covered_list = []
        std_frac_of_demand_vectors_covered_list = []

        P_model_list = []

        for active_obj_demand_rv in active_obj_demand_rv_list:
            log(INFO, f"> active_obj_demand_rv= {active_obj_demand_rv}")

            x_list.append(
                func_active_obj_demand_to_x(active_obj_demand_rv)
            )

            demand_vector_sampler = demand.DemandVectorSamplerWithFixedNumActiveObjs(
                num_objs=storage_design.k,
                num_active_objs=num_active_objs,
                active_obj_demand_rv=active_obj_demand_rv,
            )

            # Sim
            E_frac_of_demand_vectors_covered = None
            if run_sim:
                frac_of_demand_vectors_covered_list = sim.sim_frac_of_demand_vectors_covered(
                    demand_vector_sampler=demand_vector_sampler,
                    storage_design=storage_design,
                    num_samples=num_samples,
                    num_sim_run=num_sim_run,
                    maximal_load=maximal_load,
                )

                E_frac_of_demand_vectors_covered = numpy.mean(frac_of_demand_vectors_covered_list)
                E_frac_of_demand_vectors_covered_list.append(E_frac_of_demand_vectors_covered)
                std_frac_of_demand_vectors_covered_list.append(numpy.std(frac_of_demand_vectors_covered_list))

            # UB
            P_ub = None
            if plot_model:
                # P_ub = None
                P_ub = storage_model.prob_serving_upper_bound(
                    demand_rv=active_obj_demand_rv,
                    num_active_objs=num_active_objs,
                    max_combination_size=num_active_objs,
                    maximal_load=maximal_load,
                )
                P_model_list.append(P_ub)

            # if (
            #     (E_frac_of_demand_vectors_covered and E_frac_of_demand_vectors_covered < 0.01)
            #     or (P_ub and P_ub < 0.01)
            # ):
            #     break

        log(INFO, f"storage_design= {storage_design}",
            x_list=x_list,
            E_frac_of_demand_vectors_covered_list=E_frac_of_demand_vectors_covered_list,
            std_frac_of_demand_vectors_covered_list=std_frac_of_demand_vectors_covered_list,
            P_model_list=P_model_list,
        )

        color = next(dark_color_cycle)
        if run_sim:
            plot.errorbar(x_list, E_frac_of_demand_vectors_covered_list, yerr=std_frac_of_demand_vectors_covered_list, label=f"{storage_design.repr_for_plot()}, sim", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

        if plot_model:
            plot.plot(x_list, P_model_list, label=f"{storage_design.repr_for_plot()}", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    n = k
    use_cvxpy = True

    storage_design_and_model_list = [
        (
            design.ClusteringDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
            model.ClusteringDesignModelForGivenDemandDistribution(k=k, n=n, d=d)
        ),

        (
            design.CyclicDesign(k=k, n=n, d=d, shift_size=1, use_cvxpy=use_cvxpy),
            model.CyclicDesignModelForGivenDemandDistribution(k=k, n=n, d=d)
        ),

        # (
        #     design.RandomExpanderDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
        #     model.RandomDesignModelForGivenDemandDistribution(k=k, n=n, d=d)
        # ),

        (
            design.RandomBlockDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
            model.BlockDesignModelForGivenDemandDistribution(k=k, n=n, d=d)
        ),
    ]

    for storage_design, storage_model in storage_design_and_model_list:
        plot_(storage_design=storage_design, storage_model=storage_model)

    fontsize = 14
    plot.xlabel(x_label, fontsize=fontsize)


def plot_P(
    d_list: list[int],
    active_obj_demand_rv_list: list[random_variable.RandomVariable],
    func_active_obj_demand_to_x: Callable,
    x_label: str,
    demand_dist: str,
    maximal_load: float,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    k = 120

    for d in d_list:
        plot_P_for_given_params(
            k=k,
            d=d,
            active_obj_demand_rv_list=active_obj_demand_rv_list,
            func_active_obj_demand_to_x=func_active_obj_demand_to_x,
            x_label=x_label,
            maximal_load=1,
            run_sim=False,
            plot_model=True,
            num_samples=num_samples,
            num_sim_run=num_sim_run,
        )

    fontsize = 14
    plot.legend(fontsize=fontsize, loc="upper right", bbox_to_anchor=(1.25, 0.75))
    plot.ylabel(r"$\mathcal{P}$", fontsize=fontsize)

    plot.title(
        fr"$k= n= {k}$, "
        fr"$m= {maximal_load}$, "
        r"$\rho \sim {}$".format(demand_dist)
        # r"$N_{\textrm{sample}}= $" + fr"${num_samples}$, "
        # r"$N_{\textrm{sim}}= $" + fr"${num_sim_run}$"
    )

    # Save the plot
    plot.gcf().set_size_inches(6, 4)
    file_name = (
        "plots/plot_impact_of_demand_and_overlaps"
        + f"_k_{k}"
        # + f"_m_{maximal_load}"
        + f"_demand_dist_{demand_dist}"
        + ".pdf"
    )
    plot.savefig(file_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")


def manage_plot_P_w_joblib():
    log(INFO, "Started")

    @dataclasses.dataclass
    class Params:
        active_obj_demand_rv_list: list[random_variable.RandomVariable]
        func_active_obj_demand_to_x: Callable[[random_variable.RandomVariable], float]
        x_label: str
        demand_dist: str

    params_list = [
        # Params(
        #     active_obj_demand_rv_list=[
        #         random_variable.Exponential(mu=mu)
        #         for mu in numpy.linspace(0.1, 5, 30)
        #     ],
        #     func_active_obj_demand_to_x=lambda demand_rv: demand_rv.mu,
        #     x_label=r"$\mu$",
        #     demand_dist="Exp",
        # ),

        # Params(
        #     active_obj_demand_rv_list=[
        #         random_variable.Normal(mu=mu, sigma=mu)
        #         # for mu in numpy.linspace(0.1, 5, 30)
        #         for mu in numpy.linspace(0.1, 5, 5)
        #     ],
        #     func_active_obj_demand_to_x=lambda demand_rv: demand_rv.mu,
        #     x_label=r"$\mu = \sigma$",
        #     demand_dist="Normal",
        # ),
    ]

    # Pareto
    # _D = 1.5
    # for D in numpy.arange(_D, _D + 1 + 0.1, 0.1):
    #     params_list.append(
    #         Params(
    #             active_obj_demand_rv_list=[
    #                 random_variable.Pareto(loc=D, a=a)
    #                 for a in numpy.linspace(0.1, 10, 25)
    #                 # for a in range(1, 10)
    #             ],
    #             func_active_obj_demand_to_x=lambda demand_rv: demand_rv.a,
    #             x_label=r"$a$",
    #             demand_dist=f"Pareto(D={round(D, 1)})",
    #         ),
    #     )

    # Bernoulli
    # _D = 7
    # for D in numpy.arange(_D, _D + 1 + 0.1, 0.1):
    #     params_list.append(
    #         Params(
    #             active_obj_demand_rv_list=[
    #                 random_variable.Bernoulli(p=p, D=D)
    #                 for p in numpy.linspace(0.1, 1, 20)
    #             ],
    #             func_active_obj_demand_to_x=lambda demand_rv: demand_rv.p,
    #             x_label=r"$p$",
    #             demand_dist=f"Bern(D={round(D, 1)})",
    #         )
    #     )

    # Beta
    _D = 1.5
    for D in numpy.arange(_D, _D + 0 + 0.1, 0.1):
        params_list.append(
            Params(
                active_obj_demand_rv_list=[
                    random_variable.Beta(a=a, b=a, D=D)
                    for a in numpy.linspace(0.1, 1, 10)
                ],
                func_active_obj_demand_to_x=lambda demand_rv: demand_rv.a,
                x_label=r"$a = b$",
                demand_dist=f"Beta(D={round(D, 1)})",
            )
        )

    joblib.Parallel(n_jobs=-1, prefer="processes")(
        joblib.delayed(plot_P)(
            # d_list=[3],
            # d_list=[8, 10],
            # d_list=[2, 3],
            d_list=[2, 3, 4],
            # d_list=[5, 6],
            active_obj_demand_rv_list=params.active_obj_demand_rv_list,
            func_active_obj_demand_to_x=params.func_active_obj_demand_to_x,
            x_label=params.x_label,
            demand_dist=params.demand_dist,
            maximal_load=1,  # 0.7,
            num_samples=100,  # 300,
            # num_samples=1000,
            num_sim_run=3,
        )
        for params in params_list
    )

    log(INFO, "Done")


if __name__ == "__main__":
    manage_plot_P_w_joblib()