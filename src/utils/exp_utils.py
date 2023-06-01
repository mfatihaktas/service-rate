import enum
import numpy

from src.storage_overlap import (
    design,
    model,
    sim,
)
from src.model import demand
from src.sim import random_variable

from src.utils.plot import *


class DemandDistribution(enum.Enum):
    Bernoulli = "Bernoulli"
    Exp = "Exp"
    Pareto = "Pareto"


def plot_P_for_given_params(
    k: int,
    d: int,
    num_active_objs: int,
    maximal_load: float,
    demand_dist: DemandDistribution,
    plot_ub: bool = True,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    log(INFO, "Started",
        k=k,
        d=d,
        num_active_objs=num_active_objs,
        maximal_load=maximal_load,
        demand_dist=demand_dist,
        plot_ub=plot_ub,
        num_samples=num_samples,
        num_sim_run=num_sim_run,
    )

    if demand_dist == DemandDistribution.Bernoulli:
        # D = 2
        D = 3
        p_l = numpy.linspace(0.1, 0.8, 10)
        active_obj_demand_rv_list = [
            random_variable.Bernoulli(p=p, D=D) for p in p_l
        ]

        x_l = p_l
        xlabel = r"$p$"
        dist_in_title = fr"{D}" + r" \times \mathrm{Bernoulli(p)}"

    elif demand_dist == DemandDistribution.Exp:
        mu_list = numpy.linspace(0.1, 4, 15)
        active_obj_demand_rv_list = [
            random_variable.Exponential(mu=mu) for mu in mu_list

        ]
        x_l = mu_list
        xlabel = r"$\mu$"
        dist_in_title = r"\mathrm{Exp}(\mu)"

    elif demand_dist == DemandDistribution.Pareto:
        min_value = 0.1
        tail_index_list = numpy.linspace(0.1, 3, 10)
        active_obj_demand_rv_list = [
            random_variable.Pareto(loc=min_value, a=tail_index) for tail_index in tail_index_list
        ]

        x_l = tail_index_list
        xlabel = r"$\alpha$"
        dist_in_title = r"\mathrm{Pareto}" + fr"(\lambda={min_value}, \alpha)"

    def plot_(
        storage_design: design.CyclicDesign,
        storage_model: model.CyclicDesignModelForGivenDemandDistribution,
        run_sim: bool = False,
    ):
        log(INFO, f">> storage_design= {storage_design}")

        E_frac_of_demand_vectors_covered_list = []
        std_frac_of_demand_vectors_covered_list = []

        P_ub_list = []

        for active_obj_demand_rv in active_obj_demand_rv_list:
            log(INFO, f"> active_obj_demand_rv= {active_obj_demand_rv}")

            demand_vector_sampler = demand.DemandVectorSamplerWithFixedNumActiveObjs(
                num_objs=storage_design.k,
                num_active_objs=num_active_objs,
                active_obj_demand_rv=active_obj_demand_rv,
            )

            # Sim
            E_frac_of_demand_vectors_covered = 0.02
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
            if plot_ub:
                P_ub = storage_model.prob_serving_upper_bound(
                    demand_rv=active_obj_demand_rv,
                    num_active_objs=num_active_objs,
                    # max_combination_size=2,
                    max_combination_size=num_active_objs,
                    maximal_load=maximal_load,
                )
                P_ub_list.append(P_ub)

            # if E_frac_of_demand_vectors_covered <= 0.01 and P_ub <= 0.01:
            #     break

        log(INFO, f"storage_design= {storage_design}",
            x_l=x_l,
            E_frac_of_demand_vectors_covered_list=E_frac_of_demand_vectors_covered_list,
            std_frac_of_demand_vectors_covered_list=std_frac_of_demand_vectors_covered_list,
            P_ub_list=P_ub_list,
        )

        color = next(dark_color_cycle)
        if run_sim:
            plot.errorbar(x_l[:len(E_frac_of_demand_vectors_covered_list)], E_frac_of_demand_vectors_covered_list, yerr=std_frac_of_demand_vectors_covered_list, label=f"{storage_design.repr_for_plot()}", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
        if plot_ub:
            plot.plot(x_l[:len(P_ub_list)], P_ub_list, label=f"{storage_design.repr_for_plot()}, UB", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

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

        (
            design.RandomExpanderDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
            model.RandomDesignModelForGivenDemandDistribution(k=k, n=n, d=d)
        ),

        (
            design.RandomBlockDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
            model.BlockDesignModelForGivenDemandDistribution(k=k, n=n, d=d)
        ),
    ]

    run_sim = True
    for storage_design, storage_model in storage_design_and_model_list:
        plot_(storage_design=storage_design, storage_model=storage_model, run_sim=run_sim)

    fontsize = 16
    plot.xlabel(xlabel, fontsize=fontsize)
    plot.title(
        (
            fr"$k= n= {k}$, "
            fr"$d= {d}$, "
            fr"$m= {maximal_load}$, "
            # r"$n_{\textrm{active}}= $" + fr"${num_active_objs}$, "
            fr"$\rho \sim {dist_in_title}$"
            # r"$N_{\textrm{sample}}= $" + fr"${num_samples}$, "
            # r"$N_{\textrm{sim}}= $" + fr"${num_sim_run}$",
        ),
        fontsize=fontsize,
    )
