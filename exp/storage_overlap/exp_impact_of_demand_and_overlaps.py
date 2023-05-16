import joblib
import numpy

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
    active_obj_demand_rv: random_variable.RandomVariable,
    maximal_load: float,
    run_sim: bool = False,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    log(INFO, "Started",
        k=k,
        d=d,
        active_obj_demand_rv=active_obj_demand_rv,
        maximal_load=maximal_load,
        run_sim=run_sim,
        num_samples=num_samples,
        num_sim_run=num_sim_run,
    )

    def plot_(
        storage_design: design.CyclicDesign,
        storage_model: model.CyclicDesignModelForGivenDemandDistribution,
    ):
        log(INFO, f">> storage_design= {storage_design}")

        num_active_objs_list = []

        E_frac_of_demand_vectors_covered_list = []
        std_frac_of_demand_vectors_covered_list = []

        P_ub_list = []
        P_lb_list = []

        # for num_active_objs in [2]:
        # for num_active_objs in range(2, k // 2):
        # for num_active_objs in range(2, k // 10):
        # for num_active_objs in range(2, 6):
        # for num_active_objs in range(2, 11):
        for num_active_objs in range(2, 31):
            log(INFO, f"> num_active_objs= {num_active_objs}")

            num_active_objs_list.append(num_active_objs)

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
            P_ub = storage_model.prob_serving_upper_bound(
                demand_rv=active_obj_demand_rv,
                num_active_objs=num_active_objs,
                # max_combination_size=2,
                # max_combination_size=storage_design.d,
                max_combination_size=num_active_objs,
                maximal_load=maximal_load,
            )
            P_ub_list.append(P_ub)

            P_lb = storage_model.prob_serving_upper_bound(
                demand_rv=active_obj_demand_rv,
                num_active_objs=num_active_objs,
                max_combination_size=num_active_objs,
                maximal_load=maximal_load,
            )
            P_lb_list.append(P_lb)

            if (
                (E_frac_of_demand_vectors_covered and E_frac_of_demand_vectors_covered < 0.01)
                or (P_ub and P_ub < 0.01)
            ):
                break

        log(INFO, f"storage_design= {storage_design}",
            num_active_objs_list=num_active_objs_list,
            E_frac_of_demand_vectors_covered_list=E_frac_of_demand_vectors_covered_list,
            std_frac_of_demand_vectors_covered_list=std_frac_of_demand_vectors_covered_list,
            P_ub_list=P_ub_list,
            P_lb_list=P_lb_list,
        )

        color = next(dark_color_cycle)
        if run_sim:
            plot.errorbar(num_active_objs_list, E_frac_of_demand_vectors_covered_list, yerr=std_frac_of_demand_vectors_covered_list, label=f"{storage_design.repr_for_plot()}", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

        plot.plot(num_active_objs_list, P_ub_list, label=f"{storage_design.repr_for_plot()}, UB", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
        plot.plot(num_active_objs_list, P_lb_list, label=f"{storage_design.repr_for_plot()}, LB", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    n = k
    use_cvxpy = True

    storage_design_and_model_list = [
        # (
        #     design.ClusteringDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
        #     model.ClusteringDesignModelForGivenDemandDistribution(k=k, n=n, d=d)
        # ),

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


def plot_P(
    d_list: list[int],
    active_obj_demand: float,
    maximal_load: float,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    k = 120

    active_obj_demand_rv = random_variable.Bernoulli(p=0.5, D=active_obj_demand)
    # active_obj_demand_rv = random_variable.Constant(value=active_obj_demand)
    # active_obj_demand_rv = random_variable.Exponential(mu=1 / active_obj_demand)
    # active_obj_demand_rv = random_variable.Pareto(loc=active_obj_demand, a=3)

    run_sim = False
    for d in d_list:
        plot_P_for_given_params(
            k=k,
            d=d,
            active_obj_demand_rv=active_obj_demand_rv,
            maximal_load=maximal_load,
            run_sim=run_sim,
            num_samples=num_samples,
            num_sim_run=num_sim_run,
        )

    fontsize = 14
    plot.legend(fontsize=fontsize, loc="upper right", bbox_to_anchor=(1.25, 0.75))
    plot.ylabel(r"$\mathcal{P}$", fontsize=fontsize)
    # plot.xlabel(r"$n_{\textrm{active}}$", fontsize=fontsize)
    plot.xlabel("Number of active objects", fontsize=fontsize)

    plot.title(
        fr"$k= n= {k}$, "
        fr"$m= {maximal_load}$, "
        fr"$\rho \sim {active_obj_demand_rv.to_latex()}$"
        # r"$N_{\textrm{sample}}= $" + fr"${num_samples}$, "
        # r"$N_{\textrm{sim}}= $" + fr"${num_sim_run}$"
    )

    # Save the plot
    plot.gcf().set_size_inches(6, 4)
    file_name = (
        "plots/plot_impact_of_demand_and_overlaps"
        + f"_k_{k}"
        + f"_m_{maximal_load}"
        + f"_active_obj_demand_{active_obj_demand}"
        + f"_demand_dist_{active_obj_demand_rv.to_short_repr()}"
        + ".pdf"
    )
    plot.savefig(file_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")


def manage_plot_P_w_joblib():
    log(INFO, "Started")

    joblib.Parallel(n_jobs=-1, prefer="processes")(
        joblib.delayed(plot_P)(
            # d_list=[2, 3, 4],
            d_list=[2, 3],
            # d_list=[5, 6],
            # d_list=[3],
            active_obj_demand=active_obj_demand,
            maximal_load=1,  # 0.7,
            num_samples=100,  # 300,
            # num_samples=1000,
            num_sim_run=3,
        )
        # for active_obj_demand in [1.5, 2]
        for active_obj_demand in [2, 3]
        # for active_obj_demand in [2]
        # for active_obj_demand in [3]
    )

    log(INFO, "Done")


if __name__ == "__main__":
    manage_plot_P_w_joblib()
