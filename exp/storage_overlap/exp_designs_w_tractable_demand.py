import joblib
import numpy

from src.storage_overlap import (
    design,
    design_w_stripe,
    sim,
)
from src.model import demand

from src.utils.plot import *


def plot_P_for_given_params(
    k: int,
    d: int,
    demand_for_active_obj: float,
    maximal_load: float,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    log(INFO, "Started",
        k=k,
        d=d,
        demand_for_active_obj=demand_for_active_obj,
        maximal_load=maximal_load,
        num_samples=num_samples,
        num_sim_run=num_sim_run,
    )

    def plot_(storage_design: design.ReplicaDesign):
        log(INFO, f">> storage_design= {storage_design}")

        num_active_objs_list = []

        E_frac_of_demand_vectors_covered_list = []
        std_frac_of_demand_vectors_covered_list = []

        E_P_ub_w_combination_size_2_list = []
        std_P_ub_w_combination_size_2_list = []

        E_P_ub_w_combination_size_3_list = []
        std_P_ub_w_combination_size_3_list = []

        E_P_lb_list = []
        std_P_lb_list = []

        for num_active_objs in range(1, storage_design.k):
            log(INFO, f"> num_active_objs= {num_active_objs}")

            num_active_objs_list.append(num_active_objs)

            demand_vector_sampler = demand.DemandVectorSamplerWithFixedActiveObjDemand(
                num_objs=storage_design.k,
                num_active_objs=num_active_objs,
                demand_for_active_obj=demand_for_active_obj,
            )

            # Sim
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

            # UB, with combination_size = 2
            sim_P_ub_list = sim.sim_frac_of_demand_vectors_covered(
                demand_vector_sampler=demand_vector_sampler,
                storage_design=storage_design,
                num_samples=num_samples,
                num_sim_run=num_sim_run,
                combination_size_for_is_demand_vector_covered=2,
                maximal_load=maximal_load,
            )

            E_P_ub = numpy.mean(sim_P_ub_list)
            E_P_ub_w_combination_size_2_list.append(E_P_ub)
            std_P_ub_w_combination_size_2_list.append(numpy.std(sim_P_ub_list))

            # UB, with combination_size = 3
            sim_P_ub_list = sim.sim_frac_of_demand_vectors_covered(
                demand_vector_sampler=demand_vector_sampler,
                storage_design=storage_design,
                num_samples=num_samples,
                num_sim_run=num_sim_run,
                combination_size_for_is_demand_vector_covered=3,
                maximal_load=maximal_load,
            )

            E_P_ub = numpy.mean(sim_P_ub_list)
            E_P_ub_w_combination_size_3_list.append(E_P_ub)
            std_P_ub_w_combination_size_3_list.append(numpy.std(sim_P_ub_list))

            # LB
            # sim_P_lb_list = sim.sim_frac_of_demand_vectors_covered(
            #     demand_vector_sampler=demand_vector_sampler,
            #     storage_design=storage_design,
            #     num_samples=num_samples,
            #     num_sim_run=num_sim_run,
            #     # split_obj_demands_evenly_across_choices=True,
            #     assign_obj_demands_to_leftmost_choice_first=True,
            #     maximal_load=maximal_load,
            # )

            # E_P_lb = numpy.mean(sim_P_lb_list)
            # E_P_lb_list.append(E_P_lb)
            # std_P_lb_list.append(numpy.std(sim_P_lb_list))

            if E_frac_of_demand_vectors_covered < 0.01:
                break

        log(INFO, f"storage_design= {storage_design}",
            num_active_objs_list=num_active_objs_list,
            E_frac_of_demand_vectors_covered_list=E_frac_of_demand_vectors_covered_list,
            std_frac_of_demand_vectors_covered_list=std_frac_of_demand_vectors_covered_list,
        )

        color = next(dark_color_cycle)
        plot.errorbar(num_active_objs_list, E_frac_of_demand_vectors_covered_list, yerr=std_frac_of_demand_vectors_covered_list, label=f"{storage_design.repr_for_plot()}", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
        # plot.errorbar(num_active_objs_list, E_P_lb_list, yerr=std_P_lb_list, label=f"{storage_design.repr_for_plot()}, LB", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
        plot.errorbar(num_active_objs_list, E_P_ub_w_combination_size_2_list, yerr=std_P_ub_w_combination_size_2_list, label=f"{storage_design.repr_for_plot()}, UB-2", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
        plot.errorbar(num_active_objs_list, E_P_ub_w_combination_size_3_list, yerr=std_P_ub_w_combination_size_3_list, label=f"{storage_design.repr_for_plot()}, UB-3", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    n = k
    use_cvxpy = True  # False

    replica_design_list = [
        design.ClusteringDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
        # design_w_stripe.ClusteringDesignWithStripe(k=k, n=n, d=d, s=2, use_cvxpy=use_cvxpy),
        design.CyclicDesign(k=k, n=n, d=d, shift_size=1, use_cvxpy=use_cvxpy),
        # design.CyclicDesign(k=k, n=n, d=d, shift_size=2, use_cvxpy=use_cvxpy),
        # design.CyclicDesign(k=k, n=n, d=d, shift_size=3, use_cvxpy=use_cvxpy),
        design.RandomBlockDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
        design.RandomExpanderDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
        # design.RandomExpanderDesign_wClusters(k=k, n=n, d=d, num_clusters=2, use_cvxpy=use_cvxpy),
        # design.TwoXORDesign(k=124, n=124, d=d, use_cvxpy=use_cvxpy),
    ]

    # for s in [2, 3, 4]:
    # for s in [2, 3]:
    #     if s <= d:
    #         replica_design_list.append(
    #             # design_w_stripe.RandomBlockDesignWithStripe(k=k, n=n, d=d, s=s, use_cvxpy=True),
    #             design_w_stripe.RandomExpanderDesignWithStripe(k=k, n=n, d=d, s=s, use_cvxpy=True)
    #         )

    for storage_design in replica_design_list:
        plot_(storage_design=storage_design)


def plot_P(
    d: int,
    demand_for_active_obj: float,
    maximal_load: float,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    k = 120
    # k = 6

    plot_P_for_given_params(
        k=k,
        d=d,
        demand_for_active_obj=demand_for_active_obj,
        maximal_load=maximal_load,
        num_samples=num_samples,
        num_sim_run=num_sim_run,
    )

    fontsize = 14
    plot.legend(fontsize=fontsize, loc="upper right", bbox_to_anchor=(1.25, 0.75))
    # plot.yscale("log")
    plot.ylabel(r"$\mathcal{P}$", fontsize=fontsize)
    plot.xlabel("Number of active objects", fontsize=fontsize)

    plot.title(
        # fr"$d= {d}$, "
        fr"$k= n= {k}$, "
        fr"$m= {maximal_load}$, "
        # r"$n_\mathrm{active}= $" + fr"${num_active_objs}$, "
        fr"$\lambda= {demand_for_active_obj}$"
        # r"$\rho \sim$ Exp"
        # fr"$\rho \sim {demand_for_active_obj} \times$ Bernoulli"
        # r"$N_{\textrm{sample}}= $" + fr"${num_samples}$, "
        # r"$N_{\textrm{sim}}= $" + fr"${num_sim_run}$"
    )

    # Save the plot
    # plot.gcf().set_size_inches(8, 6)
    plot.gcf().set_size_inches(6, 4)
    file_name = (
        "plots/plot_P_tractable_demand"
        + f"_k_{k}"
        + f"_d_{d}"
        + f"_m_{maximal_load}"
        + ".pdf"
    )
    plot.savefig(file_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")


def manage_plot_P_w_joblib():
    log(INFO, "Started")

    joblib.Parallel(n_jobs=-1, prefer="processes")(
        joblib.delayed(plot_P)(
            d=d,
            demand_for_active_obj=demand_for_active_obj,
            maximal_load=1,  # 0.7,
            num_samples=100,  # 300,
            # num_samples=1000,
            num_sim_run=3,
        )
        # for d in [2, 3]
        for d in [3, 4]
        # for d in range(2, 7)
        # for d in [3, 4]
        # for demand_for_active_obj in [d - 1]
        for demand_for_active_obj in [3 * d / 4]
    )

    log(INFO, "Done")


if __name__ == "__main__":
    manage_plot_P_w_joblib()
