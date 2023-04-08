import joblib
import numpy

from src.storage_overlap import (
    design,
    design_w_stripe,
    sim,
)

from src.utils.plot import *


def plot_frac_demand_vectors_covered_vs_tail_index():
    num_popular_obj = 2
    cum_demand = 3
    d = 3
    # zipf_tail_index_list = [0, 1, 2]
    # zipf_tail_index_list = [0, 0.5, 1, 1.5, 2]
    # zipf_tail_index_list = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
    zipf_tail_index_list = numpy.linspace(start=0, stop=4, num=20, endpoint=True)

    num_samples = 200
    num_sim_run = 3

    log(INFO, "Started",
        num_popular_obj=num_popular_obj,
        cum_demand=cum_demand,
        d=d,
        zipf_tail_index_list=zipf_tail_index_list,
        num_samples=num_samples,
        num_sim_run=num_sim_run,
    )

    def plot_(replica_design: design.ReplicaDesign):
        log(INFO, f">> replica_design= {replica_design}")

        E_frac_of_demand_vectors_covered_list = []
        std_frac_of_demand_vectors_covered_list = []

        for zipf_tail_index in zipf_tail_index_list:
            # With `get_demand_vectors_w_zipf_law`
            # demand_vector_list = demand.get_demand_vectors_w_zipf_law(
            #     num_objs=replica_design.k,
            #     num_popular_obj=num_popular_obj,
            #     cum_demand=cum_demand,
            #     zipf_tail_index=zipf_tail_index,
            # )
            # frac_of_demand_vectors_covered = replica_design.frac_of_demand_vectors_covered(
            #     demand_vector_list=demand_vector_list
            # )

            # With `frac_of_demand_vectors_covered_w_generator_input`
            # frac_of_demand_vectors_covered = replica_design.frac_of_demand_vectors_covered_w_generator_input(
            #     demand_vector_generator=demand.gen_demand_vector_w_zipf_law(
            #         num_objs=replica_design.k,
            #         num_popular_obj=num_popular_obj,
            #         cum_demand=cum_demand,
            #         zipf_tail_index=zipf_tail_index,
            #     )
            # )

            # With `sim_frac_of_demand_vectors_covered`
            frac_of_demand_vectors_covered_list = sim.sim_frac_of_demand_vectors_covered(
                storage_design=replica_design,
                num_popular_obj=num_popular_obj,
                cum_demand=cum_demand,
                zipf_tail_index=zipf_tail_index,
                num_samples=num_samples,
                num_sim_run=num_sim_run,
            )

            log(INFO, f"> zipf_tail_index= {zipf_tail_index}",
                frac_of_demand_vectors_covered_list=frac_of_demand_vectors_covered_list,
            )
            E_frac_of_demand_vectors_covered_list.append(numpy.mean(frac_of_demand_vectors_covered_list))
            std_frac_of_demand_vectors_covered_list.append(numpy.std(frac_of_demand_vectors_covered_list))

        log(INFO, f"replica_design= {replica_design}",
            zipf_tail_index_list=zipf_tail_index_list,
            E_frac_of_demand_vectors_covered_list=E_frac_of_demand_vectors_covered_list,
            std_frac_of_demand_vectors_covered_list=std_frac_of_demand_vectors_covered_list,
        )
        plot.errorbar(zipf_tail_index_list, E_frac_of_demand_vectors_covered_list, yerr=std_frac_of_demand_vectors_covered_list, color=next(dark_color_cycle), label=f"{replica_design.repr_for_plot()}", marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    # k = 21
    # k = 111
    k = 45
    n = k
    replica_design_list = [
        design.ClusteringDesign(k=k, n=n, d=d),
        design.CyclicDesign(k=k, n=n, d=d, shift_size=1),
        design.CyclicDesign(k=k, n=n, d=d, shift_size=2),
        design.CyclicDesign(k=k, n=n, d=d, shift_size=3),
    ]

    for replica_design in replica_design_list:
        plot_(replica_design=replica_design)

    fontsize = 14
    plot.legend(fontsize=fontsize)
    # plot.yscale("log")
    plot.ylabel("Fraction of demand vectors covered", fontsize=fontsize)
    plot.xlabel("Popularity skew index", fontsize=fontsize)

    plot.title(
        f"$D= {cum_demand}$"
        + r", $N_{\textrm{pop}}= $" + fr"${num_popular_obj}$"
        + r", $N_{\textrm{sample}}= $" + fr"${num_samples}$"
        + r", $N_{\textrm{sim}}= $" + fr"${num_sim_run}$"
    )

    # Save the plot
    plot.gcf().set_size_inches(8, 6)
    file_name = (
        "plots/plot_frac_demand_vectors_covered_vs_tail_index"
        + f"_k_{k}"
        + f"_d_{d}"
        + f"_D_{cum_demand}"
        + f"_Np_{num_popular_obj}"
        + ".png"
    )
    plot.savefig(file_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")


def plot_frac_demand_vectors_covered_vs_num_popular_objs(
    d: int,
    demand_for_popular: int,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    num_popular_obj_list = list(range(1, 10))

    log(INFO, "Started",
        d=d,
        num_popular_obj_list=num_popular_obj_list,
        demand_for_popular=demand_for_popular,
        num_samples=num_samples,
        num_sim_run=num_sim_run,
    )

    def plot_(replica_design: design.ReplicaDesign):
        log(INFO, f">> replica_design= {replica_design}")

        E_frac_of_demand_vectors_covered_list = []
        std_frac_of_demand_vectors_covered_list = []
        # E_frac_of_demand_vectors_covered_list_w_combination_size_2 = []
        # std_frac_of_demand_vectors_covered_list_w_combination_size_2 = []
        # E_frac_of_demand_vectors_covered_list_w_combination_size_3 = []
        # std_frac_of_demand_vectors_covered_list_w_combination_size_3 = []
        # frac_of_demand_vectors_covered_lower_bound_list = []
        # frac_of_demand_vectors_covered_upper_bound_list = []

        for num_popular_obj in num_popular_obj_list:
            log(INFO, f"> num_popular_obj= {num_popular_obj}")

            frac_of_demand_vectors_covered_list = sim.sim_frac_of_demand_vectors_covered(
                storage_design=replica_design,
                num_popular_obj=num_popular_obj,
                cum_demand=demand_for_popular * num_popular_obj,
                zipf_tail_index=0,
                num_samples=num_samples,
                num_sim_run=num_sim_run,
            )
            E_frac_of_demand_vectors_covered_list.append(numpy.mean(frac_of_demand_vectors_covered_list))
            std_frac_of_demand_vectors_covered_list.append(numpy.std(frac_of_demand_vectors_covered_list))

            # frac_of_demand_vectors_covered_lower_bound, frac_of_demand_vectors_covered_upper_bound = sim.sim_frac_of_demand_vectors_covered_lower_and_upper_bound(
            #     storage_design=replica_design,
            #     num_popular_obj=num_popular_obj,
            #     cum_demand=demand_for_popular * num_popular_obj,
            #     zipf_tail_index=0,
            #     num_samples=num_samples,
            #     num_sim_run=num_sim_run,
            # )
            # frac_of_demand_vectors_covered_lower_bound_list.append(frac_of_demand_vectors_covered_lower_bound)
            # frac_of_demand_vectors_covered_upper_bound_list.append(frac_of_demand_vectors_covered_upper_bound)

            # frac_of_demand_vectors_covered_list_w_combination_size_2 = sim.sim_frac_of_demand_vectors_covered(
            #     storage_design=replica_design,
            #     num_popular_obj=num_popular_obj,
            #     cum_demand=demand_for_popular * num_popular_obj,
            #     zipf_tail_index=0,
            #     num_samples=num_samples,
            #     num_sim_run=num_sim_run,
            #     combination_size_for_is_demand_vector_covered=2,
            # )
            # E_frac_of_demand_vectors_covered_list_w_combination_size_2.append(numpy.mean(frac_of_demand_vectors_covered_list_w_combination_size_2))
            # std_frac_of_demand_vectors_covered_list_w_combination_size_2.append(numpy.std(frac_of_demand_vectors_covered_list_w_combination_size_2))

            # frac_of_demand_vectors_covered_list_w_combination_size_3 = sim.sim_frac_of_demand_vectors_covered(
            #     storage_design=replica_design,
            #     num_popular_obj=num_popular_obj,
            #     cum_demand=demand_for_popular * num_popular_obj,
            #     zipf_tail_index=0,
            #     num_samples=num_samples,
            #     num_sim_run=num_sim_run,
            #     combination_size_for_is_demand_vector_covered=3,
            # )
            # E_frac_of_demand_vectors_covered_list_w_combination_size_3.append(numpy.mean(frac_of_demand_vectors_covered_list_w_combination_size_3))
            # std_frac_of_demand_vectors_covered_list_w_combination_size_3.append(numpy.std(frac_of_demand_vectors_covered_list_w_combination_size_3))

        log(INFO, f"replica_design= {replica_design}",
            num_popular_obj_list=num_popular_obj_list,
            E_frac_of_demand_vectors_covered_list=E_frac_of_demand_vectors_covered_list,
            std_frac_of_demand_vectors_covered_list=std_frac_of_demand_vectors_covered_list,
            # E_frac_of_demand_vectors_covered_list_w_combination_size_2=E_frac_of_demand_vectors_covered_list_w_combination_size_2,
            # std_frac_of_demand_vectors_covered_list_w_combination_size_2=std_frac_of_demand_vectors_covered_list_w_combination_size_2,
            # E_frac_of_demand_vectors_covered_list_w_combination_size_3=E_frac_of_demand_vectors_covered_list_w_combination_size_3,
            # std_frac_of_demand_vectors_covered_list_w_combination_size_3=std_frac_of_demand_vectors_covered_list_w_combination_size_3,
            # frac_of_demand_vectors_covered_lower_bound_list=frac_of_demand_vectors_covered_lower_bound_list,
        )

        color = next(dark_color_cycle)
        plot.errorbar(num_popular_obj_list, E_frac_of_demand_vectors_covered_list, yerr=std_frac_of_demand_vectors_covered_list, label=f"{replica_design.repr_for_plot()}", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
        # plot.errorbar(num_popular_obj_list, E_frac_of_demand_vectors_covered_list_w_combination_size_2, yerr=std_frac_of_demand_vectors_covered_list_w_combination_size_2, label=f"{replica_design.repr_for_plot()}, C=2", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
        # plot.errorbar(num_popular_obj_list, E_frac_of_demand_vectors_covered_list_w_combination_size_3, yerr=std_frac_of_demand_vectors_covered_list_w_combination_size_3, label=f"{replica_design.repr_for_plot()}, C=3", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
        # plot.plot(num_popular_obj_list, frac_of_demand_vectors_covered_lower_bound_list, label=f"{replica_design.repr_for_plot()}-LB", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
        # plot.plot(num_popular_obj_list, frac_of_demand_vectors_covered_upper_bound_list, label=f"{replica_design.repr_for_plot()}-UB", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    # k = 45
    # k = 111
    k = 120
    # k = 200
    # k = 24
    n = k
    use_cvxpy = False
    replica_design_list = [
        design.ClusteringDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
        # design_w_stripe.ClusteringDesignWithStripe(k=k, n=n, d=d, s=2, use_cvxpy=use_cvxpy),
        design.CyclicDesign(k=k, n=n, d=d, shift_size=1, use_cvxpy=use_cvxpy),
        # design.CyclicDesign(k=k, n=n, d=d, shift_size=2, use_cvxpy=use_cvxpy),
        # design.CyclicDesign(k=k, n=n, d=d, shift_size=3, use_cvxpy=use_cvxpy),
        # design.RandomBlockDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
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

    for replica_design in replica_design_list:
        plot_(replica_design=replica_design)

    fontsize = 14
    plot.legend(fontsize=fontsize)
    # plot.yscale("log")
    plot.ylabel(r"$\mathcal{P}_{m, \lambda}$", fontsize=fontsize)
    plot.xlabel(r"$m$", fontsize=fontsize)

    plot.title(
        fr"$d= {d}$, "
        fr"$\lambda= {demand_for_popular}$, "
        r"$N_{\textrm{sample}}= $" + fr"${num_samples}$, "
        r"$N_{\textrm{sim}}= $" + fr"${num_sim_run}$"
    )

    # Save the plot
    plot.gcf().set_size_inches(8, 6)
    file_name = (
        "plots/plot_frac_demand_vectors_covered_vs_num_popular_objs"
        + f"_k_{k}"
        + f"_d_{d}"
        + f"_lambda_{demand_for_popular}"
        + ".png"
    )
    plot.savefig(file_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")


def manage_plot_frac_demand_vectors_covered_vs_num_popular_objs():
    log(INFO, "Started")

    def plot_(d: int):
        for demand_for_popular in range(1, d + 1):
            plot_frac_demand_vectors_covered_vs_num_popular_objs(
                d=d,
                demand_for_popular=demand_for_popular,
                num_samples=300,
                num_sim_run=3,
            )

    # plot_(d=2)
    # plot_(d=3)
    plot_(d=4)
    # plot_(d=5)
    # plot_(d=6)

    log(INFO, "Done")


def manage_plot_frac_demand_vectors_covered_vs_num_popular_objs_w_joblib():
    log(INFO, "Started")

    joblib.Parallel(n_jobs=-1, prefer="processes")(
        joblib.delayed(plot_frac_demand_vectors_covered_vs_num_popular_objs)(
            d=d,
            demand_for_popular=demand_for_popular,
            num_samples=300,
            # num_samples=1000,
            num_sim_run=3,
        )
        for d in range(2, 7)
        for demand_for_popular in range(2, d + 1)
        # for d in range(2, 7)
        # for demand_for_popular in range(2, d + 1)
        # for d in [4]
        # for demand_for_popular in [d - 1]
    )

    log(INFO, "Done")


if __name__ == "__main__":
    # plot_frac_demand_vectors_covered_vs_tail_index()
    # manage_plot_frac_demand_vectors_covered_vs_num_popular_objs()
    manage_plot_frac_demand_vectors_covered_vs_num_popular_objs_w_joblib()
