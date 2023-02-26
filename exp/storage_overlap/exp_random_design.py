import numpy

from src.storage_overlap import (
    design,
    sim,
)

from src.utils.debug import *
from src.utils.plot import *


def plot_frac_demand_vectors_covered_for_given_combination_size_vs_num_popular_objs(
    d: int,
    demand_for_popular: int,
    num_sample: int = 300,
    num_sim_run: int = 3,
):
    k = 120
    n = k
    block_design = design.RandomDesign(k=k, n=n, d=d)
    random_expander_design = design.RandomExpanderDesign(k=k, n=n, d=d)

    num_popular_obj_list = [2, 5, 10] + [int(k * frac) for frac in [0.1, 0.2, 0.3]]
    # num_popular_obj_list = [2, 5, 10] + [int(k * frac) for frac in [0.1, 0.6, 0.8]]

    log(INFO, "Started",
        block_design=block_design,
        random_expander_design=random_expander_design,
        num_popular_obj_list=num_popular_obj_list,
        demand_for_popular=demand_for_popular,
        num_sample=num_sample,
        num_sim_run=num_sim_run,
    )

    def plot_(combination_size: int):
        log(INFO, f">> combination_size= {combination_size}")

        E_frac_of_demand_vectors_covered_list_for_block_design = []
        std_frac_of_demand_vectors_covered_list_for_block_design = []
        E_frac_of_demand_vectors_covered_list_for_expander_design = []
        std_frac_of_demand_vectors_covered_list_for_expander_design = []
        # frac_of_demand_vectors_covered_lower_bound_list = []
        # frac_of_demand_vectors_covered_upper_bound_list = []

        for num_popular_obj in num_popular_obj_list:
            log(INFO, f"> num_popular_obj= {num_popular_obj}")

            # Run sim for `block_design`
            frac_of_demand_vectors_covered_list_for_block_design = [0]
            if (
                len(E_frac_of_demand_vectors_covered_list_for_block_design) == 0
                or E_frac_of_demand_vectors_covered_list_for_block_design[-1] > 0.01
            ):
                frac_of_demand_vectors_covered_list_for_block_design = sim.sim_frac_of_demand_vectors_covered(
                    storage_design=block_design,
                    num_popular_obj=num_popular_obj,
                    cum_demand=demand_for_popular * num_popular_obj,
                    zipf_tail_index=0,
                    num_sample=num_sample,
                    num_sim_run=num_sim_run,
                    combination_size_for_is_demand_vector_covered=combination_size,
                )
            E_frac_of_demand_vectors_covered_list_for_block_design.append(numpy.mean(frac_of_demand_vectors_covered_list_for_block_design))
            std_frac_of_demand_vectors_covered_list_for_block_design.append(numpy.std(frac_of_demand_vectors_covered_list_for_block_design))

            # Run sim for `random_expander_design`
            frac_of_demand_vectors_covered_list_for_expander_design = [0]
            if (
                len(E_frac_of_demand_vectors_covered_list_for_expander_design) == 0
                or E_frac_of_demand_vectors_covered_list_for_expander_design[-1] > 0.01
            ):
                frac_of_demand_vectors_covered_list_for_expander_design = sim.sim_frac_of_demand_vectors_covered(
                    storage_design=random_expander_design,
                    num_popular_obj=num_popular_obj,
                    cum_demand=demand_for_popular * num_popular_obj,
                    zipf_tail_index=0,
                    num_sample=num_sample,
                    num_sim_run=num_sim_run,
                    combination_size_for_is_demand_vector_covered=combination_size,
                )
            E_frac_of_demand_vectors_covered_list_for_expander_design.append(numpy.mean(frac_of_demand_vectors_covered_list_for_expander_design))
            std_frac_of_demand_vectors_covered_list_for_expander_design.append(numpy.std(frac_of_demand_vectors_covered_list_for_expander_design))

            # frac_of_demand_vectors_covered_lower_bound, frac_of_demand_vectors_covered_upper_bound = sim.sim_frac_of_demand_vectors_covered_lower_and_upper_bound(
            #     storage_design=replica_design,
            #     num_popular_obj=num_popular_obj,
            #     cum_demand=demand_for_popular * num_popular_obj,
            #     zipf_tail_index=0,
            #     num_sample=num_sample,
            #     num_sim_run=num_sim_run,
            # )
            # frac_of_demand_vectors_covered_lower_bound_list.append(frac_of_demand_vectors_covered_lower_bound)
            # frac_of_demand_vectors_covered_upper_bound_list.append(frac_of_demand_vectors_covered_upper_bound)

        log(INFO, "",
            num_popular_obj_list=num_popular_obj_list,
            E_frac_of_demand_vectors_covered_list_for_block_design=E_frac_of_demand_vectors_covered_list_for_block_design,
            std_frac_of_demand_vectors_covered_list_for_block_design=std_frac_of_demand_vectors_covered_list_for_block_design,
            E_frac_of_demand_vectors_covered_list_for_expander_design=E_frac_of_demand_vectors_covered_list_for_expander_design,
            std_frac_of_demand_vectors_covered_list_for_expander_design=std_frac_of_demand_vectors_covered_list_for_expander_design,
        )

        color = next(dark_color_cycle)
        plot.errorbar(num_popular_obj_list, E_frac_of_demand_vectors_covered_list_for_block_design, yerr=std_frac_of_demand_vectors_covered_list_for_block_design, label=f"{block_design.repr_for_plot()}", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
        plot.errorbar(num_popular_obj_list, E_frac_of_demand_vectors_covered_list_for_expander_design, yerr=std_frac_of_demand_vectors_covered_list_for_expander_design, label=f"{random_expander_design.repr_for_plot()}", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
        # plot.plot(num_popular_obj_list, frac_of_demand_vectors_covered_lower_bound_list, label=f"{replica_design.repr_for_plot()}-LB", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
        # plot.plot(num_popular_obj_list, frac_of_demand_vectors_covered_upper_bound_list, label=f"{replica_design.repr_for_plot()}-UB", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    for combination_size in range(2, d + 1):
        plot_(combination_size=combination_size)

    fontsize = 14
    plot.legend(fontsize=fontsize)
    # plot.yscale("log")
    plot.ylabel("Fraction of demand vectors covered", fontsize=fontsize)
    plot.xlabel("Number of popular objects", fontsize=fontsize)

    plot.title(
        fr"$d= {d}$, "
        r"$\lambda= $" + fr"${demand_for_popular}$, "
        r"$N_{\textrm{sample}}= $" + fr"${num_sample}$, "
        r"$N_{\textrm{sim}}= $" + fr"${num_sim_run}$"
    )

    # Save the plot
    plot.gcf().set_size_inches(8, 6)
    file_name = (
        "plots/plot_frac_demand_vectors_covered_for_given_combination_size_vs_num_popular_objs"
        + f"_k_{k}"
        + f"_d_{d}"
        + f"_lambda_{demand_for_popular}"
        + ".png"
    )
    plot.savefig(file_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")


def manage_plot_frac_demand_vectors_covered_for_given_combination_size_vs_num_popular_objs():
    log(INFO, "Started")

    def plot_(d: int):
        for demand_for_popular in range(1, d + 1):
            plot_frac_demand_vectors_covered_for_given_combination_size_vs_num_popular_objs(
                d=d,
                demand_for_popular=demand_for_popular,
                # num_sample=300,
                num_sample=1000,
                num_sim_run=3,
            )

    # plot_(d=2)
    plot_(d=3)
    # plot_(d=4)
    # plot_(d=5)
    # plot_(d=6)

    log(INFO, "Done")


def manage_plot_frac_demand_vectors_covered_for_given_combination_size_vs_num_popular_objs_w_joblib():
    log(INFO, "Started")

    joblib.Parallel(n_jobs=-1, prefer="processes")(
        joblib.delayed(plot_frac_demand_vectors_covered_for_given_combination_size_vs_num_popular_objs)(
            d=d,
            demand_for_popular=demand_for_popular,
            # num_sample=300,
            num_sample=1000,
            num_sim_run=3,
        )
        # for d in range(2, 3)
        # for d in range(3, 4)
        # for d in range(4, 5)
        for d in range(3, 5)
        # for d in range(2, 7)
        for demand_for_popular in range(2, d + 1)
    )

    log(INFO, "Done")


if __name__ == "__main__":
    # manage_plot_frac_demand_vectors_covered_for_given_combination_size_vs_num_popular_objs()
    manage_plot_frac_demand_vectors_covered_for_given_combination_size_vs_num_popular_objs_w_joblib()
