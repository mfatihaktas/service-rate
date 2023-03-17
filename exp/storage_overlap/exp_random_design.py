import joblib
import numpy

from src.storage_overlap import (
    design,
    sim,
)

from src.utils.debug import *
from src.utils.plot import *


COMBINATION_SIZE_TO_COLOR_MAP = {}


def plot_frac_demand_vectors_covered_for_given_combination_size_vs_num_popular_objs_for_storage_design(
    storage_design: design.StorageDesign,
    demand_for_popular: int,
    num_popular_obj_list: list[int],
    combination_size: int,
    num_sample: int = 300,
    num_sim_run: int = 3,
):
    log(INFO, "Started",
        storage_design=storage_design,
        demand_for_popular=demand_for_popular,
        num_popular_obj_list=num_popular_obj_list,
        combination_size=combination_size,
        num_sample=num_sample,
        num_sim_run=num_sim_run,
    )

    if combination_size not in COMBINATION_SIZE_TO_COLOR_MAP:
        COMBINATION_SIZE_TO_COLOR_MAP[combination_size] = next(dark_color_cycle)

    E_frac_of_demand_vectors_covered_list = []
    std_frac_of_demand_vectors_covered_list = []

    for num_popular_obj in num_popular_obj_list:
        log(INFO, f"> num_popular_obj= {num_popular_obj}")

        if (
            len(E_frac_of_demand_vectors_covered_list) > 0
            and E_frac_of_demand_vectors_covered_list[-1] <= 0.01
        ):
            frac_of_demand_vectors_covered_list = [0]

        else:
            frac_of_demand_vectors_covered_list = sim.sim_frac_of_demand_vectors_covered(
                storage_design=storage_design,
                num_popular_obj=num_popular_obj,
                cum_demand=demand_for_popular * num_popular_obj,
                zipf_tail_index=0,
                num_sample=num_sample,
                num_sim_run=num_sim_run,
                combination_size_for_is_demand_vector_covered=combination_size,
            )
        E_frac_of_demand_vectors_covered_list.append(numpy.mean(frac_of_demand_vectors_covered_list))
        std_frac_of_demand_vectors_covered_list.append(numpy.std(frac_of_demand_vectors_covered_list))

    log(INFO, "",
        storage_design=storage_design,
        demand_for_popular=demand_for_popular,
        num_popular_obj_list=num_popular_obj_list,
        E_frac_of_demand_vectors_covered_list=E_frac_of_demand_vectors_covered_list,
        std_frac_of_demand_vectors_covered_list=std_frac_of_demand_vectors_covered_list,
    )

    plot.errorbar(num_popular_obj_list, E_frac_of_demand_vectors_covered_list, yerr=std_frac_of_demand_vectors_covered_list, label=f"{storage_design.repr_for_plot()}, C={combination_size}", color=COMBINATION_SIZE_TO_COLOR_MAP[combination_size], marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    fontsize = 14
    plot.legend(fontsize=fontsize)
    # plot.yscale("log")
    plot.ylabel("Fraction of demand vectors covered", fontsize=fontsize)
    plot.xlabel("Number of popular objects", fontsize=fontsize)

    plot.title(
        fr"$d= {storage_design.d}$, "
        r"$\lambda= $" + fr"${demand_for_popular}$, "
        r"$N_{\textrm{sample}}= $" + fr"${num_sample}$, "
        r"$N_{\textrm{sim}}= $" + fr"${num_sim_run}$"
    )

    log(INFO, "Done")


def plot_frac_demand_vectors_covered_for_given_combination_size_vs_num_popular_objs(
    d: int,
    demand_for_popular: int,
    num_sample: int = 300,
    num_sim_run: int = 3,
):
    k = 120
    n = k
    use_cvxpy = False

    # num_popular_obj_list = [2, 5, 10] + [int(k * frac) for frac in [0.1, 0.15, 0.2, 0.25]]
    # num_popular_obj_list = [2, 5, 10] + [int(k * frac) for frac in [0.1, 0.6, 0.8]]
    num_popular_obj_list = [2, 5, 10, 20]
    # num_popular_obj_list = [2, 5]

    log(INFO, "Started",
        num_popular_obj_list=num_popular_obj_list,
        demand_for_popular=demand_for_popular,
        num_sample=num_sample,
        num_sim_run=num_sim_run,
    )

    storage_design_list = [
        # design.RandomBlockDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
        design.RandomExpanderDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
        # design.RandomExpanderDesign_wClusters(k=k, n=n, d=d, use_cvxpy=use_cvxpy, num_clusters=2),
    ]

    for storage_design in storage_design_list:
        for combination_size in range(2, d + 1):
            plot_frac_demand_vectors_covered_for_given_combination_size_vs_num_popular_objs_for_storage_design(
                storage_design=storage_design,
                demand_for_popular=demand_for_popular,
                num_popular_obj_list=num_popular_obj_list,
                combination_size=combination_size,
                num_sample=num_sample,
                num_sim_run=num_sim_run,
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
                num_sample=300,
                # num_sample=1000,
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
