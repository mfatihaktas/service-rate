import joblib
import numpy

from src.model import storage_overlap as storage_overlap_model
from src.storage_overlap import (
    design,
    sim,
)

from src.utils.debug import *
from src.utils.plot import *


def plot_frac_demand_vectors_covered_vs_num_popular_objs_for_storage_design(
    storage_design: design.StorageDesign,
    storage_design_model: storage_overlap_model.StorageDesignModel,
    demand_for_popular: int,
    num_popular_obj_list: list[int],
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    log(INFO, "Started",
        storage_design=storage_design,
        storage_design_model=storage_design_model,
        demand_for_popular=demand_for_popular,
        num_popular_obj_list=num_popular_obj_list,
        num_samples=num_samples,
        num_sim_run=num_sim_run,
    )

    E_frac_of_demand_vectors_covered_list = []
    std_frac_of_demand_vectors_covered_list = []

    frac_of_demand_vectors_covered_upper_bound_list = []
    frac_of_demand_vectors_covered_lower_bound_list = []
    frac_of_demand_vectors_covered_lower_bound_power_d_list = []

    for num_popular_obj in num_popular_obj_list:
        log(INFO, f"> num_popular_obj= {num_popular_obj}")

        if run_sim:
            frac_of_demand_vectors_covered_list = sim.sim_frac_of_demand_vectors_covered(
                storage_design=storage_design,
                num_popular_obj=num_popular_obj,
                cum_demand=demand_for_popular * num_popular_obj,
                zipf_tail_index=0,
                num_samples=num_samples,
                num_sim_run=num_sim_run,
            )

            E_frac_of_demand_vectors_covered_list.append(numpy.mean(frac_of_demand_vectors_covered_list))
            std_frac_of_demand_vectors_covered_list.append(numpy.std(frac_of_demand_vectors_covered_list))

        frac_of_demand_vectors_covered_lower_bound = storage_design_model.prob_serving_lower_bound(m=num_popular_obj, lambda_=demand_for_popular)
        # frac_of_demand_vectors_covered_lower_bound = storage_design_model.wrong_prob_serving_lower_bound(m=num_popular_obj, lambda_=demand_for_popular)
        frac_of_demand_vectors_covered_lower_bound_list.append(frac_of_demand_vectors_covered_lower_bound)
        frac_of_demand_vectors_covered_lower_bound_power_d_list.append(frac_of_demand_vectors_covered_lower_bound**(storage_design.d))

        frac_of_demand_vectors_covered_upper_bound = storage_design_model.prob_serving_upper_bound(m=num_popular_obj, lambda_=demand_for_popular)
        frac_of_demand_vectors_covered_upper_bound_list.append(frac_of_demand_vectors_covered_upper_bound)

        if frac_of_demand_vectors_covered_upper_bound < 0.01:
            log(WARNING, "Early break", prob_obj_is_active=prob_obj_is_active)
            break

    log(INFO, "",
        storage_design=storage_design,
        demand_for_popular=demand_for_popular,
        num_popular_obj_list=num_popular_obj_list,
        E_frac_of_demand_vectors_covered_list=E_frac_of_demand_vectors_covered_list,
        std_frac_of_demand_vectors_covered_list=std_frac_of_demand_vectors_covered_list,
        frac_of_demand_vectors_covered_lower_bound_list=frac_of_demand_vectors_covered_lower_bound_list,
        frac_of_demand_vectors_covered_upper_bound_list=frac_of_demand_vectors_covered_upper_bound_list,
    )

    color = next(dark_color_cycle)
    if run_sim:
        plot.errorbar(num_popular_obj_list, E_frac_of_demand_vectors_covered_list, yerr=std_frac_of_demand_vectors_covered_list, label=f"{storage_design.repr_for_plot()}, Sim", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
    plot.plot(num_popular_obj_list, frac_of_demand_vectors_covered_lower_bound_list, label=f"{storage_design.repr_for_plot()}, LB", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
    plot.plot(num_popular_obj_list, frac_of_demand_vectors_covered_upper_bound_list, label=f"{storage_design.repr_for_plot()}, UB", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
    # plot.plot(num_popular_obj_list, frac_of_demand_vectors_covered_lower_bound_power_d_list, label=f"{storage_design.repr_for_plot()}, LB**d", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    fontsize = 14
    plot.legend(fontsize=fontsize)
    # plot.yscale("log")
    plot.ylabel("Fraction of demand vectors covered", fontsize=fontsize)
    plot.xlabel("Number of popular objects", fontsize=fontsize)

    plot.title(
        fr"$d= {storage_design.d}$, "
        r"$\lambda= $" + fr"${demand_for_popular}$, "
        r"$N_{\textrm{sample}}= $" + fr"${num_samples}$, "
        r"$N_{\textrm{sim}}= $" + fr"${num_sim_run}$"
    )

    log(INFO, "Done")


def plot_frac_demand_vectors_covered_vs_num_popular_objs(
    d: int,
    demand_for_popular: int,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    # k = 120
    k = 30
    n = k
    use_cvxpy = True  # False

    # num_popular_obj_list = [2, 5, 10] + [int(k * frac) for frac in [0.1, 0.15, 0.2, 0.25]]
    # num_popular_obj_list = [2, 5, 10] + [int(k * frac) for frac in [0.1, 0.6, 0.8]]
    # num_popular_obj_list = [2, 5, 10]
    # num_popular_obj_list = list(range(1, 15))
    # num_popular_obj_list = list(range(1, 20))
    # num_popular_obj_list = list(range(1, 10)) + [int(k * frac) for frac in [0.1, 0.6, 0.8]]
    # num_popular_obj_list = [2, 5, 10, 15, 20]
    num_popular_obj_list = [int(num) for num in numpy.linspace(2, 30, 8)]

    log(INFO, "Started",
        num_popular_obj_list=num_popular_obj_list,
        demand_for_popular=demand_for_popular,
        num_samples=num_samples,
        num_sim_run=num_sim_run,
    )

    for storage_design, storage_design_model in [
        (
            design.RandomExpanderDesign(k=k, n=n, d=d_, use_cvxpy=use_cvxpy),
            storage_overlap_model.RandomExpanderDesignModel(k=k, n=n, d=d_)
        )
        # for d_ in range(demand_for_popular, d + 1)
        # for d_ in range(2, d + 1)
        for d_ in [3]
    ]:
        plot_frac_demand_vectors_covered_vs_num_popular_objs_for_storage_design(
            storage_design=storage_design,
            storage_design_model=storage_design_model,
            demand_for_popular=demand_for_popular,
            num_popular_obj_list=num_popular_obj_list,
            num_samples=num_samples,
            num_sim_run=num_sim_run,
        )

    # Save the plot
    plot.gcf().set_size_inches(8, 6)
    file_name = (
        "plots/exp_random_design_plot_frac_demand_vectors_covered_vs_num_popular_objs"
        + f"_k_{k}"
        + f"_d_{d}"
        + f"_lambda_{demand_for_popular}"
        # + f"_cvxpy_{use_cvxpy}"
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
                # num_samples=1000,
                num_sim_run=3,
            )

    # plot_(d=2)
    plot_(d=3)
    # plot_(d=4)
    # plot_(d=5)
    # plot_(d=6)

    log(INFO, "Done")


def manage_plot_frac_demand_vectors_covered_vs_num_popular_objs_w_joblib():
    log(INFO, "Started")

    joblib.Parallel(n_jobs=-1, prefer="processes")(
        joblib.delayed(plot_frac_demand_vectors_covered_vs_num_popular_objs)(
            d=d,
            demand_for_popular=demand_for_popular,
            # num_samples=1000,
            num_samples=300,
            num_sim_run=3,
            # num_samples=5000,
            # num_sim_run=5,
        )
        # for d in range(2, 7)
        # for demand_for_popular in range(1, d + 1)
        # for d in [4]
        # for demand_for_popular in [3]
        for d in [4]
        # for d in [6]
        # for demand_for_popular in [3]
        for demand_for_popular in [1]
    )

    log(INFO, "Done")


if __name__ == "__main__":
    # manage_plot_frac_demand_vectors_covered_vs_num_popular_objs()
    manage_plot_frac_demand_vectors_covered_vs_num_popular_objs_w_joblib()
