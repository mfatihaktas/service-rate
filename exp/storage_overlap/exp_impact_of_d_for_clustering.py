import joblib
import numpy

from src.model import (
    demand,
)
from src.storage_overlap import (
    design,
    model,
    sim,
)

from src.utils.plot import *


def plot_frac_demand_vectors_covered_vs_d(
    d_max: int,
    demand_for_active_obj: int,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    # k = 45
    # k = 111
    # k = 120
    k = 24
    n = k

    log(INFO, "Started",
        d_max=d_max,
        demand_for_active_obj=demand_for_active_obj,
        num_samples=num_samples,
        num_sim_run=num_sim_run,
    )

    def plot_(
        storage_design: design.ReplicaDesign,
        storage_model: model.ClusteringDesignModel,
    ):
        log(INFO, f">> storage_design= {storage_design}")

        prob_obj_is_active_list = []

        E_frac_of_demand_vectors_covered_list = []
        std_frac_of_demand_vectors_covered_list = []
        prob_serving_model_list = []

        # for prob_obj_is_active in numpy.linspace(0.1, 0.8, 8):
        for prob_obj_is_active in numpy.linspace(0.1, 1, 10):
            log(INFO, f"> prob_obj_is_active= {prob_obj_is_active}")

            prob_obj_is_active_list.append(prob_obj_is_active)

            # frac_of_demand_vectors_covered_list = [0.02, 0.02]
            # sample_demand_vector = lambda: demand.sample_demand_vector_w_p(
            #     num_objs=storage_design.k,
            #     demand_for_active_obj=demand_for_active_obj,
            #     prob_obj_is_active=prob_obj_is_active,
            # )

            frac_of_demand_vectors_covered_list = [0, 0]
            # frac_of_demand_vectors_covered_list = sim.sim_frac_of_demand_vectors_covered(
            #     sample_demand_vector=sample_demand_vector,
            #     storage_design=storage_design,
            #     num_samples=num_samples,
            #     num_sim_run=num_sim_run,
            # )

            E_frac_of_demand_vectors_covered = numpy.mean(frac_of_demand_vectors_covered_list)
            E_frac_of_demand_vectors_covered_list.append(E_frac_of_demand_vectors_covered)
            std_frac_of_demand_vectors_covered_list.append(numpy.std(frac_of_demand_vectors_covered_list))

            # prob_serving_model = storage_model.prob_serving(p=prob_obj_is_active, lambda_=demand_for_active_obj)
            prob_serving_model = storage_model.prob_serving_downscaling_p_per_obj(p=prob_obj_is_active, lambda_=demand_for_active_obj)
            prob_serving_model_list.append(prob_serving_model)

            if prob_serving_model < 0.01:
                break

        log(INFO, f"storage_design= {storage_design}",
            prob_obj_is_active_list=prob_obj_is_active_list,
            E_frac_of_demand_vectors_covered_list=E_frac_of_demand_vectors_covered_list,
            std_frac_of_demand_vectors_covered_list=std_frac_of_demand_vectors_covered_list,
        )

        color = next(dark_color_cycle)
        # plot.errorbar(prob_obj_is_active_list, E_frac_of_demand_vectors_covered_list, yerr=std_frac_of_demand_vectors_covered_list, label=f"d={storage_model.d}, sim", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
        plot.plot(prob_obj_is_active_list, prob_serving_model_list, label=f"d={storage_model.d}, b={storage_model.b}", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    use_cvxpy = True  # False
    storage_design_model_list = [
        (
            design.ClusteringDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
            model.ClusteringDesignModel(k=k, n=n, b=b, d=d)
        )
        # for d in [2]
        for d in range(2, d_max + 1)
        for b in range(1, 4)
        if n % d == 0
    ]

    for storage_design, storage_model in storage_design_model_list:
        plot_(storage_design=storage_design, storage_model=storage_model)

    fontsize = 14
    plot.legend(fontsize=fontsize)
    # plot.yscale("log")
    plot.ylabel(r"$\mathcal{P}_{p, \lambda}$", fontsize=fontsize)
    plot.xlabel(r"$p$", fontsize=fontsize)

    plot.title(
        r"$\lambda= $" + fr"${demand_for_active_obj}$, "
        r"$N_{\textrm{sample}}= $" + fr"${num_samples}$, "
        r"$N_{\textrm{sim}}= $" + fr"${num_sim_run}$"
    )

    # Save the plot
    plot.gcf().set_size_inches(8, 6)
    file_name = (
        "plots/plot_frac_demand_vectors_covered_vs_d_for_clustering"
        + f"_k_{k}"
        + f"_d_max_{d_max}"
        + "_lambda_{}_".format(f"{demand_for_active_obj}".replace(".", "_"))
        + ".png"
    )
    plot.savefig(file_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")


def manage_plot_frac_demand_vectors_covered_vs_d_w_joblib():
    log(INFO, "Started")

    joblib.Parallel(n_jobs=-1, prefer="processes")(
        joblib.delayed(plot_frac_demand_vectors_covered_vs_d)(
            d_max=d_max,
            demand_for_active_obj=round(demand_for_active_obj, 1),
            num_samples=300,
            # num_samples=1000,
            num_sim_run=3,
        )

        for d_max in [10]
        for demand_for_active_obj in numpy.arange(1.01, 2, 0.1)
        # for demand_for_active_obj in numpy.arange(0.2, 1.0, 0.1)
        # for demand_for_active_obj in numpy.arange(1, 2, 0.1)
        # for demand_for_active_obj in [0.6]
        # for demand_for_active_obj in [1.1]
        # for demand_for_active_obj in [1.5]
        # for demand_for_active_obj in [1.5, 2, 2.5, 3]
    )

    log(INFO, "Done")


if __name__ == "__main__":
    manage_plot_frac_demand_vectors_covered_vs_d_w_joblib()
