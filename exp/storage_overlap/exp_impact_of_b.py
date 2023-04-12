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


def plot_frac_demand_vectors_covered_vs_b(
    b_max: int,
    demand_b_1: int,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    # k = 45
    # k = 111
    # k = 120
    k = 24
    n = k

    log(INFO, "Started",
        b_max=b_max,
        demand_b_1=demand_b_1,
        num_samples=num_samples,
        num_sim_run=num_sim_run,
    )

    def plot_(
        b: int,
        storage_design: design.ReplicaDesign,
        storage_model: model.NoRedundancyDesignModel,
    ):
        log(INFO, f">> b= {b}, storage_design= {storage_design}")

        prob_obj_is_active_list = []

        E_frac_of_demand_vectors_covered_list = []
        std_frac_of_demand_vectors_covered_list = []
        prob_serving_model_list = []

        for prob_obj_is_active in numpy.linspace(0.1, 0.8, 8):
            log(INFO, f"> prob_obj_is_active= {prob_obj_is_active}")

            prob_obj_is_active_list.append(prob_obj_is_active)

            # frac_of_demand_vectors_covered_list = [0.02, 0.02]
            # sample_demand_vector = lambda: demand.sample_demand_vector_w_p(
            #     num_objs=storage_design.k,
            #     demand_for_active_obj=demand_b_1,
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

            # prob_serving_model = storage_model.prob_serving_downscaling_demand_per_obj(p=prob_obj_is_active, lambda_b_1=demand_b_1)
            prob_serving_model = storage_model.prob_serving_downscaling_p_per_obj(p=prob_obj_is_active, lambda_b_1=demand_b_1)
            # prob_serving_model = storage_model.prob_serving_downscaling_p_per_obj_to_keep_prob_serving_fixed(p=prob_obj_is_active, lambda_b_1=demand_b_1)

            prob_serving_model_list.append(prob_serving_model)

            if prob_serving_model < 0.01:
                break

        log(INFO, f"storage_design= {storage_design}",
            prob_obj_is_active_list=prob_obj_is_active_list,
            E_frac_of_demand_vectors_covered_list=E_frac_of_demand_vectors_covered_list,
            std_frac_of_demand_vectors_covered_list=std_frac_of_demand_vectors_covered_list,
        )

        color = next(dark_color_cycle)
        # plot.errorbar(prob_obj_is_active_list, E_frac_of_demand_vectors_covered_list, yerr=std_frac_of_demand_vectors_covered_list, label=f"b={b}", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
        plot.plot(prob_obj_is_active_list, prob_serving_model_list, label=f"b={b}, Model", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    use_cvxpy = True  # False
    b_storage_design_model_list = [
        (
            b,
            design.NoRedundancyDesign(k=b * k, n=n, use_cvxpy=use_cvxpy),
            model.NoRedundancyDesignModel(k=k, n=n, b=b)
        )
        for b in range(2, b_max + 1)
    ]

    for b, storage_design, storage_model in b_storage_design_model_list:
        plot_(b=b, storage_design=storage_design, storage_model=storage_model)

    fontsize = 14
    plot.legend(fontsize=fontsize)
    # plot.yscale("log")
    plot.ylabel(r"$\mathcal{P}_{p, \lambda}$", fontsize=fontsize)
    plot.xlabel(r"$p$", fontsize=fontsize)

    plot.title(
        r"$\lambda_{b=1}= $" + fr"${demand_b_1}$, "
        r"$N_{\textrm{sample}}= $" + fr"${num_samples}$, "
        r"$N_{\textrm{sim}}= $" + fr"${num_sim_run}$"
    )

    # Save the plot
    plot.gcf().set_size_inches(8, 6)
    file_name = (
        "plots/plot_frac_demand_vectors_covered_vs_b"
        + f"_k_{k}"
        + f"_b_max_{b_max}"
        + f"_lambda_{demand_b_1}"
        + ".png"
    )
    plot.savefig(file_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")


def manage_plot_frac_demand_vectors_covered_vs_b_w_joblib():
    log(INFO, "Started")

    joblib.Parallel(n_jobs=-1, prefer="processes")(
        joblib.delayed(plot_frac_demand_vectors_covered_vs_b)(
            b_max=b_max,
            demand_b_1=demand_b_1,
            num_samples=300,
            # num_samples=1000,
            num_sim_run=3,
        )
        # for b_max in [4]
        for b_max in [10]
        # for demand_b_1 in numpy.arange(0.2, 1.0, 0.1)
        for demand_b_1 in [round(demand, 2) for demand in numpy.arange(0.2, 1.0, 0.1)]
        # for demand_b_1 in [round(demand, 2) for demand in numpy.arange(1, 2, 0.1)]
        # for demand_b_1 in [0.6]
        # for demand_b_1 in [1.1]
        # for demand_b_1 in [1.5]
        # for demand_b_1 in [1.5, 2, 2.5, 3]
    )

    log(INFO, "Done")


if __name__ == "__main__":
    manage_plot_frac_demand_vectors_covered_vs_b_w_joblib()
