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
    maximal_load: float = 1,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    # k = 45
    # k = 111
    # k = 400
    # k = 120
    k = 20
    # k = 10
    n = k

    log(INFO, "Started",
        d_max=d_max,
        demand_for_active_obj=demand_for_active_obj,
        num_samples=num_samples,
        num_sim_run=num_sim_run,
    )

    def plot_(
        storage_design: design.RandomExpanderDesign,
        storage_model: model.RandomExpanderDesignModel,
        run_sim=False,
    ):
        log(INFO, f">> storage_design= {storage_design}")

        prob_obj_is_active_list = []
        mean_obj_demand_list = []

        E_frac_of_demand_vectors_covered_list = []
        std_frac_of_demand_vectors_covered_list = []
        prob_serving_upper_bound_list = []
        prob_serving_lower_bound_list = []

        for prob_obj_is_active in numpy.linspace(0.1, 0.8, 10):
            log(INFO, f"> prob_obj_is_active= {prob_obj_is_active}")

            prob_obj_is_active_list.append(prob_obj_is_active)
            mean_obj_demand_list.append(prob_obj_is_active * demand_for_active_obj)

            demand_vector_sampler = demand.DemandVectorSamplerWithBernoulliObjDemands(
                num_objs=storage_design.k,
                demand_for_active_obj=demand_for_active_obj,
                prob_obj_is_active=prob_obj_is_active,
            )

            frac_of_demand_vectors_covered_list = [0.02, 0.02]
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

            # Upper bound
            prob_serving_upper_bound = storage_model.prob_serving_upper_bound(
                p=prob_obj_is_active, lambda_=demand_for_active_obj, maximal_load=maximal_load
            )
            prob_serving_upper_bound_list.append(prob_serving_upper_bound)

            # Lower bound
            prob_serving_lower_bound = storage_model.prob_serving_lower_bound(
                p=prob_obj_is_active, lambda_=demand_for_active_obj, maximal_load=maximal_load
            )
            prob_serving_lower_bound_list.append(prob_serving_lower_bound)

            # if E_frac_of_demand_vectors_covered < 0.01:
            if prob_serving_upper_bound < 0.01:
                log(WARNING, "Early break", prob_obj_is_active=prob_obj_is_active)
                break

        log(INFO, f"storage_design= {storage_design}",
            prob_obj_is_active_list=prob_obj_is_active_list,
            E_frac_of_demand_vectors_covered_list=E_frac_of_demand_vectors_covered_list,
            std_frac_of_demand_vectors_covered_list=std_frac_of_demand_vectors_covered_list,
        )

        # x_list = prob_obj_is_active_list
        x_list = mean_obj_demand_list

        color = next(dark_color_cycle)
        if run_sim:
            plot.errorbar(x_list, E_frac_of_demand_vectors_covered_list, yerr=std_frac_of_demand_vectors_covered_list, label=fr"$d={storage_design.d}$, sim", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
        plot.plot(x_list, prob_serving_upper_bound_list, label=fr"$d={storage_design.d}$, ub", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
        plot.plot(x_list, prob_serving_lower_bound_list, label=fr"$d={storage_design.d}$, lb", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    use_cvxpy = True  # False
    storage_design_model_list = [
        (
            design.RandomExpanderDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
            model.RandomExpanderDesignModel(k=k, n=n, d=d)
        )
        # for d in [2, 3]
        # for d in range(2, d_max + 1)
        # for d in set(range(2, d_max + 1)) - set([4])
        # for d in [2, 3]
        # for d in [4, 5]
        for d in [2, 3, 5]
        # for d in range(4, d_max + 1)
    ]

    run_sim = True
    for storage_design, storage_model in storage_design_model_list:
        plot_(storage_design=storage_design, storage_model=storage_model, run_sim=run_sim)

    fontsize = 14
    plot.legend(fontsize=fontsize, loc="center right", bbox_to_anchor=(1.25, 0.65))
    # plot.yscale("log")
    plot.ylabel(r"$\mathcal{P}$ for random design", fontsize=fontsize)
    # plot.xlabel(r"$p$", fontsize=fontsize)
    plot.xlabel(r"$E[\rho]$", fontsize=fontsize)

    plot.title(
        rf"$k= {k}$, "
        rf"$m= {maximal_load}$, "
        # rf"$\lambda= {demand_for_active_obj}$, "
        # rf"$\rho \sim {demand_for_active_obj} \times$ Bernoulli"
        rf"$\rho \sim$ Bernoulli"
        # r"$N_{\textrm{sample}}= $" + fr"${num_samples}$, "
        # r"$N_{\textrm{sim}}= $" + fr"${num_sim_run}$"
    )

    # Save the plot
    # plot.gcf().set_size_inches(8, 6)
    plot.gcf().set_size_inches(6, 4)

    file_name = (
        "plots/plot_P_vs_d_for_random"
        + f"_k_{k}"
        # + f"_d_max_{d_max}"
        + f"_lambda_{demand_for_active_obj}"
        + ".pdf"
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
            maximal_load=0.7,  # 0.7,
            # num_samples=100,
            num_samples=1000,
            num_sim_run=3,
        )

        for d_max in [6]
        # for d_max in [4]
        for demand_for_active_obj in [1]
        # for demand_for_active_obj in [3]
        # for demand_for_active_obj in [1.5, 2]
        # for demand_for_active_obj in [1.5, 2, 3, 4, 5]
        # for demand_for_active_obj in numpy.arange(1.01, 2, 0.1)
    )

    log(INFO, "Done")


if __name__ == "__main__":
    manage_plot_frac_demand_vectors_covered_vs_d_w_joblib()
