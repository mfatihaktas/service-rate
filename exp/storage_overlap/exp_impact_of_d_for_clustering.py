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


def plot_w_bernoulli_obj_demands(
    demand_for_active_obj: int,
    storage_design: design.ReplicaDesign,
    storage_model: model.ClusteringDesignModelForBernoulliObjDemands,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    log(INFO, "Started", storage_design=storage_design, storage_model=storage_model)

    prob_obj_is_active_list = []

    E_frac_of_demand_vectors_covered_list = []
    std_frac_of_demand_vectors_covered_list = []
    prob_serving_model_list = []
    prob_serving_upper_bound_list = []

    # for prob_obj_is_active in numpy.linspace(0.1, 0.8, 8):
    for prob_obj_is_active in numpy.linspace(0.1, 1, 50):
        log(INFO, f"> prob_obj_is_active= {prob_obj_is_active}")

        prob_obj_is_active_list.append(prob_obj_is_active)

        # frac_of_demand_vectors_covered_list = [0.02, 0.02]
        demand_vector_sampler = demand.DemandVectorSamplerWithBernoulliObjDemands(
            num_objs=storage_design.k,
            demand_for_active_obj=demand_for_active_obj,
            prob_obj_is_active=prob_obj_is_active,
        )

        frac_of_demand_vectors_covered_list = [0, 0]
        # frac_of_demand_vectors_covered_list = sim.sim_frac_of_demand_vectors_covered(
        #     demand_vector_sampler=demand_vector_sampler,
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

        prob_serving_upper_bound = storage_model.prob_serving_upper_bound(p=prob_obj_is_active, lambda_=demand_for_active_obj)
        prob_serving_upper_bound_list.append(prob_serving_upper_bound)

        if prob_serving_model < 0.01:
            break

    log(INFO, f"storage_design= {storage_design}",
        prob_obj_is_active_list=prob_obj_is_active_list,
        E_frac_of_demand_vectors_covered_list=E_frac_of_demand_vectors_covered_list,
        std_frac_of_demand_vectors_covered_list=std_frac_of_demand_vectors_covered_list,
    )

    color = next(dark_color_cycle)
    # label = f"d={storage_model.d}, b={storage_model.b}"
    label = f"d={storage_model.d}"
    # plot.errorbar(prob_obj_is_active_list, E_frac_of_demand_vectors_covered_list, yerr=std_frac_of_demand_vectors_covered_list, label=f"{label}, sim", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
    plot.plot(prob_obj_is_active_list, prob_serving_model_list, label=f"{label}", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
    plot.plot(prob_obj_is_active_list, prob_serving_upper_bound_list, label=f"{label}, UB", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)


def plot_w_exp_obj_demands(
    storage_design: design.ReplicaDesign,
    storage_model: model.ClusteringDesignModelForExpObjDemands,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    log(INFO, "Started", storage_design=storage_design, storage_model=storage_model)

    mean_obj_demand_list = []

    E_frac_of_demand_vectors_covered_list = []
    std_frac_of_demand_vectors_covered_list = []
    prob_serving_model_list = []
    prob_serving_upper_bound_list = []

    for mean_obj_demand in numpy.linspace(0.1, 1, 20):
        log(INFO, f"> mean_obj_demand= {mean_obj_demand}")

        mean_obj_demand_list.append(mean_obj_demand)

        # frac_of_demand_vectors_covered_list = [0.02, 0.02]
        demand_vector_sampler = demand.DemandVectorSamplerWithExpObjDemands(
            num_objs=storage_design.k,
            mean_obj_demand=mean_obj_demand,
        )

        frac_of_demand_vectors_covered_list = [0, 0]
        # frac_of_demand_vectors_covered_list = sim.sim_frac_of_demand_vectors_covered(
        #     demand_vector_sampler=demand_vector_sampler,
        #     storage_design=storage_design,
        #     num_samples=num_samples,
        #     num_sim_run=num_sim_run,
        # )

        E_frac_of_demand_vectors_covered = numpy.mean(frac_of_demand_vectors_covered_list)
        E_frac_of_demand_vectors_covered_list.append(E_frac_of_demand_vectors_covered)
        std_frac_of_demand_vectors_covered_list.append(numpy.std(frac_of_demand_vectors_covered_list))

        # prob_serving_model = storage_model.prob_serving(mean_obj_demand=mean_obj_demand)
        prob_serving_model = storage_model.prob_serving_w_downscaling_mean_obj_demand_w_b(mean_obj_demand_b_1=mean_obj_demand)
        prob_serving_model_list.append(prob_serving_model)

        # prob_serving_upper_bound = storage_model.prob_serving_upper_bound(mean_obj_demand=mean_obj_demand)
        # prob_serving_upper_bound_list.append(prob_serving_upper_bound)

        if prob_serving_model < 0.01:
            break

    log(INFO, f"storage_design= {storage_design}",
        mean_obj_demand_list=mean_obj_demand_list,
        E_frac_of_demand_vectors_covered_list=E_frac_of_demand_vectors_covered_list,
        std_frac_of_demand_vectors_covered_list=std_frac_of_demand_vectors_covered_list,
    )

    color = next(dark_color_cycle)
    label = f"d={storage_model.d}, b={storage_model.b}"
    # label = f"d={storage_model.d}"
    # plot.errorbar(mean_obj_demand_list, E_frac_of_demand_vectors_covered_list, yerr=std_frac_of_demand_vectors_covered_list, label=f"{label}, sim", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
    plot.plot(mean_obj_demand_list, prob_serving_model_list, label=f"{label}", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
    # plot.plot(mean_obj_demand_list, prob_serving_upper_bound_list, label=f"{label}, UB", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)


def plot_frac_demand_vectors_covered_vs_d(
    d_max: int,
    mean_obj_demand: int,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    # k = 45
    # k = 111
    k = 120
    # k = 24
    n = k

    log(INFO, "Started",
        d_max=d_max,
        mean_obj_demand=mean_obj_demand,
        num_samples=num_samples,
        num_sim_run=num_sim_run,
    )

    use_cvxpy = True  # False
    storage_design_model_list = [
        (
            design.ClusteringDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
            # model.ClusteringDesignModelForBernoulliObjDemands(k=k, n=n, b=b, d=d)
            model.ClusteringDesignModelForExpObjDemands(k=k, n=n, b=b, d=d)
        )
        # for d in [1]
        for d in range(2, d_max + 1)
        for b in range(1, 3)
        if n % d == 0
    ]

    for storage_design, storage_model in storage_design_model_list:
        if isinstance(storage_model, model.ClusteringDesignModelForBernoulliObjDemands):
            plot_w_bernoulli_obj_demands(
                demand_for_active_obj=mean_obj_demand,
                storage_design=storage_design,
                storage_model=storage_model,
                num_samples=num_samples,
                num_sim_run=num_sim_run,
            )

        elif isinstance(storage_model, model.ClusteringDesignModelForExpObjDemands):
            plot_w_exp_obj_demands(
                storage_design=storage_design,
                storage_model=storage_model,
                num_samples=num_samples,
                num_sim_run=num_sim_run,
            )

    fontsize = 14
    plot.legend(fontsize=fontsize)
    # plot.yscale("log")
    plot.ylabel(r"$\mathcal{P}_{p, \lambda}$", fontsize=fontsize)
    plot.xlabel(r"$p$", fontsize=fontsize)

    plot.title(
        r"$\lambda= $" + fr"${mean_obj_demand}$, "
        r"$N_{\textrm{sample}}= $" + fr"${num_samples}$, "
        r"$N_{\textrm{sim}}= $" + fr"${num_sim_run}$"
    )

    # Save the plot
    plot.gcf().set_size_inches(8, 6)
    file_name = (
        "plots/plot_frac_demand_vectors_covered_vs_d_for_clustering"
        + f"_k_{k}"
        + f"_d_max_{d_max}"
        + "_lambda_{}_".format(f"{mean_obj_demand}".replace(".", "_"))
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
            mean_obj_demand=round(mean_obj_demand, 1),
            num_samples=200,
            # num_samples=300,
            # num_samples=1000,
            num_sim_run=3,
        )

        for d_max in [10]
        # for d_max in [20]
        for mean_obj_demand in [1.8]
        # for mean_obj_demand in numpy.arange(1.01, 2, 0.1)
        # for mean_obj_demand in numpy.arange(0.2, 1.0, 0.1)
        # for mean_obj_demand in numpy.arange(1, 2, 0.1)
        # for mean_obj_demand in [0.6]
        # for mean_obj_demand in [1.1]
        # for mean_obj_demand in [1.5]
        # for mean_obj_demand in [1.5, 2, 2.5, 3]
    )

    log(INFO, "Done")


if __name__ == "__main__":
    manage_plot_frac_demand_vectors_covered_vs_d_w_joblib()
