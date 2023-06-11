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
    num_samples: int,
    num_sim_run: int,
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
    maximal_load: float,
    num_samples: int,
    num_sim_run: int,
):
    log(INFO, "Started",
        storage_design=storage_design,
        storage_model=storage_model,
        maximal_load=maximal_load
    )

    mean_obj_demand_list = []

    E_frac_of_demand_vectors_covered_list = []
    std_frac_of_demand_vectors_covered_list = []
    prob_serving_model_list = []
    prob_serving_model_power_d_list = []
    prob_serving_lower_bound_list = []

    # for mean_obj_demand in numpy.linspace(0.1, 1, 20):
    for mean_obj_demand in numpy.linspace(0.1, 1, 50):
        log(INFO, f"> mean_obj_demand= {mean_obj_demand}")

        mean_obj_demand_list.append(mean_obj_demand)

        # frac_of_demand_vectors_covered_list = [0.02, 0.02]
        # demand_vector_sampler = demand.DemandVectorSamplerWithExpObjDemands(
        #     num_objs=storage_design.k,
        #     mean_obj_demand=mean_obj_demand,
        # )

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
        prob_serving_model = storage_model.prob_serving_w_downscaling_mean_obj_demand_w_b(
            mean_obj_demand_b_1=mean_obj_demand, maximal_load=maximal_load
        )
        prob_serving_model_list.append(prob_serving_model)

        prob_serving_model_power_d_list.append(prob_serving_model ** (storage_design.d ** 2))

        prob_serving_lower_bound = storage_model.prob_serving_lower_bound_w_chernoff(
            mean_obj_demand=mean_obj_demand, maximal_load=maximal_load
        )
        prob_serving_lower_bound_list.append(prob_serving_lower_bound)

        if prob_serving_model < 0.0001:
            break

    log(INFO, f"storage_design= {storage_design}",
        mean_obj_demand_list=mean_obj_demand_list,
        E_frac_of_demand_vectors_covered_list=E_frac_of_demand_vectors_covered_list,
        std_frac_of_demand_vectors_covered_list=std_frac_of_demand_vectors_covered_list,
    )

    color = next(dark_color_cycle)
    # label = f"b={storage_model.b}" if storage_model.d == 1 else f"d={storage_model.d}"  # f"d={storage_model.d}, b={storage_model.b}"
    label = rf"$d={storage_model.d}$"
    # plot.errorbar(mean_obj_demand_list, E_frac_of_demand_vectors_covered_list, yerr=std_frac_of_demand_vectors_covered_list, label=f"{label}, sim", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
    plot.plot(mean_obj_demand_list, prob_serving_model_list, label=f"{label}", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
    # plot.plot(mean_obj_demand_list, prob_serving_model_power_d_list, label=f"{label}, **d", color=color, marker=None, linestyle="--", lw=2, mew=3, ms=5)
    plot.plot(mean_obj_demand_list, prob_serving_lower_bound_list, label=f"{label}, LB", color=color, marker=None, linestyle="--", lw=2, mew=3, ms=5)


def plot_w_pareto_obj_demands(
    storage_design: design.ReplicaDesign,
    storage_model: model.ClusteringDesignModelForParetoObjDemands,
    num_samples: int,
    num_sim_run: int,
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
        demand_vector_sampler = demand.DemandVectorSamplerWithParetoObjDemands(
            num_objs=storage_design.k,
            min_value=mean_obj_demand / 2,
            max_value=2 * mean_obj_demand,
            a=1.5,
            # min_value=mean_obj_demand / 10,
            # a=0.8,
            b=storage_design.k // storage_design.n,
        )

        frac_of_demand_vectors_covered_list = [0.02, 0.02]
        # frac_of_demand_vectors_covered_list = sim.sim_frac_of_demand_vectors_covered(
        #     demand_vector_sampler=demand_vector_sampler,
        #     storage_design=storage_design,
        #     num_samples=num_samples,
        #     num_sim_run=num_sim_run,
        # )

        E_frac_of_demand_vectors_covered = numpy.mean(frac_of_demand_vectors_covered_list)
        E_frac_of_demand_vectors_covered_list.append(E_frac_of_demand_vectors_covered)
        std_frac_of_demand_vectors_covered_list.append(numpy.std(frac_of_demand_vectors_covered_list))

        prob_serving_model = storage_model.prob_serving(min_value=mean_obj_demand / 2, tail_index=2)
        # prob_serving_model = storage_model.prob_serving_w_downscaling_mean_obj_demand_w_b(mean_obj_demand_b_1=mean_obj_demand)
        prob_serving_model_list.append(prob_serving_model)

        # prob_serving_upper_bound = storage_model.prob_serving_upper_bound(mean_obj_demand=mean_obj_demand)
        # prob_serving_upper_bound_list.append(prob_serving_upper_bound)

        if E_frac_of_demand_vectors_covered < 0.01:
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
    maximal_load: float = 1,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    # k = 45
    # k = 111
    # k = 120
    k = 1200
    # k = 24
    n = k

    log(INFO, "Started",
        d_max=d_max,
        mean_obj_demand=mean_obj_demand,
        maximal_load=maximal_load,
        num_samples=num_samples,
        num_sim_run=num_sim_run,
    )

    use_cvxpy = True  # False
    storage_design_model_list = [
        (
            design.ClusteringDesign(k=k * b, n=n, d=d, use_cvxpy=use_cvxpy),
            # model.ClusteringDesignModelForBernoulliObjDemands(k=k, n=n, b=b, d=d)
            model.ClusteringDesignModelForExpObjDemands(k=k, n=n, b=b, d=d)
            # model.ClusteringDesignModelForParetoObjDemands(k=k, n=n, b=b, d=d)
        )

        # for d in [1]
        for d in range(1, d_max + 1)
        for b in [1]
        # for b in [2]
        # for b in range(1, 10)
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
                maximal_load=maximal_load,
                num_samples=num_samples,
                num_sim_run=num_sim_run,
            )

        elif isinstance(storage_model, model.ClusteringDesignModelForParetoObjDemands):
            plot_w_pareto_obj_demands(
                storage_design=storage_design,
                storage_model=storage_model,
                num_samples=num_samples,
                num_sim_run=num_sim_run,
            )

    fontsize = 16
    plot.legend(fontsize=14)
    # plot.yscale("log")
    # plot.ylabel(r"$\mathcal{P}$ with no redundancy", fontsize=fontsize)
    plot.ylabel(r"$\mathcal{P}$ for clustering design", fontsize=fontsize)
    # plot.xlabel(r"$p$", fontsize=fontsize)
    plot.xlabel(r"$E[\rho]$", fontsize=fontsize)

    plot.title(
        (
            # rf"$k= {k}$, "
            rf"$k= n= {k}$, "
            # f"$d= 1$, "
            rf"$m= {maximal_load}$, "
            r"$\rho \sim \mathrm{Exp}$"
            # r"$\lambda= $" + fr"${mean_obj_demand}$, "
            # r"$N_{\textrm{sample}}= $" + fr"${num_samples}$, "
            # r"$N_{\textrm{sim}}= $" + fr"${num_sim_run}$",
        ),
        fontsize=fontsize
    )

    # Save the plot
    # plot.gcf().set_size_inches(8, 6)
    plot.gcf().set_size_inches(6, 4)
    file_name = (
        # "plots/plot_frac_demand_vectors_covered_vs_d_for_clustering"
        # "plots/plot_P_vs_b_for_clustering"
        "plots/plot_P_vs_d_for_clustering"
        + f"_k_{k}"
        # + f"_d_max_{d_max}"
        # + "_lambda_{}_".format(f"{mean_obj_demand}".replace(".", "_"))
        # + ".png"
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
            mean_obj_demand=round(mean_obj_demand, 1),
            maximal_load=0.7,
            num_samples=200,
            # num_samples=300,
            # num_samples=1000,
            num_sim_run=3,
        )

        for d_max in [10]
        # for d_max in [10]
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
