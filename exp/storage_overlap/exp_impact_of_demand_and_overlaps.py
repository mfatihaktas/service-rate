import joblib
import numpy

from src.storage_overlap import (
    design,
    model,
    sim,
)
from src.model import demand
from src.sim import random_variable

from src.utils.plot import *


def plot_P_w_exp_demand_distribution(
    k: int,
    d: int,
    num_active_objs: int,
    maximal_load: float,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    log(INFO, "Started",
        k=k,
        d=d,
        num_active_objs=num_active_objs,
        maximal_load=maximal_load,
        num_samples=num_samples,
        num_sim_run=num_sim_run,
    )

    def plot_(
        storage_design: design.CyclicDesign,
        storage_model: model.CyclicDesignModelForGivenDemandDistribution,
        run_sim: bool = False,
    ):
        log(INFO, f">> storage_design= {storage_design}")

        mean_obj_demand_list = []

        E_frac_of_demand_vectors_covered_list = []
        std_frac_of_demand_vectors_covered_list = []

        P_ub_list = []

        for mean_obj_demand in numpy.linspace(0.1, 3, 10):
            log(INFO, f"> mean_obj_demand= {mean_obj_demand}")

            mean_obj_demand_list.append(mean_obj_demand)

            active_obj_demand_rv = random_variable.Exponential(mu=1 / mean_obj_demand)

            demand_vector_sampler = demand.DemandVectorSamplerWithFixedNumActiveObjs(
                num_objs=storage_design.k,
                num_active_objs=num_active_objs,
                active_obj_demand_rv=active_obj_demand_rv,
            )

            # Sim
            E_frac_of_demand_vectors_covered = 0.02
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

            # UB
            P_ub = storage_model.prob_serving_upper_bound(
                demand_rv=active_obj_demand_rv,
                # max_combination_size=2,
                max_combination_size=num_active_objs,
                maximal_load=maximal_load,
            )
            P_ub_list.append(P_ub)

            if E_frac_of_demand_vectors_covered < 0.01:
                break

        log(INFO, f"storage_design= {storage_design}",
            mean_obj_demand_list=mean_obj_demand_list,
            E_frac_of_demand_vectors_covered_list=E_frac_of_demand_vectors_covered_list,
            std_frac_of_demand_vectors_covered_list=std_frac_of_demand_vectors_covered_list,
        )

        color = next(dark_color_cycle)
        if run_sim:
            plot.errorbar(mean_obj_demand_list, E_frac_of_demand_vectors_covered_list, yerr=std_frac_of_demand_vectors_covered_list, label=f"{storage_design.repr_for_plot()}", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
        plot.plot(mean_obj_demand_list, P_ub_list, label=f"{storage_design.repr_for_plot()}, UB", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    n = k
    use_cvxpy = True

    storage_design_and_model_list = [
        (
            design.ClusteringDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
            model.ClusteringDesignModelForGivenDemandDistribution(k=k, n=n, d=d)
        ),

        (
            design.CyclicDesign(k=k, n=n, d=d, shift_size=1, use_cvxpy=use_cvxpy),
            model.CyclicDesignModelForGivenDemandDistribution(k=k, n=n, d=d)
        ),

        # (
        #     design.RandomExpanderDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
        #     model.RandomDesignModelForGivenDemandDistribution(k=k, n=n, d=d)
        # ),

        # (
        #     design.RandomBlockDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
        #     model.BlockDesignModelForGivenDemandDistribution(k=k, n=n, d=d)
        # ),
    ]

    run_sim = True
    for storage_design, storage_model in storage_design_and_model_list:
        plot_(storage_design=storage_design, storage_model=storage_model, run_sim=run_sim)


def plot_P_w_pareto_demand_distribution(
    k: int,
    d: int,
    num_active_objs: int,
    maximal_load: float,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    log(INFO, "Started",
        k=k,
        d=d,
        num_active_objs=num_active_objs,
        maximal_load=maximal_load,
        num_samples=num_samples,
        num_sim_run=num_sim_run,
    )

    def plot_(
        storage_design: design.CyclicDesign,
        storage_model: model.CyclicDesignModelForGivenDemandDistribution,
        run_sim: bool = False,
    ):
        log(INFO, f">> storage_design= {storage_design}")

        tail_index_list = []

        E_frac_of_demand_vectors_covered_list = []
        std_frac_of_demand_vectors_covered_list = []

        P_ub_list = []

        for tail_index in range(8, 1, -1):
            log(INFO, f"> tail_index= {tail_index}")

            tail_index_list.append(tail_index)

            active_obj_demand_rv = random_variable.Pareto(loc=1, a=tail_index)

            demand_vector_sampler = demand.DemandVectorSamplerWithFixedNumActiveObjs(
                num_objs=storage_design.k,
                num_active_objs=num_active_objs,
                active_obj_demand_rv=active_obj_demand_rv,
            )

            # Sim
            E_frac_of_demand_vectors_covered = 0.02
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

            # UB
            P_ub = storage_model.prob_serving_upper_bound(
                demand_rv=active_obj_demand_rv,
                # max_combination_size=2,
                max_combination_size=num_active_objs,
                maximal_load=maximal_load,
            )
            P_ub_list.append(P_ub)

            # LB
            # if P_lb_list and P_lb_list[-1] < 0.01:
            #     P_lb = 0

            if E_frac_of_demand_vectors_covered < 0.01:
                break

        log(INFO, f"storage_design= {storage_design}",
            tail_index_list=tail_index_list,
            E_frac_of_demand_vectors_covered_list=E_frac_of_demand_vectors_covered_list,
            std_frac_of_demand_vectors_covered_list=std_frac_of_demand_vectors_covered_list,
        )

        color = next(dark_color_cycle)
        if run_sim:
            plot.errorbar(tail_index_list, E_frac_of_demand_vectors_covered_list, yerr=std_frac_of_demand_vectors_covered_list, label=f"{storage_design.repr_for_plot()}", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
        plot.plot(tail_index_list, P_ub_list, label=f"{storage_design.repr_for_plot()}, UB", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    n = k
    use_cvxpy = True

    storage_design_and_model_list = [
        (
            design.ClusteringDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
            model.ClusteringDesignModelForGivenDemandDistribution(k=k, n=n, d=d)
        ),

        (
            design.CyclicDesign(k=k, n=n, d=d, shift_size=1, use_cvxpy=use_cvxpy),
            model.CyclicDesignModelForGivenDemandDistribution(k=k, n=n, d=d)
        ),

        # (
        #     design.RandomExpanderDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
        #     model.RandomDesignModelForGivenDemandDistribution(k=k, n=n, d=d)
        # ),

        (
            design.RandomBlockDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
            model.BlockDesignModelForGivenDemandDistribution(k=k, n=n, d=d)
        ),
    ]

    run_sim = False
    for storage_design, storage_model in storage_design_and_model_list:
        plot_(storage_design=storage_design, storage_model=storage_model, run_sim=run_sim)


def plot_P_w_bernoulli_demand_distribution(
    k: int,
    d: int,
    num_active_objs: int,
    maximal_load: float,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    log(INFO, "Started",
        k=k,
        d=d,
        num_active_objs=num_active_objs,
        maximal_load=maximal_load,
        num_samples=num_samples,
        num_sim_run=num_sim_run,
    )

    def plot_(
        storage_design: design.CyclicDesign,
        storage_model: model.CyclicDesignModelForGivenDemandDistribution,
        run_sim: bool = False,
    ):
        log(INFO, f">> storage_design= {storage_design}")

        active_obj_demand_list = []

        E_frac_of_demand_vectors_covered_list = []
        std_frac_of_demand_vectors_covered_list = []

        P_ub_list = []

        for active_obj_demand in numpy.linspace(0.1, 3, 10):
            log(INFO, f"> active_obj_demand= {active_obj_demand}")

            active_obj_demand_list.append(active_obj_demand)

            active_obj_demand_rv = random_variable.CustomDiscrete(
                value_list=[active_obj_demand],
                prob_weight_list=[1],
            )

            demand_vector_sampler = demand.DemandVectorSamplerWithFixedNumActiveObjs(
                num_objs=storage_design.k,
                num_active_objs=num_active_objs,
                active_obj_demand_rv=active_obj_demand_rv,
            )

            # Sim
            E_frac_of_demand_vectors_covered = 0.02
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

            # UB
            # P_ub = storage_model.prob_serving_upper_bound(
            #     demand_rv=active_obj_demand_rv,
            #     # max_combination_size=2,
            #     max_combination_size=num_active_objs,
            #     maximal_load=maximal_load,
            # )
            # P_ub_list.append(P_ub)

            if E_frac_of_demand_vectors_covered < 0.01:
                break

        log(INFO, f"storage_design= {storage_design}",
            active_obj_demand_list=active_obj_demand_list,
            E_frac_of_demand_vectors_covered_list=E_frac_of_demand_vectors_covered_list,
            std_frac_of_demand_vectors_covered_list=std_frac_of_demand_vectors_covered_list,
        )

        color = next(dark_color_cycle)
        if run_sim:
            plot.errorbar(active_obj_demand_list, E_frac_of_demand_vectors_covered_list, yerr=std_frac_of_demand_vectors_covered_list, label=f"{storage_design.repr_for_plot()}", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
        # plot.plot(active_obj_demand_list, P_ub_list, label=f"{storage_design.repr_for_plot()}, UB", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    n = k
    use_cvxpy = True

    storage_design_and_model_list = [
        (
            design.ClusteringDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
            model.ClusteringDesignModelForGivenDemandDistribution(k=k, n=n, d=d)
        ),

        (
            design.CyclicDesign(k=k, n=n, d=d, shift_size=1, use_cvxpy=use_cvxpy),
            model.CyclicDesignModelForGivenDemandDistribution(k=k, n=n, d=d)
        ),

        # (
        #     design.RandomExpanderDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
        #     model.RandomDesignModelForGivenDemandDistribution(k=k, n=n, d=d)
        # ),

        (
            design.RandomBlockDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
            model.BlockDesignModelForGivenDemandDistribution(k=k, n=n, d=d)
        ),
    ]

    run_sim = True
    for storage_design, storage_model in storage_design_and_model_list:
        plot_(storage_design=storage_design, storage_model=storage_model, run_sim=run_sim)


def plot_P(
    d_list: list[int],
    num_active_objs: int,
    maximal_load: float,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    k = 120

    for d in d_list:
        # plot_P_w_exp_demand_distribution(
        #     k=k,
        #     d=d,
        #     num_active_objs=num_active_objs,
        #     maximal_load=maximal_load,
        #     num_samples=num_samples,
        #     num_sim_run=num_sim_run,
        # )

        # plot_P_w_pareto_demand_distribution(
        #     k=k,
        #     d=d,
        #     num_active_objs=num_active_objs,
        #     maximal_load=maximal_load,
        #     num_samples=num_samples,
        #     num_sim_run=num_sim_run,
        # )

        plot_P_w_bernoulli_demand_distribution(
            k=k,
            d=d,
            num_active_objs=num_active_objs,
            maximal_load=maximal_load,
            num_samples=num_samples,
            num_sim_run=num_sim_run,
        )


    fontsize = 14
    plot.legend(fontsize=fontsize, loc="upper right", bbox_to_anchor=(1.25, 0.75))
    plot.ylabel(r"$\mathcal{P}$", fontsize=fontsize)
    plot.xlabel(r"$E[\rho]$", fontsize=fontsize)

    plot.title(
        fr"$k= n= {k}$, "
        fr"$m= {maximal_load}$, "
        r"$n_{\textrm{active}}= $" + fr"${num_active_objs}$, "
        r"$\rho \sim$ Exp"
        # r"$N_{\textrm{sample}}= $" + fr"${num_samples}$, "
        # r"$N_{\textrm{sim}}= $" + fr"${num_sim_run}$"
    )

    # Save the plot
    plot.gcf().set_size_inches(6, 4)
    file_name = (
        "plots/plot_design_w_general_demand_distribution"
        + f"_k_{k}"
        + f"_m_{maximal_load}"
        + f"_n_active_{num_active_objs}"
        + ".pdf"
    )
    plot.savefig(file_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")


def manage_plot_P_w_joblib():
    log(INFO, "Started")

    joblib.Parallel(n_jobs=-1, prefer="processes")(
        joblib.delayed(plot_P)(
            # d_list=[2, 3, 4, 5],
            d_list=[2, 3],
            # d_list=[5, 6],
            num_active_objs=num_active_objs,
            maximal_load=1,  # 0.7,
            num_samples=100,  # 300,
            # num_samples=1000,
            num_sim_run=3,
        )
        # for num_active_objs in [3]
        for num_active_objs in [2, 3]
        # for num_active_objs in [4, 5]
    )

    log(INFO, "Done")


if __name__ == "__main__":
    manage_plot_P_w_joblib()
