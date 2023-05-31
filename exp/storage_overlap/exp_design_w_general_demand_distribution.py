import enum
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


class DemandDistribution(enum.Enum):
    Bernoulli = "Bernoulli"
    Exp = "Exp"
    Pareto = "Pareto"


def plot_P_for_given_params(
    k: int,
    d: int,
    num_active_objs: int,
    maximal_load: float,
    demand_dist: DemandDistribution,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    log(INFO, "Started",
        k=k,
        d=d,
        num_active_objs=num_active_objs,
        maximal_load=maximal_load,
        demand_dist=demand_dist,
        num_samples=num_samples,
        num_sim_run=num_sim_run,
    )

    if demand_dist == DemandDistribution.Bernoulli:
        D = 2
        p_l = numpy.linspace(0.1, 0.8, 10)
        active_obj_demand_rv_list = [
            random_variable.Bernoulli(p=p, D=D) for p in p_l
        ]

        x_l = p_l
        xlabel = r"$p$"
        dist_in_title = fr"{D}" + r" \times \textrm{Bernoulli(p)}"

    elif demand_dist == DemandDistribution.Exp:
        mu_list = [1 / mean for mean in numpy.linspace(0.1, 2, 30)]
        active_obj_demand_rv_list = [
            random_variable.Exponential(mu=mu) for mu in mu_list

        ]
        x_l = mu_list
        xlabel = r"$\mu$"
        dist_in_title = r"\textrm{Exp}"

    elif demand_dist == DemandDistribution.Pareto:
        min_value = 0.1
        tail_index_list = numpy.linspace(0.1, 3, 10)
        active_obj_demand_rv_list = [
            random_variable.Pareto(loc=min_value, a=tail_index) for tail_index in tail_index_list
        ]

        x_l = tail_index_list
        xlabel = r"$\alpha$"
        dist_in_title = r"\textrm{Pareto}" + fr"(\lambda={min_value}, \alpha)"

    def plot_(
        storage_design: design.CyclicDesign,
        storage_model: model.CyclicDesignModelForGivenDemandDistribution,
        run_sim: bool = False,
    ):
        log(INFO, f">> storage_design= {storage_design}")

        E_frac_of_demand_vectors_covered_list = []
        std_frac_of_demand_vectors_covered_list = []

        P_ub_list = []

        for active_obj_demand_rv in active_obj_demand_rv_list:
            log(INFO, f"> active_obj_demand_rv= {active_obj_demand_rv}")

            demand_vector_sampler = demand.DemandVectorSamplerWithFixedNumActiveObjs(
                num_objs=storage_design.k,
                num_active_objs=num_active_objs,
                active_obj_demand_rv=active_obj_demand_rv,
            )

            # Sim
            E_frac_of_demand_vectors_covered = 0.02
            if run_sim:
                if E_frac_of_demand_vectors_covered_list and E_frac_of_demand_vectors_covered_list[-1] <= 0.01:
                    frac_of_demand_vectors_covered_list = [0.01]
                else:
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
                num_active_objs=num_active_objs,
                # max_combination_size=2,
                max_combination_size=num_active_objs,
                maximal_load=maximal_load,
            )
            P_ub_list.append(P_ub)

            if E_frac_of_demand_vectors_covered <= 0.01 and P_ub <= 0.01:
                break

        log(INFO, f"storage_design= {storage_design}",
            x_l=x_l,
            E_frac_of_demand_vectors_covered_list=E_frac_of_demand_vectors_covered_list,
            std_frac_of_demand_vectors_covered_list=std_frac_of_demand_vectors_covered_list,
            P_ub_list=P_ub_list,
        )

        color = next(dark_color_cycle)
        if run_sim:
            plot.errorbar(x_l, E_frac_of_demand_vectors_covered_list, yerr=std_frac_of_demand_vectors_covered_list, label=f"{storage_design.repr_for_plot()}", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
        plot.plot(x_l, P_ub_list, label=f"{storage_design.repr_for_plot()}, UB", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

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

        (
            design.RandomExpanderDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
            model.RandomDesignModelForGivenDemandDistribution(k=k, n=n, d=d)
        ),

        # (
        #     design.RandomBlockDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
        #     model.BlockDesignModelForGivenDemandDistribution(k=k, n=n, d=d)
        # ),
    ]

    run_sim = True
    for storage_design, storage_model in storage_design_and_model_list:
        plot_(storage_design=storage_design, storage_model=storage_model, run_sim=run_sim)

    plot.xlabel(xlabel, fontsize=fontsize)
    plot.title(
        fr"$k= n= {k}$, "
        fr"$d= {d}$, "
        fr"$m= {maximal_load}$, "
        # r"$n_{\textrm{active}}= $" + fr"${num_active_objs}$, "
        fr"$\rho \sim {dist_in_title}$"
        # r"$\rho \sim \textrm{Pareto}(\lambda=0.1, \alpha)$"
        # r"$N_{\textrm{sample}}= $" + fr"${num_samples}$, "
        # r"$N_{\textrm{sim}}= $" + fr"${num_sim_run}$"
    )


def plot_P(
    d_list: list[int],
    num_active_objs: int,
    maximal_load: float,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    k = 120
    if num_active_objs is None:
        num_active_objs = k

    demand_dist = DemandDistribution.Bernoulli
    # demand_dist = DemandDistribution.Exp
    # demand_dist = DemandDistribution.Pareto

    for d in d_list:
        plot_P_for_given_params(
            k=k,
            d=d,
            num_active_objs=num_active_objs,
            maximal_load=maximal_load,
            demand_dist=demand_dist,
            num_samples=num_samples,
            num_sim_run=num_sim_run,
        )

    # if demand_dist == DemandDistribution.Exp:
    #     xlabel = r"$E[\rho]$"
    #     dist_in_title = r"\textrm{Exp}"
    # elif demand_dist == DemandDistribution.Pareto:
    #     xlabel = r"$\alpha$"
    #     dist_in_title = r"\textrm{Pareto}(\lambda=0.1, \alpha)"

    fontsize = 16
    plot.legend(fontsize=14, loc="upper right", bbox_to_anchor=(1.35, 0.75))
    plot.xlabel(xlabel, fontsize=fontsize)
    plot.ylabel(r"$\mathcal{P}$", fontsize=fontsize)

    # d = d_list[0]
    # plot.title(
    #     fr"$k= n= {k}$, "
    #     fr"$d= {d}$, "
    #     fr"$m= {maximal_load}$, "
    #     # r"$n_{\textrm{active}}= $" + fr"${num_active_objs}$, "
    #     fr"$\rho \sim {dist_in_title}$"
    #     # r"$\rho \sim \textrm{Pareto}(\lambda=0.1, \alpha)$"
    #     # r"$N_{\textrm{sample}}= $" + fr"${num_samples}$, "
    #     # r"$N_{\textrm{sim}}= $" + fr"${num_sim_run}$"
    # )

    # Save the plot
    plot.gcf().set_size_inches(6, 4)
    file_name = (
        f"plots/plot_P_{demand_dist.value}_demand"
        + f"_k_{k}"
        + f"_d_{d}"
        + f"_m_{maximal_load}"
        # + f"_n_active_{num_active_objs}"
        + ".pdf"
    )
    plot.savefig(file_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")


def manage_plot_P_w_joblib():
    log(INFO, "Started")

    joblib.Parallel(n_jobs=-1, prefer="processes")(
        joblib.delayed(plot_P)(
            d_list=[3],
            # d_list=[6],
            # d_list=[10],
            # d_list=[2, 3, 4],
            # d_list=[5, 6],
            # d_list=[2, 3, 4, 5],
            num_active_objs=num_active_objs,
            maximal_load=0.7,  # 0.7,
            num_samples=10,  # 300,
            # num_samples=1000,
            num_sim_run=3,
        )
        for num_active_objs in [None]
        # for num_active_objs in [3]
        # for num_active_objs in [2, 3]
        # for num_active_objs in [4, 5]
    )

    log(INFO, "Done")


if __name__ == "__main__":
    manage_plot_P_w_joblib()
