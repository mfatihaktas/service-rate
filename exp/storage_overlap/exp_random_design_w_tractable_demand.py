import joblib
import numpy

from src.storage_overlap import (
    design,
    model,
    sim,
)
from src.model import demand

from src.utils.plot import *


def plot_P_for_given_params(
    k: int,
    d: int,
    demand_for_active_obj: float,
    maximal_load: float,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    log(INFO, "Started",
        k=k,
        d=d,
        demand_for_active_obj=demand_for_active_obj,
        maximal_load=maximal_load,
        num_samples=num_samples,
        num_sim_run=num_sim_run,
    )

    def plot_(
        storage_design: design.RandomExpanderDesign,
        storage_model: model.RandomExpanderDesignModel,
    ):
        log(INFO, f">> storage_design= {storage_design}")

        num_active_objs_list = []

        E_frac_of_demand_vectors_covered_list = []
        std_frac_of_demand_vectors_covered_list = []

        P_ub_list = []
        P_lb_list = []

        for num_active_objs in range(1, storage_design.k):
            log(INFO, f"> num_active_objs= {num_active_objs}")

            num_active_objs_list.append(num_active_objs)

            demand_vector_sampler = demand.DemandVectorSamplerWithFixedActiveObjDemand(
                num_objs=storage_design.k,
                num_active_objs=num_active_objs,
                demand_for_active_obj=demand_for_active_obj,
            )

            # Sim
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
            P_ub = storage_model.prob_serving_upper_bound_for_given_m_w_joblib(
                m=num_active_objs, lambda_=demand_for_active_obj, maximal_load=maximal_load
            )
            P_ub_list.append(P_ub)

            # LB
            if P_lb_list and P_lb_list[-1] < 0.01:
                P_lb = 0
            else:
                P_lb = storage_model.prob_serving_lower_bound_for_given_m_w_joblib(
                    m=num_active_objs, lambda_=demand_for_active_obj, maximal_load=maximal_load
                )
            P_lb_list.append(P_lb)

            if E_frac_of_demand_vectors_covered < 0.01:
                break

        log(INFO, f"storage_design= {storage_design}",
            num_active_objs_list=num_active_objs_list,
            E_frac_of_demand_vectors_covered_list=E_frac_of_demand_vectors_covered_list,
            std_frac_of_demand_vectors_covered_list=std_frac_of_demand_vectors_covered_list,
        )

        color = next(dark_color_cycle)
        plot.errorbar(num_active_objs_list, E_frac_of_demand_vectors_covered_list, yerr=std_frac_of_demand_vectors_covered_list, label=f"{storage_design.repr_for_plot()}", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
        plot.plot(num_active_objs_list, P_ub_list, label=f"{storage_design.repr_for_plot()}, UB", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
        plot.plot(num_active_objs_list, P_lb_list, label=f"{storage_design.repr_for_plot()}, LB", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    n = k
    use_cvxpy = True  # False

    storage_design_and_model_list = [
        (
            design.RandomExpanderDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
            model.RandomExpanderDesignModel(k=k, n=n, d=d),
        )
    ]

    for storage_design, storage_model in storage_design_and_model_list:
        plot_(storage_design=storage_design, storage_model=storage_model)


def plot_P(
    d: int,
    demand_for_active_obj: float,
    maximal_load: float,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    k = 60
    # k = 120
    # k = 6

    plot_P_for_given_params(
        k=k,
        d=d,
        demand_for_active_obj=demand_for_active_obj,
        maximal_load=maximal_load,
        num_samples=num_samples,
        num_sim_run=num_sim_run,
    )

    fontsize = 14
    plot.legend(fontsize=fontsize, loc="upper right", bbox_to_anchor=(1.25, 0.75))
    # plot.yscale("log")
    plot.ylabel(r"$\mathcal{P}$", fontsize=fontsize)
    plot.xlabel("Number of active objects", fontsize=fontsize)

    plot.title(
        # fr"$d= {d}$, "
        fr"$k= n= {k}$, "
        fr"$m= {maximal_load}$, "
        # r"$n_\mathrm{active}= $" + fr"${num_active_objs}$, "
        fr"$\lambda= {demand_for_active_obj}$"
        # r"$\rho \sim$ Exp"
        # fr"$\rho \sim {demand_for_active_obj} \times$ Bernoulli"
        # r"$N_{\textrm{sample}}= $" + fr"${num_samples}$, "
        # r"$N_{\textrm{sim}}= $" + fr"${num_sim_run}$"
    )

    # Save the plot
    # plot.gcf().set_size_inches(8, 6)
    plot.gcf().set_size_inches(6, 4)
    file_name = (
        "plots/plot_random_design_P_tractable_demand"
        + f"_k_{k}"
        + f"_d_{d}"
        + f"_maximal_load_{maximal_load}"
        + ".pdf"
    )
    plot.savefig(file_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")


def manage_plot_P_w_joblib():
    log(INFO, "Started")

    joblib.Parallel(n_jobs=-1, prefer="processes")(
        joblib.delayed(plot_P)(
            d=d,
            demand_for_active_obj=demand_for_active_obj,
            maximal_load=1,  # 0.7,
            num_samples=100,  # 300,
            # num_samples=1000,
            num_sim_run=3,
        )
        # for d in [2, 3]
        for d in [3, 4]
        # for d in range(2, 7)
        # for d in [3, 4]
        for demand_for_active_obj in [d - 1]
        # for demand_for_active_obj in [3 * d / 4]
    )

    log(INFO, "Done")


if __name__ == "__main__":
    manage_plot_P_w_joblib()
