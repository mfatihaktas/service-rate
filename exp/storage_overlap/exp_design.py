import joblib
import numpy

from src.storage_overlap import (
    design,
    design_w_stripe,
    sim,
)
from src.model import demand

from src.utils.plot import *


def plot_frac_demand_vectors_covered_vs_prob_obj_is_active(
    d: int,
    demand_for_active_obj: int,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    log(INFO, "Started",
        d=d,
        demand_for_active_obj=demand_for_active_obj,
        num_samples=num_samples,
        num_sim_run=num_sim_run,
    )

    def plot_(storage_design: design.ReplicaDesign):
        log(INFO, f">> storage_design= {storage_design}")

        prob_obj_is_active_list = []

        E_frac_of_demand_vectors_covered_list = []
        std_frac_of_demand_vectors_covered_list = []

        for prob_obj_is_active in numpy.linspace(0.1, 0.7, 8):
            log(INFO, f"> prob_obj_is_active= {prob_obj_is_active}")

            prob_obj_is_active_list.append(prob_obj_is_active)

            # sample_demand_vector = lambda: demand.sample_demand_vector_w_zipf_law(
            #     num_objs=storage_design.k,
            #     num_active_obj=num_active_obj,
            #     cum_demand=demand_for_active_obj * num_active_obj,
            #     zipf_tail_index=0,
            # )
            sample_demand_vector = lambda: demand.sample_demand_vector_w_bernoulli_demands(
                num_objs=storage_design.k,
                demand_for_active_obj=demand_for_active_obj,
                prob_obj_is_active=prob_obj_is_active,
            )

            frac_of_demand_vectors_covered_list = sim.sim_frac_of_demand_vectors_covered(
                sample_demand_vector=sample_demand_vector,
                storage_design=storage_design,
                num_samples=num_samples,
                num_sim_run=num_sim_run,
            )
            E_frac_of_demand_vectors_covered_list.append(numpy.mean(frac_of_demand_vectors_covered_list))
            std_frac_of_demand_vectors_covered_list.append(numpy.std(frac_of_demand_vectors_covered_list))

        log(INFO, f"storage_design= {storage_design}",
            prob_obj_is_active_list=prob_obj_is_active_list,
            E_frac_of_demand_vectors_covered_list=E_frac_of_demand_vectors_covered_list,
            std_frac_of_demand_vectors_covered_list=std_frac_of_demand_vectors_covered_list,
        )

        color = next(dark_color_cycle)
        plot.errorbar(prob_obj_is_active_list, E_frac_of_demand_vectors_covered_list, yerr=std_frac_of_demand_vectors_covered_list, label=f"{storage_design.repr_for_plot()}", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    # k = 45
    # k = 111
    k = 120
    # k = 200
    # k = 24
    n = k
    use_cvxpy = True  # False
    replica_design_list = [
        design.ClusteringDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
        # design_w_stripe.ClusteringDesignWithStripe(k=k, n=n, d=d, s=2, use_cvxpy=use_cvxpy),
        design.CyclicDesign(k=k, n=n, d=d, shift_size=1, use_cvxpy=use_cvxpy),
        # design.CyclicDesign(k=k, n=n, d=d, shift_size=2, use_cvxpy=use_cvxpy),
        # design.CyclicDesign(k=k, n=n, d=d, shift_size=3, use_cvxpy=use_cvxpy),
        design.RandomBlockDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
        design.RandomExpanderDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
        # design.RandomExpanderDesign_wClusters(k=k, n=n, d=d, num_clusters=2, use_cvxpy=use_cvxpy),
        # design.TwoXORDesign(k=124, n=124, d=d, use_cvxpy=use_cvxpy),
    ]

    # for s in [2, 3, 4]:
    # for s in [2, 3]:
    #     if s <= d:
    #         replica_design_list.append(
    #             # design_w_stripe.RandomBlockDesignWithStripe(k=k, n=n, d=d, s=s, use_cvxpy=True),
    #             design_w_stripe.RandomExpanderDesignWithStripe(k=k, n=n, d=d, s=s, use_cvxpy=True)
    #         )

    for storage_design in replica_design_list:
        plot_(storage_design=storage_design)

    fontsize = 14
    plot.legend(fontsize=fontsize)
    # plot.yscale("log")
    plot.ylabel(r"$\mathcal{P}_{m, \lambda}$", fontsize=fontsize)
    plot.xlabel(r"$p$", fontsize=fontsize)

    plot.title(
        fr"$d= {d}$, "
        fr"$\lambda= {demand_for_active_obj}$, "
        r"$N_{\textrm{sample}}= $" + fr"${num_samples}$, "
        r"$N_{\textrm{sim}}= $" + fr"${num_sim_run}$"
    )

    # Save the plot
    plot.gcf().set_size_inches(8, 6)
    file_name = (
        "plots/plot_frac_demand_vectors_covered_vs_prob_obj_is_active"
        + f"_k_{k}"
        + f"_d_{d}"
        + f"_lambda_{demand_for_active_obj}"
        + ".png"
    )
    plot.savefig(file_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")


def manage_plot_frac_demand_vectors_covered_vs_prob_obj_is_active_w_joblib():
    log(INFO, "Started")

    joblib.Parallel(n_jobs=-1, prefer="processes")(
        joblib.delayed(plot_frac_demand_vectors_covered_vs_prob_obj_is_active)(
            d=d,
            demand_for_active_obj=demand_for_active_obj,
            num_samples=300,
            # num_samples=1000,
            num_sim_run=3,
        )
        for d in range(2, 7)
        for demand_for_active_obj in numpy.linspace(1.5, d, 4)
        # for d in [3]
        # for demand_for_active_obj in [2]
        # for d in [3, 4]
        # for demand_for_active_obj in [2, 3]
        # for demand_for_active_obj in [d - 1]
    )

    log(INFO, "Done")


if __name__ == "__main__":
    manage_plot_frac_demand_vectors_covered_vs_prob_obj_is_active_w_joblib()
