import joblib
import numpy

from src.storage_overlap import (
    design,
    design_w_stripe,
    sim,
)
from src.model import demand

from src.utils.plot import *


DEMAND_FOR_ACTIVE_OBJ = 2


def plot_P_vs_E_ro_for_given_d(
    k: int,
    d: int,
    maximal_load: float,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    log(INFO, "Started",
        d=d,
        maximal_load=maximal_load,
        num_samples=num_samples,
        num_sim_run=num_sim_run,
    )

    def plot_(storage_design: design.ReplicaDesign):
        log(INFO, f">> storage_design= {storage_design}")

        mean_obj_demand_list = []

        E_frac_of_demand_vectors_covered_list = []
        std_frac_of_demand_vectors_covered_list = []

        """
        for prob_obj_is_active in numpy.linspace(0.1, 0.8, 10):
            log(INFO, f"> prob_obj_is_active= {prob_obj_is_active}")

            mean_obj_demand_list.append(prob_obj_is_active * DEMAND_FOR_ACTIVE_OBJ)

            demand_vector_sampler = demand.DemandVectorSamplerWithBernoulliObjDemands(
                num_objs=storage_design.k,
                demand_for_active_obj=DEMAND_FOR_ACTIVE_OBJ,
                prob_obj_is_active=prob_obj_is_active,
            )

            frac_of_demand_vectors_covered_list = sim.sim_frac_of_demand_vectors_covered(
                demand_vector_sampler=demand_vector_sampler,
                storage_design=storage_design,
                maximal_load=maximal_load,
                num_samples=num_samples,
                num_sim_run=num_sim_run,
            )

            E_frac_of_demand_vectors_covered = numpy.mean(frac_of_demand_vectors_covered_list)
            E_frac_of_demand_vectors_covered_list.append(E_frac_of_demand_vectors_covered)
            std_frac_of_demand_vectors_covered_list.append(numpy.std(frac_of_demand_vectors_covered_list))

            if E_frac_of_demand_vectors_covered < 0.01:
                break
        """

        for mean_obj_demand in numpy.linspace(0.1, 1, 25):
            log(INFO, f"> mean_obj_demand= {mean_obj_demand}")

            mean_obj_demand_list.append(mean_obj_demand)

            demand_vector_sampler = demand.DemandVectorSamplerWithExpObjDemands(
                num_objs=storage_design.k,
                mean_obj_demand=mean_obj_demand,
            )

            frac_of_demand_vectors_covered_list = sim.sim_frac_of_demand_vectors_covered(
                demand_vector_sampler=demand_vector_sampler,
                storage_design=storage_design,
                maximal_load=maximal_load,
                num_samples=num_samples,
                num_sim_run=num_sim_run,
            )

            E_frac_of_demand_vectors_covered = numpy.mean(frac_of_demand_vectors_covered_list)
            E_frac_of_demand_vectors_covered_list.append(E_frac_of_demand_vectors_covered)
            std_frac_of_demand_vectors_covered_list.append(numpy.std(frac_of_demand_vectors_covered_list))

            if E_frac_of_demand_vectors_covered < 0.01:
                break

        log(INFO, f"storage_design= {storage_design}",
            mean_obj_demand_list=mean_obj_demand_list,
            E_frac_of_demand_vectors_covered_list=E_frac_of_demand_vectors_covered_list,
            std_frac_of_demand_vectors_covered_list=std_frac_of_demand_vectors_covered_list,
        )

        color = next(dark_color_cycle)
        plot.errorbar(mean_obj_demand_list, E_frac_of_demand_vectors_covered_list, yerr=std_frac_of_demand_vectors_covered_list, label=f"{storage_design.repr_for_plot()}", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

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


def plot_P_vs_E_ro(
    d: int,
    maximal_load: float,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    k = 120

    # for d_ in [d, d + 1]:
    # for d_ in [d]:
    for d_ in [d + 1]:
        plot_P_vs_E_ro_for_given_d(
            k=k,
            d=d_,
            maximal_load=maximal_load,
            num_samples=num_samples,
            num_sim_run=num_sim_run,
        )

    fontsize = 14
    plot.legend(fontsize=fontsize, loc="upper right", bbox_to_anchor=(1.25, 0.75))
    # plot.yscale("log")
    plot.ylabel(r"$\mathcal{P}$", fontsize=fontsize)
    plot.xlabel(r"$E[\rho]$", fontsize=fontsize)

    plot.title(
        # fr"$d= {d}$, "
        fr"$k= n= {k}$, "
        fr"$m= {maximal_load}$, "
        r"$\rho \sim$ Exp"
        # fr"$\rho \sim {DEMAND_FOR_ACTIVE_OBJ} \times$ Bernoulli"
        # r"$N_{\textrm{sample}}= $" + fr"${num_samples}$, "
        # r"$N_{\textrm{sim}}= $" + fr"${num_sim_run}$"
    )

    # Save the plot
    # plot.gcf().set_size_inches(8, 6)
    plot.gcf().set_size_inches(6, 4)
    file_name = (
        "plots/plot_P_vs_design"
        + f"_k_{k}"
        + f"_d_{d}"
        + f"_maximal_load_{maximal_load}"
        + ".pdf"
    )
    plot.savefig(file_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")


def manage_plot_frac_demand_vectors_covered_vs_prob_obj_is_active_w_joblib():
    log(INFO, "Started")

    joblib.Parallel(n_jobs=-1, prefer="processes")(
        joblib.delayed(plot_P_vs_E_ro)(
            d=d,
            maximal_load=0.7,
            num_samples=100,  # 300,
            # num_samples=1000,
            num_sim_run=3,
        )
        for d in [3]
        # for d in range(2, 7)
        # for d in [3, 4]
    )

    log(INFO, "Done")


if __name__ == "__main__":
    manage_plot_frac_demand_vectors_covered_vs_prob_obj_is_active_w_joblib()