import joblib
import numpy

from src.storage_overlap import (
    design,
    design_w_stripe,
    sim,
)

from src.utils.plot import *


def plot_frac_demand_vectors_covered_vs_num_popular_objs(
    d_max: int,
    lambda_: int,
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
        lambda_=lambda_,
        num_samples=num_samples,
        num_sim_run=num_sim_run,
    )

    def plot_(replica_design: design.ReplicaDesign):
        log(INFO, f">> replica_design= {replica_design}")

        m_list = []
        E_frac_of_demand_vectors_covered_list = []
        std_frac_of_demand_vectors_covered_list = []
        frac_of_demand_vectors_covered_power_d_list = []
        frac_of_demand_vectors_covered_power_d_d_list = []

        # for m in range(1, k):
        for m in range(1, k, 3):
            log(INFO, f"> m= {m}")

            m_list.append(m)

            frac_of_demand_vectors_covered_list = sim.sim_frac_of_demand_vectors_covered(
                storage_design=replica_design,
                num_popular_obj=m,
                cum_demand=lambda_ * m,
                zipf_tail_index=0,
                num_samples=num_samples,
                num_sim_run=num_sim_run,
            )
            E_frac_of_demand_vectors_covered = numpy.mean(frac_of_demand_vectors_covered_list)
            E_frac_of_demand_vectors_covered_list.append(E_frac_of_demand_vectors_covered)
            std_frac_of_demand_vectors_covered_list.append(numpy.std(frac_of_demand_vectors_covered_list))

            frac_of_demand_vectors_covered_power_d_list.append(E_frac_of_demand_vectors_covered**d)
            frac_of_demand_vectors_covered_power_d_d_list.append((E_frac_of_demand_vectors_covered**d)**d)

            if E_frac_of_demand_vectors_covered < 0.01:
                break

        log(INFO, f"replica_design= {replica_design}",
            m_list=m_list,
            E_frac_of_demand_vectors_covered_list=E_frac_of_demand_vectors_covered_list,
            std_frac_of_demand_vectors_covered_list=std_frac_of_demand_vectors_covered_list,
        )

        color = next(dark_color_cycle)
        plot.errorbar(m_list, E_frac_of_demand_vectors_covered_list, yerr=std_frac_of_demand_vectors_covered_list, label=f"{replica_design.repr_for_plot()}, d={replica_design.d}", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
        plot.plot(m_list, frac_of_demand_vectors_covered_power_d_list, label=f"{replica_design.repr_for_plot()}, d={replica_design.d}, 'd", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
        plot.plot(m_list, frac_of_demand_vectors_covered_power_d_d_list, label=f"{replica_design.repr_for_plot()}, d={replica_design.d}, 'd'd", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    use_cvxpy = False
    replica_design_list = []

    for d in range(1, d_max + 1):
        replica_design_list_ = [
            design.ClusteringDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
            # design_w_stripe.ClusteringDesignWithStripe(k=k, n=n, d=d, s=2, use_cvxpy=use_cvxpy),
            # design.CyclicDesign(k=k, n=n, d=d, shift_size=1, use_cvxpy=use_cvxpy),
            # design.CyclicDesign(k=k, n=n, d=d, shift_size=2, use_cvxpy=use_cvxpy),
            # design.CyclicDesign(k=k, n=n, d=d, shift_size=3, use_cvxpy=use_cvxpy),
            # design.RandomBlockDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
            # design.RandomExpanderDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
            # design.RandomExpanderDesign_wClusters(k=k, n=n, d=d, num_clusters=2, use_cvxpy=use_cvxpy),
            # design.TwoXORDesign(k=124, n=124, d=d, use_cvxpy=use_cvxpy),
        ]

        replica_design_list.extend(replica_design_list_)

    # for s in [2, 3, 4]:
    # for s in [2, 3]:
    #     if s <= d:
    #         replica_design_list.append(
    #             # design_w_stripe.RandomBlockDesignWithStripe(k=k, n=n, d=d, s=s, use_cvxpy=True),
    #             design_w_stripe.RandomExpanderDesignWithStripe(k=k, n=n, d=d, s=s, use_cvxpy=True)
    #         )

    for replica_design in replica_design_list:
        plot_(replica_design=replica_design)

    fontsize = 14
    plot.legend(fontsize=fontsize)
    # plot.yscale("log")
    plot.ylabel(r"$\mathcal{P}_{m, \lambda}$", fontsize=fontsize)
    plot.xlabel(r"$m$", fontsize=fontsize)

    plot.title(
        fr"$\lambda= {lambda_}$, "
        r"$N_{\textrm{sample}}= $" + fr"${num_samples}$, "
        r"$N_{\textrm{sim}}= $" + fr"${num_sim_run}$"
    )

    # Save the plot
    plot.gcf().set_size_inches(8, 6)
    file_name = (
        "plots/plot_frac_demand_vectors_covered_vs_num_popular_objs"
        + f"_k_{k}"
        + f"_d_max_{d_max}"
        + f"_lambda_{lambda_}"
        + ".png"
    )
    plot.savefig(file_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")


def manage_plot_frac_demand_vectors_covered_vs_num_popular_objs_w_joblib():
    log(INFO, "Started")

    joblib.Parallel(n_jobs=-1, prefer="processes")(
        joblib.delayed(plot_frac_demand_vectors_covered_vs_num_popular_objs)(
            d_max=d_max,
            lambda_=lambda_,
            num_samples=300,
            # num_samples=1000,
            num_sim_run=3,
        )
        for d_max in [3]
        for lambda_ in [1.5, 2]
    )

    log(INFO, "Done")


if __name__ == "__main__":
    manage_plot_frac_demand_vectors_covered_vs_num_popular_objs_w_joblib()
