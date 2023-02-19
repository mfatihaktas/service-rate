import numpy

from src.storage_overlap import design
from src.model import demand

from src.utils.plot import *


def plot_frac_demand_vectors_covered_vs_d_for_different_replication_designs():
    num_popular_objs = 5
    cum_demand = 5
    # zipf_tail_index_list = [0, 1, 2]
    # zipf_tail_index_list = [0, 0.5, 1, 1.5, 2]
    # zipf_tail_index_list = [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3]
    zipf_tail_index_list = numpy.linspace(start=0, stop=4, num=20, endpoint=True)

    num_samples = 100
    num_sim_run = 3

    log(INFO, "Started",
        num_popular_objs=num_popular_objs,
        cum_demand=cum_demand,
        zipf_tail_index_list=zipf_tail_index_list,
        num_samples=num_samples,
        num_sim_run=num_sim_run,
    )

    def plot_(replica_design: design.ReplicaDesign):
        log(INFO, f">> replica_design= {replica_design}")

        E_frac_of_demand_vectors_covered_list = []
        std_frac_of_demand_vectors_covered_list = []

        for zipf_tail_index in zipf_tail_index_list:
            # With `get_demand_vectors_w_zipf_law`
            # demand_vector_list = demand.get_demand_vectors_w_zipf_law(
            #     num_objs=replica_design.k,
            #     num_popular_objs=num_popular_objs,
            #     cum_demand=cum_demand,
            #     zipf_tail_index=zipf_tail_index,
            # )
            # frac_of_demand_vectors_covered = replica_design.frac_of_demand_vectors_covered(
            #     demand_vector_list=demand_vector_list
            # )

            # With `frac_of_demand_vectors_covered_w_generator_input`
            # frac_of_demand_vectors_covered = replica_design.frac_of_demand_vectors_covered_w_generator_input(
            #     demand_vector_generator=demand.gen_demand_vector_w_zipf_law(
            #         num_objs=replica_design.k,
            #         num_popular_objs=num_popular_objs,
            #         cum_demand=cum_demand,
            #         zipf_tail_index=zipf_tail_index,
            #     )
            # )

            # With `sim_frac_of_demand_vectors_covered`
            frac_of_demand_vectors_covered_list = replica_design.sim_frac_of_demand_vectors_covered(
                num_popular_objs=num_popular_objs,
                cum_demand=cum_demand,
                zipf_tail_index=zipf_tail_index,
                num_samples=num_samples,
                num_sim_run=num_sim_run,
            )

            log(INFO, f"> zipf_tail_index= {zipf_tail_index}",
                frac_of_demand_vectors_covered_list=frac_of_demand_vectors_covered_list,
            )
            E_frac_of_demand_vectors_covered_list.append(numpy.mean(frac_of_demand_vectors_covered_list))
            std_frac_of_demand_vectors_covered_list.append(numpy.std(frac_of_demand_vectors_covered_list))

        log(INFO, f"replica_design= {replica_design}",
            zipf_tail_index_list=zipf_tail_index_list,
            E_frac_of_demand_vectors_covered_list=E_frac_of_demand_vectors_covered_list,
            std_frac_of_demand_vectors_covered_list=std_frac_of_demand_vectors_covered_list,
        )
        plot.errorbar(zipf_tail_index_list, E_frac_of_demand_vectors_covered_list, yerr=std_frac_of_demand_vectors_covered_list, color=next(dark_color_cycle), label=f"{replica_design.repr_for_plot()}", marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    # k, d = 21, 3
    # k, d = 111, 3
    k, d = 30, 5
    # k, d = 100, 5
    n = k
    replica_design_list = [
        design.ClusteringDesign(k=k, n=n, d=d),
        design.CyclicDesign(k=k, n=n, d=d),
    ]

    for replica_design in replica_design_list:
        plot_(replica_design=replica_design)

    fontsize = 14
    plot.legend(fontsize=fontsize)
    # plot.yscale("log")
    plot.ylabel("Frac of demands covered", fontsize=fontsize)
    plot.xlabel("Popularity skew index", fontsize=fontsize)

    plot.title(
        f"$D= {cum_demand}$"
        + r", $N_{\textrm{pop}}= $" + fr"$= {num_popular_objs}$"
        + r", $N_{\textrm{sample}}= $" + fr"$= {num_samples}$"
        + r", $N_{\textrm{sim}}= $" + fr"$= {num_popular_objs}$"
    )

    # Save the plot
    plot.gcf().set_size_inches(6, 4)
    file_name = (
        "plots/plot_frac_demand_vectors_covered_vs_d"
        + f"_D_{cum_demand}"
        + f"_k_{k}"
        + f"_p_{num_popular_objs}"
        + f"_d_{d}"
        + ".png"
    )
    plot.savefig(file_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")


if __name__ == "__main__":
    plot_frac_demand_vectors_covered_vs_d_for_different_replication_designs()
