from src.storage_overlap import design
from src.model import demand

from src.utils.plot import *


def plot_frac_demand_vectors_covered_vs_d_for_different_replication_designs():
    num_popular_objs = 2
    cum_demand = 3
    # zipf_tail_index_list = [0, 1, 2]
    zipf_tail_index_list = [0, 0.5, 1, 1.5, 2]

    log(INFO, "Started",
        num_popular_objs=num_popular_objs,
        cum_demand=cum_demand,
        zipf_tail_index_list=zipf_tail_index_list,
    )

    def plot_(replica_design: design.ReplicaDesign):
        log(INFO, f">> replica_design= {replica_design}")

        frac_of_demand_vectors_covered_list = []

        for zipf_tail_index in zipf_tail_index_list:
            ## With `get_demand_vectors_w_zipf_law`
            # demand_vector_list = demand.get_demand_vectors_w_zipf_law(
            #     num_objs=replica_design.k,
            #     num_popular_objs=num_popular_objs,
            #     cum_demand=cum_demand,
            #     zipf_tail_index=zipf_tail_index,
            # )
            # frac_of_demand_vectors_covered = replica_design.frac_of_demand_vectors_covered(demand_vector_list=demand_vector_list)

            ## With `frac_of_demand_vectors_covered_w_generator_input`
            frac_of_demand_vectors_covered = replica_design.frac_of_demand_vectors_covered_w_generator_input(
                demand_vector_generator=demand.demand_vector_generator_w_zipf_law(
                    num_objs=replica_design.k,
                    num_popular_objs=num_popular_objs,
                    cum_demand=cum_demand,
                    zipf_tail_index=zipf_tail_index,
                )
            )

            log(INFO, f"> zipf_tail_index= {zipf_tail_index}",
                frac_of_demand_vectors_covered=frac_of_demand_vectors_covered,
            )
            frac_of_demand_vectors_covered_list.append(frac_of_demand_vectors_covered)

        log(INFO, f"replica_design= {replica_design}",
            zipf_tail_index_list=zipf_tail_index_list,
            frac_of_demand_vectors_covered_list=frac_of_demand_vectors_covered_list,
        )
        plot.plot(zipf_tail_index_list, frac_of_demand_vectors_covered_list, color=next(dark_color_cycle), label=f"{replica_design.repr_for_plot()}", marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    k = 9  # 21
    n = k
    d = 3
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

    plot.title(f"D= {cum_demand}")

    # Save the plot
    plot.gcf().set_size_inches(6, 4)
    file_name = (
        "plots/plot_frac_demand_vectors_covered_vs_d"
        + f"_D_{cum_demand}"
        + f"_k_{k}"
        + ".png"
    )
    plot.savefig(file_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")


if __name__ == "__main__":
    plot_frac_demand_vectors_covered_vs_d_for_different_replication_designs()
