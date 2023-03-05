from src.storage_search import storage_searcher as storage_searcher_module

from src.utils.debug import *
from src.utils.plot import *


def plot_num_nodes_vs_num_independent_mds_objs(
    demand_vector_lists: list[list[list[float]]],
):
    log(DEBUG, "Started", demand_vector_lists=demand_vector_lists)

    num_independent_mds_objs_list = list(range(10))

    num_plots = len(demand_vector_lists)
    fig_size = (num_plots * 5, 5)
    fig, ax_list = plot.subplots(1, num_plots, figsize=fig_size)

    def plot_(plot_index: int):
        demand_vector_list = demand_vector_lists[plot_index]
        log(DEBUG, "Started", plot_index=plot_index, demand_vector_list=demand_vector_list)

        num_nodes_list = []

        for num_independent_mds_objs in num_independent_mds_objs_list:
            storage_searcher = storage_searcher_module.SearchStorageWithReplicasAndMDS(
                demand_vector_list=demand_vector_list,
                num_independent_mds_objs=num_independent_mds_objs,
            )

            # node_id_to_objs_list = storage_searcher.get_node_id_to_objs_list_w_brute_force()
            node_id_to_objs_list = storage_searcher.get_node_id_to_objs_list()
            num_nodes = len(node_id_to_objs_list)
            log(DEBUG, "", num_nodes=num_nodes, node_id_to_objs_list=node_id_to_objs_list)

            num_nodes_list.append(num_nodes)

        ax = ax_list[plot_index]
        plot.sca(ax)

        plot.plot(num_independent_mds_objs_list, num_nodes_list, color=next(dark_color_cycle), marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

        fontsize = 14
        plot.xlabel("Number of independent MDS objects", fontsize=fontsize)
        plot.ylabel("Number of nodes", fontsize=fontsize)

        title = f"Demand= {demand_vector_list}"
        plot.title(title, fontsize=fontsize)

    for plot_index in range(num_plots):
        plot_(plot_index=plot_index)

    plot.savefig("plots/plot_num_nodes_vs_num_independent_mds_objs.png", bbox_inches="tight")
    plot.gcf().clear()

    log(DEBUG, "Done")


if __name__ == "__main__":
    demand_vector_lists = []
    # for high_demand in range(3, 8):
    #     for low_demand in [0, high_demand // 2, high_demand * 3 // 4]:
    #         demand_vector_lists.append(
    #             [
    #                 [high_demand, low_demand],
    #                 [low_demand, high_demand],
    #             ]
    #         )
    for high_demand in range(2, 14):
        demand_vector_lists.append(
            [
                [high_demand, 0],
                [0, high_demand],
            ]
        )

    plot_num_nodes_vs_num_independent_mds_objs(
        demand_vector_lists=demand_vector_lists,
    )
