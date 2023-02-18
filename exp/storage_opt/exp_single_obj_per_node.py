from src.storage_opt import (
    access_graph as access_graph_module,
    data,
    single_obj_per_node as single_obj_per_node_module,
)
from src.model import demand as demand_module

from src.utils.plot import *


def get_access_graph(
    num_objs: int,
    max_demand: float,
) -> access_graph_module.AccessGraph:
    # try:
    #     obj_to_num_copies_map = data.NUM_OBJS_TO_MAX_DEMAND_TO_OBJ_TO_NUM_COPIES_MAP[num_objs][max_demand]
    #     access_graph = access_graph_module.AccessGraph(k=num_objs, obj_to_num_copies_map=obj_to_num_copies_map)
    #     return access_graph

    # except KeyError:
    #     log(WARNING, "Data does not exist, will run optimization", num_objs=num_objs, max_demand=max_demand)

    demand_vector_list = demand_module.get_demand_vectors(
        num_objs=num_objs,
        demand_ordered_for_most_popular_objs=(max_demand,),
    )

    storage_optimizer = single_obj_per_node_module.StorageOptimizerReplicationAndXOR_wSingleObjPerNode(
        demand_vector_list=demand_vector_list,
    )

    return storage_optimizer.access_graph


def plot_access_graphs(num_objs: int):
    log(INFO, "Started", num_obj=num_objs)

    max_demand_list = list(range(1, 21))

    num_plots = len(max_demand_list)
    figsize = (num_plots * 3 * num_objs, 10)
    fig, ax_list = plot.subplots(1, num_plots, figsize=figsize)

    for i, max_demand in enumerate(max_demand_list):
        ax = ax_list[i]
        plot.sca(ax)

        access_graph = get_access_graph(num_objs=num_objs, max_demand=max_demand)
        access_graph.draw()

        plot.title(f"Max demand= {max_demand}, Num nodes= {access_graph.get_total_num_nodes()}")
        # To keep graph nodes within the plot.
        # Ref: https://stackoverflow.com/questions/61520570/networkx-in-a-subplot-is-drawing-nodes-partially-outside-of-axes-frame
        ax.set_xlim([1.1 * x for x in ax.get_xlim()])
        ax.set_ylim([1.1 * y for y in ax.get_ylim()])

    # Save to PNG
    plot.subplots_adjust(wspace=0.5)
    fig.set_size_inches(figsize[0], figsize[1])

    file_name = (
        "plots/access_graphs"
        + f"_k_{num_objs}"
        + f"_max_demand_from_{min(max_demand_list)}_to_{max(max_demand_list)}"
        + ".png"
    )
    plot.savefig(file_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done", num_obj=num_objs)


def plot_num_nodes_vs_max_demand_for_StorageOptimizerReplicationAndXOR_wSingleObjPerNode():
    max_demand_list = list(range(1, 21))

    max_demand_to_object_to_num_copies_map = {}

    def plot_(num_objs: int):
        log(INFO, f"num_objs= {num_objs}")

        num_nodes_list = []

        for max_demand in max_demand_list:
            access_graph = get_access_graph(num_objs=num_objs, max_demand=max_demand)
            max_demand_to_object_to_num_copies_map[max_demand] = access_graph.obj_to_num_copies_map

            num_nodes = sum(access_graph.obj_to_num_copies_map.values())
            log(INFO, f"max_demand= {max_demand}",
                obj_to_num_copies_map=access_graph.obj_to_num_copies_map,
                num_nodes=num_nodes,
            )

            num_nodes_list.append(num_nodes)

        log(INFO, f"num_objs= {num_objs}",
            max_demand_list=max_demand_list,
            num_nodes_list=num_nodes_list,
            max_demand_to_object_to_num_copies_map=max_demand_to_object_to_num_copies_map,
        )

        plot.plot(max_demand_list, num_nodes_list, color=next(dark_color_cycle), label=fr"$k= {num_objs}$", marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    num_objs_list = [3]
    # num_objs_list = [4]
    # num_objs_list = [5, 6]
    # num_objs_list = [3, 4, 5, 6]
    for num_objs in num_objs_list:
        plot_(num_objs=num_objs)

    fontsize = 14
    plot.legend(fontsize=fontsize)
    # plot.yscale("log")
    plot.ylabel("Number of nodes", fontsize=fontsize)
    plot.xlabel("Max demand", fontsize=fontsize)

    plot.title("Replication + XOR's")

    # Save the plot
    plot.gcf().set_size_inches(6, 4)
    file_name = (
        "plots/plot_num_nodes_vs_max_demand_"
        + "k_{}_".format("-".join(str(num) for num in num_objs_list))
        + f"_max_demand_from_{min(max_demand_list)}_to_{max(max_demand_list)}"
        + ".png"
    )
    plot.savefig(file_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")


if __name__ == "__main__":
    plot_num_nodes_vs_max_demand_for_StorageOptimizerReplicationAndXOR_wSingleObjPerNode()
    # plot_access_graphs(num_objs=3)
