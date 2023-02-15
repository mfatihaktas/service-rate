from src.storage_opt import single_obj_per_node as single_obj_per_node_module
from src.model import demand as demand_module

from src.utils.plot import *


def plot_num_nodes_vs_max_demand_for_StorageOptimizerReplicationAndXOR_wSingleObjPerNode():
    max_demand_list = list(range(1, 21))

    def plot_(num_objs: int):
        log(INFO, f"num_objs= {num_objs}")

        num_nodes_list = []

        for max_demand in max_demand_list:
            demand_vector_list = demand_module.get_demand_vectors(
                num_objs=num_objs,
                demand_ordered_for_most_popular_objs=[max_demand],
            )

            storage_optimizer = single_obj_per_node_module.StorageOptimizerReplicationAndXOR_wSingleObjPerNode(
                demand_vector_list=demand_vector_list,
            )

            object_to_num_copies_map = storage_optimizer.get_object_to_num_copies_map()

            num_nodes = sum(object_to_num_copies_map.values())
            log(INFO, f"max_demand= {max_demand}", num_nodes=num_nodes)

            num_nodes_list.append(num_nodes)

        plot.plot(max_demand_list, num_nodes_list, color=next(dark_color_cycle), label=fr"$k= {num_objs}$", marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    # num_objs_list = [3]
    num_objs_list = [3, 4, 5, 6]
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
    title = (
        "plot_num_nodes_vs_max_demand_"
        + "k_{}_".format("-".join(str(num) for num in num_objs_list))
        + f"_max_demand_from_{min(max_demand_list)}_to_{max(max_demand_list)}"
        + ".png"
    )
    plot.savefig(title, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")



if __name__ == "__main__":
    plot_num_nodes_vs_max_demand_for_StorageOptimizerReplicationAndXOR_wSingleObjPerNode()
