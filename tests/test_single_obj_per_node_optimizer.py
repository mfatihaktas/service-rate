import pytest

from src.storage_opt import (
    single_obj_per_node as single_obj_per_node_module,
)
from src.model import demand as demand_module

from src.utils.debug import *
from src.utils.plot import *


@pytest.fixture(
    scope="session",
    params=[
        {
            "num_objects": 3,
            "demand_ordered_for_most_popular_objs": [9, 0],
        }
    ],
)
def demand_vector_list(request) -> list[list[float]]:
    input_map = request.param

    return demand_module.get_demand_vectors(
        num_objs=input_map["num_objects"],
        demand_ordered_for_most_popular_objs=input_map["demand_ordered_for_most_popular_objs"],
    )


def test_StorageOptimizerReplicationAndXOR_wSingleObjPerNode(demand_vector_list: list[float]):
    log(DEBUG, "", demand_vector_list=demand_vector_list)

    storage_optimizer = single_obj_per_node_module.StorageOptimizerReplicationAndXOR_wSingleObjPerNode(
        demand_vector_list=demand_vector_list,
    )

    object_to_num_copies_map = storage_optimizer.get_object_to_num_copies_map()
    log(INFO, "",
        object_to_num_copies_map=object_to_num_copies_map,
    )


def test_num_nodes_vs_max_demand_for_StorageOptimizerReplicationAndXOR_wSingleObjPerNode():
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

    for num_objs in [3, 4, 5, 6]:
        plot_(num_objs=num_objs)

    fontsize = 14
    plot.legend(fontsize=fontsize)
    # plot.yscale("log")
    plot.ylabel("Number of nodes", fontsize=fontsize)
    plot.xlabel("Max demand", fontsize=fontsize)

    plot.title("Replication + XOR's")

    # Save the plot
    plot.gcf().set_size_inches(6, 4)
    plot.savefig("plot_num_nodes_vs_max_demand.png", bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")
