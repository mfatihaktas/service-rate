from src.storage_search import storage_searcher as storage_searcher_module

from src.utils.debug import *


def get_demand_vector_list() -> list[list[float]]:
    demand_vector_list = [
        [4, 1, 0.3, 0.2],
        [0.2, 0.3, 1, 4],
    ]

    # demand_vector_list = [
    #     [3, 1, 1, 1],
    #     [1, 3, 1, 1],
    #     [1, 1, 3, 1],
    #     [1, 1, 1, 3],
    # ]

    # demand_vector_list = [
    #     [10, 0.1, 0.1],
    #     [0.1, 0.1, 2],
    #     [0.1, 20, 0.1],
    # ]

    # demand_vector_list = [
    #     [4, 0, 0],
    #     [0, 4, 0],
    #     [0, 0, 4],
    # ]

    # demand_vector_list = [
    #     [3, 0.1, 0.1, 0.1, 0.1],
    #     [0.1, 0.1, 0.1, 0.1, 3],
    #     [0.1, 0.1, 3, 0.1, 0.1],
    #     [0.1, 3, 0.1, 0.1, 0.1],
    #     [0.1, 0.1, 0.1, 10, 0.1],
    # ]

    return demand_vector_list


def exp_SearchStorageWithReplicasAndTwoXORs():
    demand_vector_list = get_demand_vector_list()

    storage_searcher = storage_searcher_module.SearchStorageWithReplicasAndTwoXORs(
        demand_vector_list=demand_vector_list
    )

    # node_id_to_objs_list = storage_searcher.get_node_id_to_objs_list_w_brute_force()
    node_id_to_objs_list = storage_searcher.get_node_id_to_objs_list()
    log(DEBUG, "Done", node_id_to_objs_list=node_id_to_objs_list)


def exp_SearchStorageWithReplicasAndMDS():
    demand_vector_list = get_demand_vector_list()

    storage_searcher = storage_searcher_module.SearchStorageWithReplicasAndMDS(
        demand_vector_list=demand_vector_list,
        num_independent_mds_objs=1,
    )

    node_id_to_objs_list = storage_searcher.get_node_id_to_objs_list_w_brute_force()
    # node_id_to_objs_list = storage_searcher.get_node_id_to_objs_list()
    log(DEBUG, "Done", node_id_to_objs_list=node_id_to_objs_list)


if __name__ == "__main__":
    # exp_SearchStorageWithReplicasAndTwoXORs()
    exp_SearchStorageWithReplicasAndMDS()
