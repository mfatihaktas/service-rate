from src.storage_search import storage_searcher as storage_searcher_module

from src.utils.debug import *


def exp_SearchStorageWithReplicasAndTwoXORs():
    demand_vector_list = [
        [4, 1, 0.3, 0.2],
        [0.2, 0.3, 1, 4],
    ]

    storage_searcher = storage_searcher_module.SearchStorageWithReplicasAndTwoXORs(
        demand_vector_list=demand_vector_list
    )

    # node_id_to_objs_list = storage_searcher.get_node_id_to_objs_list_w_brute_force()
    node_id_to_objs_list = storage_searcher.get_node_id_to_objs_list()
    log(DEBUG, "Done", node_id_to_objs_list=node_id_to_objs_list)


if __name__ == "__main__":
    exp_SearchStorageWithReplicasAndTwoXORs()
