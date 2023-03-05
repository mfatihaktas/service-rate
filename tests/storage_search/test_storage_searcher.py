import pytest

from src.storage_search import storage_searcher as storage_searcher_module

from src.utils.debug import *


@pytest.fixture(
    scope="session",
    params=[
        # [[1, 2]],
        # [[1, 2.9]],
        # [[2, 0], [0, 2]],
        # [[3, 0], [0, 3]],
        # [[1, 3, 4, 2]],
        # [[3.5, 1, 0.2, 0.1]],

        # [
        #     [4, 1, 0.3, 0.2],
        #     [0.2, 0.3, 1, 4],
        # ],

        # [
        #     [3, 1, 1, 1],
        #     [1, 3, 1, 1],
        #     [1, 1, 3, 1],
        #     [1, 1, 1, 3],
        # ],

        # [
        #     [5, 0.3, 0.3, 0.2, 0.1],
        #     [0.1, 0.2, 0.3, 0.3, 5],
        # ],

        # [
        #     [5, 0.3, 0.3, 0.2, 0.1],
        #     [0.1, 0.2, 0.3, 0.3, 5],
        #     [0.1, 0.2, 5, 0.3, 0.3],
        # ],

        # [
        #     [2, 0],
        #     [0, 2],
        # ],

        # [
        #     [2, 0, 0.1],
        #     [0, 2, 0.1],
        # ],

        # [
        #     [3, 0],
        #     [0, 1.9],
        # ],

        # [
        #     [10, 0.1, 0.1],
        #     [0.1, 0.1, 2],
        #     [0.1, 20, 0.1],
        # ],

        # [
        #     [10, 0.1, 0.1],
        #     [0.1, 0.1, 10],
        # ],

        # [
        #     [4, 0, 0],
        #     [0, 4, 0],
        #     [0, 0, 4],
        # ],

        [
            [9, 0, 0],
            [0, 9, 0],
            [0, 0, 9],
        ],

        # [
        #     [10, 0.1, 0.1, 0.1],
        #     [0.1, 0.1, 0.1, 10],
        #     [0.1, 10, 0.1, 0.1],
        # ],

        # [
        #     [10, 0, 0, 0],
        #     [0, 10, 0, 0],
        #     [0, 0, 8, 0],
        #     [0, 0, 0, 8],
        # ],

        # [
        #     [8, 1, 0, 0],
        #     [0, 8, 1, 0],
        #     [0, 0, 8, 1],
        #     [1, 0, 0, 8],
        # ],

        # [
        #     [3, 0.1, 0.1, 0.1, 0.1],
        #     [0.1, 0.1, 0.1, 0.1, 3],
        #     [0.1, 0.1, 3, 0.1, 0.1],
        #     [0.1, 3, 0.1, 0.1, 0.1],
        #     [0.1, 0.1, 0.1, 10, 0.1],
        # ],

        # [
        #     [5, 0.1, 0.1, 0.1, 0.1],
        #     [0.1, 0.1, 0.1, 0.1, 5],
        #     [0.1, 0.1, 5, 0.1, 0.1],
        #     [0.1, 5, 0.1, 0.1, 0.1],
        #     [0.1, 0.1, 0.1, 10, 0.1],
        # ],
    ],
)
def demand_vector_list(request) -> list[list[float]]:
    return request.param


def test_SearchStorageWithReplicasAndTwoXORs(demand_vector_list: list[list[float]]):
    storage_searcher = storage_searcher_module.SearchStorageWithReplicasAndTwoXORs(
        demand_vector_list=demand_vector_list
    )

    # node_id_to_objs_list = storage_searcher.get_node_id_to_objs_list_w_brute_force()
    node_id_to_objs_list = storage_searcher.get_node_id_to_objs_list()
    log(DEBUG, "Done", node_id_to_objs_list=node_id_to_objs_list)


def test_SearchStorageWithReplicasAndMDS(demand_vector_list: list[list[float]]):
    storage_searcher = storage_searcher_module.SearchStorageWithReplicasAndMDS(
        demand_vector_list=demand_vector_list,
        num_independent_mds_objs=1,
    )

    # node_id_to_objs_list = storage_searcher.get_node_id_to_objs_list_w_brute_force()
    node_id_to_objs_list = storage_searcher.get_node_id_to_objs_list()
    log(DEBUG, "Done", node_id_to_objs_list=node_id_to_objs_list)


def test_SearchStorageWithReplicasAndXORs(demand_vector_list: list[list[float]]):
    storage_searcher = storage_searcher_module.SearchStorageWithReplicasAndXORs(
        demand_vector_list=demand_vector_list,
    )

    # node_id_to_objs_list = storage_searcher.get_node_id_to_objs_list_w_brute_force()
    node_id_to_objs_list = storage_searcher.get_node_id_to_objs_list()
    log(DEBUG, "Done", node_id_to_objs_list=node_id_to_objs_list)
