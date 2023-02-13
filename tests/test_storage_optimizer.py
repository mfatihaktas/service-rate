import pytest

from src.opt_storage import (
    single_obj_per_node as single_obj_per_node_module,
    storage_optimizer as storage_optimizer_module,
)
from src.service_rate import (
    service_rate,
)
from src.utils.debug import *


@pytest.fixture(
    scope="session",
    params=[
        # [[1, 2]],
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

        [
            [8, 1, 0, 0],
            [0, 8, 1, 0],
            [0, 0, 8, 1],
            [1, 0, 0, 8],
        ],

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


def test_StorageOptimizerReplication(demand_vector_list: list[float]):
    storage_optimizer = storage_optimizer_module.StorageOptimizerReplication(
        demand_vector_list=demand_vector_list,
        max_num_nodes_factor=1,
    )
    obj_id_to_node_id_set_map = storage_optimizer.get_obj_id_to_node_id_set_map()
    log(INFO, "With minimal # nodes", obj_id_to_node_id_set_map=obj_id_to_node_id_set_map)

    storage_optimizer = storage_optimizer_module.StorageOptimizerReplication(
        demand_vector_list=demand_vector_list,
        max_num_nodes_factor=2,
    )
    obj_id_to_node_id_set_map = storage_optimizer.get_obj_id_to_node_id_set_map()
    log(INFO, "With minimal # object copies", obj_id_to_node_id_set_map=obj_id_to_node_id_set_map)


def test_StorageOptimizerReplicationAndMDS(demand_vector_list: list[float]):
    storage_optimizer = storage_optimizer_module.StorageOptimizerReplicationAndMDS(
        demand_vector_list=demand_vector_list,
        mds_node_capacity=2,
    )
    obj_id_to_node_id_set_map = storage_optimizer.get_obj_id_to_node_id_set_map()
    log(DEBUG, "With minimal # nodes", obj_id_to_node_id_set_map=obj_id_to_node_id_set_map)

    storage_optimizer = storage_optimizer_module.StorageOptimizerReplicationAndMDS(
        demand_vector_list=demand_vector_list,
        mds_node_capacity=1,
    )
    obj_id_to_node_id_set_map = storage_optimizer.get_obj_id_to_node_id_set_map()
    log(DEBUG, "With minimal # object copies", obj_id_to_node_id_set_map=obj_id_to_node_id_set_map)


def test_StorageOptimizerReplicationAndMDS_wSingleObjPerNode(demand_vector_list: list[float]):
    storage_optimizer = storage_optimizer_module.StorageOptimizerReplicationAndMDS_wSingleObjPerNode(demand_vector_list=demand_vector_list)
    # num_sys_and_mds_nodes = storage_optimizer.get_num_sys_and_mds_nodes()
    # log(DEBUG, "", num_sys_and_mds_nodes=num_sys_and_mds_nodes)

    node_id_to_objs_list = storage_optimizer.get_node_id_to_objs_list()
    log(DEBUG, "", node_id_to_objs_list=node_id_to_objs_list)

    """
    storage_scheme = storage_scheme_module.StorageScheme(node_id_to_objs_list=node_id_to_objs_list)
    log(DEBUG, "", storage_scheme=storage_scheme)

    inspector = service_rate.ServiceRateInspector(
        m=len(node_id_to_objs_list),
        C=1,
        G=storage_scheme.obj_encoding_matrix,
        obj_id_to_node_id_map=storage_scheme.obj_id_to_node_id_map,
        redundancy_w_two_xors=False,
    )

    # assert inspector.is_in_cap_region([5, 0.3, 0, 0, 0])

    for demand_vector in demand_vector_list:
    # for demand_vector in [[5, 1, 0, 0, 0]]:
        assert inspector.is_in_cap_region(demand_vector)
        # load_across_nodes = inspector.load_across_nodes(demand_vector)
        # log(DEBUG, "", demand_vector=demand_vector, load_across_nodes=load_across_nodes)
    """


def test_StorageOptimizerReplicationAnd2XORs(demand_vector_list: list[float]):
    # With minimal number of nodes
    storage_optimizer_w_minimal_num_nodes = storage_optimizer_module.StorageOptimizerReplicationAnd2XORs(
        demand_vector_list=demand_vector_list,
        max_num_nodes_factor=1,
    )

    obj_id_to_node_id_set_map, xor_to_node_id_set_map = storage_optimizer_w_minimal_num_nodes.get_obj_id_to_node_id_set_map_and_xor_to_node_id_set_map()
    log(INFO, "With minimal # nodes",
        obj_id_to_node_id_set_map=obj_id_to_node_id_set_map,
        xor_to_node_id_set_map=xor_to_node_id_set_map
    )

    # With minimal number of object copies
    storage_optimizer_w_minimal_num_obj_copies = storage_optimizer_module.StorageOptimizerReplicationAnd2XORs(
        demand_vector_list=demand_vector_list,
        max_num_nodes_factor=2,
    )

    obj_id_to_node_id_set_map, xor_to_node_id_set_map = storage_optimizer_w_minimal_num_obj_copies.get_obj_id_to_node_id_set_map_and_xor_to_node_id_set_map()
    log(INFO, "With minimal # object copies",
        obj_id_to_node_id_set_map=obj_id_to_node_id_set_map,
        xor_to_node_id_set_map=xor_to_node_id_set_map
    )


def test_StorageOptimizerReplicationAndXOR_wSingleObjPerNode(demand_vector_list: list[float]):
    storage_optimizer = single_obj_per_node_module.StorageOptimizerReplicationAndXOR_wSingleObjPerNode(
        demand_vector_list=demand_vector_list,
    )

    object_to_num_copies_map = storage_optimizer.get_object_to_num_copies_map()
    log(INFO, "",
        object_to_num_copies_map=object_to_num_copies_map,
    )

    m = {}
    m[single_obj_per_node_module.SysObject(symbol=0)] = 1
    m[single_obj_per_node_module.XORedObject(symbols=(0, 1))] = 2
    log(DEBUG, "", m=m)
