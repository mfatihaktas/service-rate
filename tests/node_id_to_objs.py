import random

from src import storage_scheme as storage_scheme_module
from src.utils.debug import *


node_id_to_objs_list_1 = [
    [storage_scheme_module.PlainObj(id_str="a")],
    [storage_scheme_module.PlainObj(id_str="b")],
    [storage_scheme_module.PlainObj(id_str="c")],
    [storage_scheme_module.PlainObj(id_str="a")],
    [storage_scheme_module.PlainObj(id_str="b")],
    [storage_scheme_module.PlainObj(id_str="c")],
    [
        storage_scheme_module.CodedObj(
            coeff_obj_list=[
                (1, storage_scheme_module.PlainObj(id_str="a")),
                (1, storage_scheme_module.PlainObj(id_str="b")),
            ]
        )
    ],
    [
        storage_scheme_module.CodedObj(
            coeff_obj_list=[
                (1, storage_scheme_module.PlainObj(id_str="a")),
                (1, storage_scheme_module.PlainObj(id_str="c")),
            ]
        )
    ],
    [
        storage_scheme_module.CodedObj(
            coeff_obj_list=[
                (1, storage_scheme_module.PlainObj(id_str="b")),
                (1, storage_scheme_module.PlainObj(id_str="c")),
            ]
        )
    ],
]


def get_random_node_id_to_objs_list_w_two_xors(
    num_original_objs: int,
    num_replicas: int,
    num_xors: int,
    num_nodes: int,
) -> list[list]:
    log(
        DEBUG,
        "",
        num_original_objs=num_original_objs,
        num_replicas=num_replicas,
        num_xors=num_xors,
        num_nodes=num_nodes,
    )

    obj_list = []

    def get_random_obj_id():
        return random.randint(0, num_original_objs - 1)

    # Add original objs
    for obj_id in range(num_original_objs):
        obj_list.append(storage_scheme_module.PlainObj(id_str=str(obj_id)))

    # Add the replicas
    for _ in range(num_replicas):
        obj_list.append(storage_scheme_module.PlainObj(id_str=str(get_random_obj_id())))

    # Add the XOR's
    def get_random_obj_ids_for_xor():
        obj_id_1 = get_random_obj_id()
        obj_id_2 = get_random_obj_id()
        while obj_id_2 == obj_id_1:
            obj_id_2 = get_random_obj_id()

        return obj_id_1, obj_id_2

    for _ in range(num_xors):
        obj_id_1, obj_id_2 = get_random_obj_ids_for_xor()

        obj_list.append(
            storage_scheme_module.CodedObj(
                coeff_obj_list=[
                    (1, storage_scheme_module.PlainObj(id_str=str(obj_id_1))),
                    (1, storage_scheme_module.PlainObj(id_str=str(obj_id_2))),
                ]
            )
        )

    # Assign objects to nodes
    node_id_to_objs_list = [[] for _ in range(num_nodes)]

    def get_random_node_id():
        return random.randint(0, num_nodes - 1)

    for obj in obj_list:
        node_id = get_random_node_id()
        node_id_to_objs_list[node_id].append(obj)

    # log(DEBUG, "",
    #     node_to_objs={
    #         f"node-{i}": obj_list
    #         for i, obj_list in enumerate(node_id_to_objs_list)
    #     },
    # )
    return node_id_to_objs_list


def get_node_id_to_objs_list_w_round_robin_design(
    num_nodes: int,
    num_original_objs: int,
    replication_factor: int,  # number of copies stored for each object
) -> list[list]:
    log(
        DEBUG,
        "",
        num_nodes=num_nodes,
        num_original_objs=num_original_objs,
        replication_factor=replication_factor,
    )

    check(
        num_original_objs % num_nodes == 0,
        "Number of original objects should be a multiple of number of nodes",
        num_original_objs=num_original_objs,
        num_nodes=num_nodes,
    )

    node_id_to_objs_list = [[] for _ in range(num_nodes)]

    for original_obj_id in range(num_original_objs):
        for replica_id in range(replication_factor):
            node_id_ = (original_obj_id + replica_id) % num_nodes

            node_id_to_objs_list[node_id_].append(
                storage_scheme_module.PlainObj(id_str=str(original_obj_id))
            )

    # log(DEBUG, "",
    #     node_to_objs={
    #         f"node-{i}": obj_list
    #         for i, obj_list in enumerate(node_id_to_objs_list)
    #     },
    # )
    return node_id_to_objs_list
