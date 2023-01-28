import collections

from src import (
    service_rate,
    storage_scheme as storage_scheme_module,
)
from src.debug_utils import *


def get_service_rate_inspector(
    storage_scheme: storage_scheme_module.StorageScheme,
    redundancy_w_two_xors: bool,
    max_repair_set_size: int = None,
) -> storage_scheme_module.StorageScheme:
    return service_rate.ServiceRateInspector(
        m=storage_scheme.num_nodes,
        C=1,
        G=storage_scheme.obj_encoding_matrix,
        obj_id_to_node_id_map=storage_scheme.obj_id_to_node_id_map,
        redundancy_w_two_xors=redundancy_w_two_xors,
        max_repair_set_size=max_repair_set_size,
    )


def remove_repair_sets_of_size_two(
    obj_to_repair_sets_map: dict[int, list]
) -> dict[int, list]:
    obj_to_repair_sets_map_ = collections.defaultdict(list)
    for obj, repair_set_list in obj_to_repair_sets_map.items():
        for repair_set in repair_set_list:
            if len(repair_set) < 3:
                obj_to_repair_sets_map_[obj].append(repair_set)

    return obj_to_repair_sets_map_


def are_repair_sets_same(
    repair_set_list_1: list[set[int]],
    repair_set_list_2: list[set[int]],
) -> bool:
    def hash_repair_set(repair_set: set[int]) -> str:
        sorted_node_id_list = sorted(list(repair_set))
        return ",".join(str(node_id) for node_id in sorted_node_id_list)

    repair_set_hash_set = set(
        hash_repair_set(repair_set) for repair_set in repair_set_list_1
    )

    for repair_set in repair_set_list_2:
        hash_ = hash_repair_set(repair_set)
        if hash_ not in repair_set_hash_set:
            log(
                ERROR,
                "Repair set is in `repair_set_list_2` but not in `repair_set_list_1`",
                repair_set=repair_set,
            )
            return False

        else:
            repair_set_hash_set.remove(hash_)

    if repair_set_hash_set:
        log(
            ERROR,
            "There are repair sets which are in `repair_set_list_1` but not in `repair_set_list_2`",
            repair_set_hash_set=repair_set_hash_set,
        )
        return False

    return True


def test_service_rate_inspector_on_redundancy_w_two_xors(
    input_dict_for_redundancy_w_two_xors: dict,
):
    storage_scheme = storage_scheme_module.StorageScheme(
        node_id_to_objs_list=input_dict_for_redundancy_w_two_xors[
            "node_id_to_objs_list"
        ],
    )
    log(DEBUG, "", storage_scheme=storage_scheme)

    service_rate_inspector_w_redundancy_w_two_xors_false = get_service_rate_inspector(
        storage_scheme=storage_scheme,
        redundancy_w_two_xors=False,
        max_repair_set_size=input_dict_for_redundancy_w_two_xors["max_repair_set_size"],
    )

    service_rate_inspector_w_redundancy_w_two_xors_true = get_service_rate_inspector(
        storage_scheme=storage_scheme,
        redundancy_w_two_xors=True,
    )

    obj_to_repair_sets_map_w_false = (
        service_rate_inspector_w_redundancy_w_two_xors_false.orig_obj_id_to_repair_sets_w_obj_ids_map
    )
    obj_to_repair_sets_map_w_false = remove_repair_sets_of_size_two(
        obj_to_repair_sets_map_w_false
    )
    obj_to_repair_sets_map_w_true = (
        service_rate_inspector_w_redundancy_w_two_xors_true.orig_obj_id_to_repair_sets_w_obj_ids_map
    )

    for obj, repair_sets_w_false in obj_to_repair_sets_map_w_false.items():
        repair_sets_w_true = obj_to_repair_sets_map_w_true[obj]

        log(
            DEBUG,
            f"obj= {obj}",
            repair_sets_w_false=repair_sets_w_false,
            repair_sets_w_true=repair_sets_w_true,
        )

        assert (
            are_repair_sets_same(
                repair_set_list_1=repair_sets_w_false,
                repair_set_list_2=repair_sets_w_true,
            )
            is True
        )
