from src.utils.debug import *


def get_obj_id_to_node_id_set_map_w_replicas_clustering_design(
    k: int,
    n: int,
    d: int,
) -> dict[int, set[int]]:
    check(n % d == 0, f"d= {d} must divide n= {n}")

    return {
        obj_id: set(
            i % n for i in range(obj_id * d , obj_id * d + d)
        )
        for obj_id in range(k)
    }


def get_obj_id_to_node_id_set_map_w_replicas_cyclic_design(
    k: int,
    n: int,
    d: int,
) -> dict[int, set[int]]:
    return {
        obj_id: set(i % n for i in range(obj_id, obj_id + d))
        for obj_id in range(k)
    }
