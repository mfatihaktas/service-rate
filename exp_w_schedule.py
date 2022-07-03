import random

from debug_utils import *
from service_rate import ServiceRateInspector
from storage_scheme import Obj, StorageScheme, name_to_node_objs_list_map


def sample_obj_demand_list(
    k: int,
    cum_demand: float,
) -> list[int]:
    obj_demand_list = [random.randint(0, 9) for _ in range(k)]
    sum_demands = sum(obj_demand_list)
    return [d / sum_demands * cum_demand for d in obj_demand_list]


def run(
    node_id_objs_list: list[list[Obj]],
    max_repair_set_size: int = None,
    compute_halfspace_intersections=True,
):
    log(DEBUG,
        "Started;",
        node_id_objs_list=node_id_objs_list,
        max_repair_set_size=max_repair_set_size,
        compute_halfspace_intersections=compute_halfspace_intersections,
    )

    scheme = StorageScheme(node_id_objs_list)
    log(DEBUG, "", storage_scheme=scheme)

    inspector = ServiceRateInspector(
        m=len(node_id_objs_list),
        C=1,
        G=scheme.obj_encoding_matrix,
        obj_to_node_id_map=scheme.obj_id_to_node_id_map,
        max_repair_set_size=max_repair_set_size,
        compute_halfspace_intersections=compute_halfspace_intersections,
    )
    inspector.plot_cap_2d()

    # Log min cost/dist/dist_approx etc.
    [k, n] = scheme.obj_encoding_matrix.shape
    cum_demand = 1.2*min(len(repair_set_list) for _, repair_set_list in inspector.obj_to_repair_sets_map.items())
    log(DEBUG, "", cum_demand=cum_demand)
    for i in range(10):
        obj_demand_list = sample_obj_demand_list(k, cum_demand)
        min_cost = inspector.min_cost(obj_demand_list)
        min_dist = inspector.min_distance_to_boundary(obj_demand_list)
        min_dist_approx = inspector.approx_min_distance_to_boundary(obj_demand_list)
        log(DEBUG, f"i= {i}",
            obj_demand_list=obj_demand_list,
            min_cost=min_cost,
            min_dist=min_dist,
            min_dist_approx=min_dist_approx
        )

    # for obj_demand_list in inspector.get_cap_boundary_point_list():
    #     inspector.min_cost(obj_demand_list)

    log(DEBUG, "Done.")

if __name__ == "__main__":
    run(name_to_node_objs_list_map["a_b_a+b"])
    # run(name_to_node_objs_list_map["a_a_b_b"])
    # run(name_to_node_objs_list_map["a_a_a_b_a+b_a+2b"])
    # run(name_to_node_objs_list_map["a,b_a,b"])
