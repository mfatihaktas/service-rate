from src.service_rate import service_rate, storage_scheme
from src.debug_utils import *

from tests import conftest, node_id_to_objs


def test_is_in_cap_region():
    node_id_to_objs_list = storage_scheme.name_to_node_objs_list_map["a,b_a,b"]

    scheme = storage_scheme.StorageScheme(node_id_to_objs_list)
    log(DEBUG, "", storage_scheme=scheme)

    inspector = service_rate.ServiceRateInspector(
        m=len(node_id_to_objs_list),
        C=1,
        G=scheme.obj_encoding_matrix,
        obj_id_to_node_id_map=scheme.obj_id_to_node_id_map,
    )

    assert inspector.is_in_cap_region([1, 1]) is True
    assert inspector.is_in_cap_region([1.5, 0.35]) is True
    assert inspector.is_in_cap_region([0.25, 2]) is False
    assert inspector.is_in_cap_region([1.8, 1.4]) is False
    assert inspector.is_in_cap_region([2.1, 2]) is False
    assert inspector.is_in_cap_region([1, 2.3]) is False


def test_plot_cap_2d(service_rate_inspector: service_rate.ServiceRateInspector):
    service_rate_inspector.plot_cap_2d()


def test_min_max_functions(service_rate_inspector: service_rate.ServiceRateInspector):
    cum_demand = 0.9 * min(
        len(repair_set_list)
        for _, repair_set_list in service_rate_inspector.orig_obj_id_to_repair_sets_w_obj_ids_map.items()
    )
    log(DEBUG, "", cum_demand=cum_demand)

    for i in range(10):
        obj_demand_list = conftest.sample_obj_demand_list(
            k=service_rate_inspector.k,
            cum_demand=cum_demand,
        )

        log(
            DEBUG,
            f"i= {i}",
            obj_demand_list=obj_demand_list,
            min_cost=service_rate_inspector.min_cost(obj_demand_list),
            min_dist_w_convex_hull=service_rate_inspector.min_distance_to_boundary_w_convex_hull(
                obj_demand_list
            ),
            min_dist_w_cvxpy=service_rate_inspector.min_distance_to_boundary_w_cvxpy(
                obj_demand_list
            ),
            min_dist_approx=service_rate_inspector.approx_min_distance_to_boundary(
                obj_demand_list
            ),
            max_load=service_rate_inspector.max_load(obj_demand_list),
            load_across_nodes=service_rate_inspector.load_across_nodes(obj_demand_list),
        )


NUM_DEMAND_VECTORS = 100


def test_w_frac_of_demand_vectors_in_cap_region(
    input_dict_for_round_robin_design: dict,
):
    node_id_to_objs_list = (
        node_id_to_objs.get_node_id_to_objs_list_w_round_robin_design(
            num_original_objs=input_dict_for_round_robin_design["num_original_objs"],
            num_nodes=input_dict_for_round_robin_design["num_nodes"],
            replication_factor=input_dict_for_round_robin_design["replication_factor"],
        )
    )

    scheme = storage_scheme.StorageScheme(node_id_to_objs_list)
    log(
        DEBUG,
        "",
        storage_scheme=scheme,
        node_id_to_objs_list=node_id_to_objs_list,
        obj_id_to_node_id_map=scheme.obj_id_to_node_id_map,
    )

    m = len(node_id_to_objs_list)
    C = 1
    service_rate_inspector = service_rate.ServiceRateInspector(
        m=m,
        C=C,
        G=scheme.obj_encoding_matrix,
        obj_id_to_node_id_map=scheme.obj_id_to_node_id_map,
        max_repair_set_size=1,
    )

    cum_demand = input_dict_for_round_robin_design["cumulative_load_factor"] * m * C

    num_in_cap_region = 0
    for i in range(NUM_DEMAND_VECTORS):
        # obj_demand_list = conftest.sample_obj_demand_list(
        #     k=service_rate_inspector.k,
        #     cum_demand=cum_demand,
        # )

        obj_demand_list = conftest.sample_obj_demand_list_w_skewed_popularity(
            k=service_rate_inspector.k,
            frac_of_popular_objects=input_dict_for_round_robin_design[
                "frac_of_popular_objects"
            ],
            cum_demand=cum_demand,
            frac_of_cum_demand_by_popular_objects=input_dict_for_round_robin_design[
                "frac_of_cum_demand_by_popular_objects"
            ],
        )

        is_in_cap_region = service_rate_inspector.is_in_cap_region(obj_demand_list)
        num_in_cap_region += int(is_in_cap_region)

        _ = service_rate_inspector.max_load(obj_demand_list)

        # log(DEBUG, f"i= {i}",
        #     # obj_demand_list=obj_demand_list,
        #     is_in_cap_region=is_in_cap_region,
        #     max_load=max_load,
        # )

    log(
        DEBUG,
        "",
        input_dict_for_round_robin_design=input_dict_for_round_robin_design,
        frac_of_demand_vectors_in_cap_region=num_in_cap_region / NUM_DEMAND_VECTORS,
    )
