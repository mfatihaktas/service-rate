from src import service_rate, storage_scheme
from src.debug_utils import *

from tests import conftest


def test_is_in_cap_region():
    node_id_objs_list = storage_scheme.name_to_node_objs_list_map["a,b_a,b"]

    scheme = storage_scheme.StorageScheme(node_id_objs_list)
    log(DEBUG, "", storage_scheme=scheme)

    inspector = service_rate.ServiceRateInspector(
        m=len(node_id_objs_list),
        C=1,
        G=scheme.obj_encoding_matrix,
        obj_to_node_id_map=scheme.obj_id_to_node_id_map,
    )

    assert inspector.is_in_cap_region([1, 1]) == True
    assert inspector.is_in_cap_region([1.5, 0.35]) == True
    assert inspector.is_in_cap_region([0.25, 2]) == False
    assert inspector.is_in_cap_region([1.8, 1.4]) == False
    assert inspector.is_in_cap_region([2.1, 2]) == False
    assert inspector.is_in_cap_region([1, 2.3]) == False


def test_plot_cap_2d(service_rate_inspector: service_rate.ServiceRateInspector):
    service_rate_inspector.plot_cap_2d()


def test_min_max_functions(service_rate_inspector: service_rate.ServiceRateInspector):
    cum_demand = 0.9*min(len(repair_set_list) for _, repair_set_list in service_rate_inspector.obj_to_repair_sets_map.items())
    log(DEBUG, "", cum_demand=cum_demand)

    for i in range(10):
        obj_demand_list = conftest.sample_obj_demand_list(
            k=service_rate_inspector.k,
            cum_demand=cum_demand
        )

        min_cost = service_rate_inspector.min_cost(obj_demand_list)
        min_dist_w_convex_hull = service_rate_inspector.min_distance_to_boundary_w_convex_hull(obj_demand_list)
        min_dist_w_cvxpy = service_rate_inspector.min_distance_to_boundary_w_cvxpy(obj_demand_list)
        min_dist_approx = service_rate_inspector.approx_min_distance_to_boundary(obj_demand_list)
        max_load = service_rate_inspector.max_load(obj_demand_list)

        log(DEBUG, f"i= {i}",
            obj_demand_list=obj_demand_list,
            min_cost=min_cost,
            min_dist_w_convex_hull=min_dist_w_convex_hull,
            min_dist_w_cvxpy=min_dist_w_cvxpy,
            min_dist_approx=min_dist_approx,
            max_load=max_load,
        )
