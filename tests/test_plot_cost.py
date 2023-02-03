from src import service_rate, storage_scheme
from src.utils import plot_cost
from src.utils.debug import *


def test_plot_cost_2d():
    node_id_to_objs_list = storage_scheme.name_to_node_objs_list_map["a_b_a+b"]

    scheme = storage_scheme.StorageScheme(node_id_to_objs_list)
    log(DEBUG, "", storage_scheme=scheme)

    service_rate_inspector = service_rate.ServiceRateInspector(
        m=len(node_id_to_objs_list),
        C=1,
        G=scheme.obj_encoding_matrix,
        obj_id_to_node_id_map=scheme.obj_id_to_node_id_map,
        compute_halfspace_intersections=True,
    )

    plot_cost.plot_cost_2d(
        service_rate_inspector=service_rate_inspector,
        file_name_suffix="a_b_a+b",
    )
