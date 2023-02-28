import pytest

from src.service_rate import plot_capacity_region, service_rate, storage_scheme

from src.utils.debug import *


@pytest.mark.parametrize(
    "simple_storage_name",
    [
        "a_a_b",
        "a_b_a+b",
        "a_a_b_b",
    ],
)
def test_plot_cost_capacity_region_2d(simple_storage_name: str):
    node_id_to_objs_list = storage_scheme.name_to_node_objs_list_map[
        simple_storage_name
    ]

    scheme = storage_scheme.StorageScheme(node_id_to_objs_list)
    log(DEBUG, "", storage_scheme=scheme)

    service_rate_inspector = service_rate.ServiceRateInspector(
        m=len(node_id_to_objs_list),
        C=1,
        G=scheme.obj_encoding_matrix,
        obj_id_to_node_id_map=scheme.obj_id_to_node_id_map,
        compute_halfspace_intersections=True,
    )

    plot_capacity_region.plot_capacity_region_2d(
        service_rate_inspector=service_rate_inspector,
        file_name_suffix=simple_storage_name,
    )
