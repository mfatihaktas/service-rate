import pytest

from typing import Tuple

from src.service_rate import (
    plot_capacity_region,
    service_rate,
    service_rate_w_stripe,
    storage_scheme,
)

from src.utils.debug import *


@pytest.fixture(
    scope="session",
    params=[
        # "a_a_b",
        # "a_b_a+b",
        # "a_a_b_b",
        # "a_a_a_b_a+b_a+2b",
        # "a_a_a_b_b_b_a+b_a+2b",
        "a_b_c_a+b_a+c_b+c",
    ],
)
def storage_name(request) -> str:
    return request.param


@pytest.fixture
def storage_name_and_service_rate_inspector(storage_name: str) -> Tuple[str, service_rate.ServiceRateInspector]:
    node_id_to_objs_list = storage_scheme.name_to_node_objs_list_map[storage_name]

    scheme = storage_scheme.StorageScheme(node_id_to_objs_list)
    log(DEBUG, "", storage_scheme=scheme)

    service_rate_inspector = service_rate.ServiceRateInspector(
        m=len(node_id_to_objs_list),
        C=1,
        G=scheme.obj_encoding_matrix,
        obj_id_to_node_id_map=scheme.obj_id_to_node_id_map,
    )

    return (storage_name, service_rate_inspector)


@pytest.fixture
def storage_name_and_service_rate_inspector_w_stripe() -> Tuple[str, service_rate_w_stripe.ServiceRateInspectorForStorageWithStripeAndParity]:
    obj_id_to_node_id_set_map = {
        0: {0, 1, 2},
        # 1: {0, 1, 2},
        1: {1, 2, 3},
    }

    service_rate_inspector_w_stripe = service_rate_w_stripe.ServiceRateInspectorForStorageWithStripeAndParity(
        k=2,
        m=4,
        s=2,
        obj_id_to_node_id_set_map=obj_id_to_node_id_set_map,
    )

    return ("storage_w_stripe", service_rate_inspector_w_stripe)


def test_plot_cost_capacity_region_2d(storage_name_and_service_rate_inspector):
    storage_name, service_rate_inspector = storage_name_and_service_rate_inspector

    plot_capacity_region.plot_capacity_region_2d(
        service_rate_inspector=service_rate_inspector,
        file_name_suffix=storage_name,
    )


def test_plot_cost_capacity_region_2d_alternative(storage_name_and_service_rate_inspector):
    storage_name, service_rate_inspector = storage_name_and_service_rate_inspector

    plot_capacity_region.plot_capacity_region_2d_alternative(
        service_rate_inspector=service_rate_inspector,
        # obj_id_list=[0, 1],
        obj_id_list=[1, 2],
        file_name_suffix=storage_name,
    )


def test_plot_cost_capacity_region_2d_alternative_for_storage_w_stripe(
    storage_name_and_service_rate_inspector_w_stripe,
):
    storage_name, service_rate_inspector_w_stripe = storage_name_and_service_rate_inspector_w_stripe

    plot_capacity_region.plot_capacity_region_2d_alternative(
        service_rate_inspector=service_rate_inspector_w_stripe,
        obj_id_list=[0, 1],
        file_name_suffix=storage_name,
    )
