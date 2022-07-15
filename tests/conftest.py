import random
import pytest

from src import service_rate, storage_scheme
from src.debug_utils import *

from tests import node_id_to_objs as node_id_to_objs


def sample_obj_demand_list(
    k: int,
    cum_demand: float,
) -> list[float]:
    obj_demand_list = [random.randint(1, 9) for _ in range(k)]
    sum_demands = sum(obj_demand_list)
    return [d / sum_demands * cum_demand for d in obj_demand_list]


@pytest.fixture(
    scope="function",
    params=[
        "a_b_a+b",
        # "a_a_b_b",
        # "a_b_a+b_a+2b",
        # "a_a_a_b_a+b_a+2b",
        # "a,b_a,b",
    ],
)
def node_id_to_objs_list(request) -> str:
    return storage_scheme.name_to_node_objs_list_map[request.param]


@pytest.fixture
def service_rate_inspector(
    node_id_to_objs_list: list[list[storage_scheme.Obj]],
    max_repair_set_size: int = None,
) -> service_rate.ServiceRateInspector:
    max_repair_set_size = None
    compute_halfspace_intersections = True

    scheme = storage_scheme.StorageScheme(node_id_to_objs_list)
    log(DEBUG, "", storage_scheme=scheme)

    return service_rate.ServiceRateInspector(
        m=len(node_id_to_objs_list),
        C=1,
        G=scheme.obj_encoding_matrix,
        obj_to_node_id_map=scheme.obj_id_to_node_id_map,
        max_repair_set_size=max_repair_set_size,
        compute_halfspace_intersections=compute_halfspace_intersections,
    )


@pytest.fixture(
    scope="function",
    params=[
        # Simplistic redundancy schemes

        # {
        #     "node_id_to_objs_list": node_id_to_objs.node_id_to_objs_list_1,
        #     "max_repair_set_size": 2,
        # },

        # {
        #     "node_id_to_objs_list": node_id_to_objs.get_random_node_id_to_objs_list_w_two_xors(
        #         num_original_objs=10,
        #         num_replicas=10,
        #         num_xors=10,
        #         num_nodes=10,
        #     ),
        #     "max_repair_set_size": 2,
        # },

        # Redundancy with replication

        # {
        #     "node_id_to_objs_list": node_id_to_objs.get_random_node_id_to_objs_list_w_two_xors(
        #         num_original_objs=200,
        #         num_replicas=200,
        #         num_xors=0,
        #         num_nodes=100,
        #     ),
        #     "max_repair_set_size": 1,
        # },

        # {
        #     "node_id_to_objs_list": node_id_to_objs.get_random_node_id_to_objs_list_w_two_xors(
        #         num_original_objs=1000,
        #         num_replicas=1000,
        #         num_xors=0,
        #         num_nodes=100,
        #     ),
        #     "max_repair_set_size": 1,
        # },

        # Redundancy with 2-XOR's

        {
            "node_id_to_objs_list": node_id_to_objs.get_random_node_id_to_objs_list_w_two_xors(
                num_original_objs=200,
                num_replicas=200,
                num_xors=100,
                num_nodes=100,
            ),
            "max_repair_set_size": 2,
        },
    ],
)
def input_dict(request) -> dict:
    return request.param
