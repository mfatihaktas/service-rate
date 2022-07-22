import random
import pytest

from src import service_rate, storage_scheme
from src.debug_utils import *

from tests import node_id_to_objs


def sample_obj_demand_list_w_skewed_popularity(
    k: int,
    frac_of_popular_objects: float,
    cum_demand: float,
    frac_of_cum_demand_by_popular_objects: float,
) -> list[float]:
    cum_demand_by_popular_objs = frac_of_cum_demand_by_popular_objects * cum_demand
    cum_demand_by_other_objs = cum_demand - cum_demand_by_popular_objs

    num_popular_objs = int(frac_of_popular_objects * k)
    _popular_obj_demand_list = [random.randint(1, 9) for _ in range(num_popular_objs)]
    demand_sum = sum(_popular_obj_demand_list)
    popular_obj_demand_list = [d / demand_sum * cum_demand_by_popular_objs for d in _popular_obj_demand_list]

    num_other_objs =  k - num_popular_objs
    _other_obj_demand_list = [random.randint(1, 9) for _ in range(num_other_objs)]
    demand_sum = sum(_other_obj_demand_list)
    other_obj_demand_list = [d / demand_sum * cum_demand_by_other_objs for d in _other_obj_demand_list]

    return [*popular_obj_demand_list, *other_obj_demand_list]


def sample_obj_demand_list(
    k: int,
    cum_demand: float,
) -> list[float]:
    obj_demand_list = [random.randint(1, 9) for _ in range(k)]
    demand_sum = sum(obj_demand_list)
    return [d / demand_sum * cum_demand for d in obj_demand_list]


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
def node_id_to_objs_list(request) -> list[list]:
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
def input_dict_for_redundancy_w_two_xors(request) -> dict:
    return request.param


@pytest.fixture(
    scope="function",
    params=[
        # {
        #     "num_nodes": 100,
        #     "replication_factor": 2,
        #     "cumulative_load_factor": 0.4,
        # },

        # {
        #     "num_nodes": 100,
        #     "replication_factor": 2,
        #     "cumulative_load_factor": 0.75,
        # },

        # {
        #     "num_nodes": 100,
        #     "replication_factor": 2,
        #     "cumulative_load_factor": 0.8,
        # },

        # {
        #     "num_nodes": 100,
        #     "replication_factor": 5,
        #     "cumulative_load_factor": 0.75,
        # },

        # {
        #     "num_nodes": 100,
        #     "replication_factor": 5,
        #     "cumulative_load_factor": 0.9,
        # },

        # {
        #     "num_nodes": 100,
        #     "replication_factor": 10,
        #     "cumulative_load_factor": 0.8,
        # },

        # {
        #     "num_nodes": 100,
        #     "replication_factor": 10,
        #     "cumulative_load_factor": 0.95,
        # },

        {
            "num_nodes": 10,
            "num_original_objs": 500,
            "replication_factor": 2,
            "cumulative_load_factor": 0.5,
            "frac_of_popular_objects": 0.5,
            "frac_of_cum_demand_by_popular_objects": 0.8,
        },
    ],
)
def input_dict_for_round_robin_design(request) -> dict:
    return request.param


CSV_FOLDER_PATH = "tests/csv"

@pytest.fixture(
    scope="function",
    params=[
        # {
        #     "csv_file_path_for_node_id_to_objs_list": f"{CSV_FOLDER_PATH}/small/SIMRESULT_SERVICE_RATE_REPLICATION_PLACE_PLACEMENT.csv",
        #     "csv_file_path_for_obj_demands_list": f"{CSV_FOLDER_PATH}/small/SIMRESULT_SERVICE_RATE_REPLICATION_PLACE_DEMAND.csv",
        #     "max_repair_set_size": 1,
        # },

        {
            "csv_file_path_for_node_id_to_objs_list": f"{CSV_FOLDER_PATH}/large/SIMRESULT_SERVICE_RATE_REPLICATION_PLACE_PLACEMENT.csv",
            "csv_file_path_for_obj_demands_list": f"{CSV_FOLDER_PATH}/large/SIMRESULT_SERVICE_RATE_REPLICATION_PLACE_DEMAND.csv",
            "max_repair_set_size": 1,
        },
    ],
)
def input_dict_for_test_w_csv(request) -> dict:
    return request.param
