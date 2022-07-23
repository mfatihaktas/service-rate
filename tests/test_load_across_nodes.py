from src import csv_utils, service_rate, storage_scheme
from src.debug_utils import *

from tests import conftest, node_id_to_objs


CSV_FOLDER_PATH = "tests/csv/load_across_nodes"

CSV_FILE_PATH_FOR_NODE_ID_TO_OBJS_LIST = f"{CSV_FOLDER_PATH}/SIMRESULT_SERVICE_RATE_REPLICATION_PLACE_PLACEMENT.csv"
CSV_FILE_PATH_FOR_OBJ_DEMANDS_LIST = f"{CSV_FOLDER_PATH}/SIMRESULT_SERVICE_RATE_REPLICATION_PLACE_DEMAND.csv"
MAX_REPAIR_SET_SIZE = 1

# CSV_FILE_PATH_FOR_NODE_ID_TO_OBJS_LIST = f"{CSV_FOLDER_PATH}/SIMRESULT_SERVICE_RATE_CODING_PLACE_PLACEMENT.csv"
# CSV_FILE_PATH_FOR_OBJ_DEMANDS_LIST = f"{CSV_FOLDER_PATH}/SIMRESULT_SERVICE_RATE_CODING_PLACE_DEMAND.csv"
# MAX_REPAIR_SET_SIZE = 2


def test_load_across_nodes():
    node_id_to_objs_list = csv_utils.get_node_id_to_objs_list_from_oleg_csv_file(
        csv_file_path=CSV_FILE_PATH_FOR_NODE_ID_TO_OBJS_LIST,
    )

    scheme = storage_scheme.StorageScheme(node_id_to_objs_list)
    log(DEBUG, "", storage_scheme=scheme)

    m = len(node_id_to_objs_list)
    C = 1
    service_rate_inspector = service_rate.ServiceRateInspector(
        m=m,
        C=C,
        G=scheme.obj_encoding_matrix,
        obj_to_node_id_map=scheme.obj_id_to_node_id_map,
        max_repair_set_size=MAX_REPAIR_SET_SIZE,
    )

    obj_demands_list = csv_utils.get_obj_demands_list_from_oleg_csv_file(
        csv_file_path=CSV_FILE_PATH_FOR_OBJ_DEMANDS_LIST,
    )

    for i, obj_demand_list in enumerate(obj_demands_list):
        log(DEBUG, f"i= {i}",
            # obj_demand_list=obj_demand_list,
            obj_demand_list_len=len(obj_demand_list),
            is_in_cap_region=service_rate_inspector.is_in_cap_region(obj_demand_list),
            load_across_nodes=service_rate_inspector.load_across_nodes(obj_demand_list),
            load_across_nodes_when_obj_demands_distributed_evenly_across_repair_sets=service_rate_inspector.load_across_nodes_when_obj_demands_distributed_evenly_across_repair_sets(obj_demand_list),
        )
