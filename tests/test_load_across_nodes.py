from src import service_rate, service_rate_utils, storage_scheme
from src.utils import csv
from src.utils.debug import *


CSV_FOLDER_PATH = "tests/csv/load_across_nodes"

# CSV_FILE_PATH_FOR_NODE_ID_TO_OBJS_LIST = f"{CSV_FOLDER_PATH}/SIMRESULT_SERVICE_RATE_REPLICATION_PLACE_PLACEMENT.csv"
# CSV_FILE_PATH_FOR_OBJ_DEMANDS_LIST = f"{CSV_FOLDER_PATH}/SIMRESULT_SERVICE_RATE_REPLICATION_PLACE_DEMAND.csv"
# MAX_REPAIR_SET_SIZE = 1

CSV_FILE_PATH_FOR_NODE_ID_TO_OBJS_LIST = (
    f"{CSV_FOLDER_PATH}/SIMRESULT_SERVICE_RATE_CODING_PLACE_PLACEMENT.csv"
)
CSV_FILE_PATH_FOR_OBJ_DEMANDS_LIST = (
    f"{CSV_FOLDER_PATH}/SIMRESULT_SERVICE_RATE_CODING_PLACE_DEMAND.csv"
)
MAX_REPAIR_SET_SIZE = 2


def test_load_across_nodes():
    node_id_to_objs_list = csv.get_node_id_to_objs_list_from_oleg_csv_file(
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
        obj_id_to_node_id_map=scheme.obj_id_to_node_id_map,
        max_repair_set_size=MAX_REPAIR_SET_SIZE,
    )

    service_rate_utils.get_orig_obj_id_to_repair_sets_w_node_ids_map(
        orig_obj_id_to_repair_sets_w_obj_ids_map=service_rate_inspector.orig_obj_id_to_repair_sets_w_obj_ids_map,
        obj_id_to_node_id_map=service_rate_inspector.obj_id_to_node_id_map,
    )

    obj_demands_list = [1]
    # csv.get_obj_demands_list_from_oleg_csv_file(
    #     csv_file_path=CSV_FILE_PATH_FOR_OBJ_DEMANDS_LIST,
    # )

    for i, obj_demand_list in enumerate(obj_demands_list):
        log(
            DEBUG,
            f"i= {i}",
            # obj_demand_list=obj_demand_list,
            obj_demand_list_len=len(obj_demand_list),
            cum_demand=sum(obj_demand_list),
            is_in_cap_region=service_rate_inspector.is_in_cap_region(obj_demand_list),
            load_across_nodes=service_rate_inspector.load_across_nodes(obj_demand_list),
            load_across_nodes_when_obj_demands_distributed_evenly_across_repair_sets=service_rate_inspector.load_across_nodes_when_obj_demands_distributed_evenly_across_repair_sets(
                obj_demand_list
            ),
            load_across_nodes_when_obj_demands_replicated_to_repair_sets=service_rate_inspector.load_across_nodes_when_obj_demands_replicated_to_repair_sets(
                obj_demand_list
            ),
        )


# def test_load_on_first_node():
#     node_id_to_objs_list = csv.get_node_id_to_objs_list_from_oleg_csv_file(
#         csv_file_path=CSV_FILE_PATH_FOR_NODE_ID_TO_OBJS_LIST,
#     )

#     scheme = storage_scheme.StorageScheme(node_id_to_objs_list)
#     log(DEBUG, "",
#         storage_scheme=scheme,
#         # obj_id_to_node_id_map=scheme.obj_id_to_node_id_map,
#     )

#     m = len(node_id_to_objs_list)
#     C = 1
#     service_rate_inspector = service_rate.ServiceRateInspector(
#         m=m,
#         C=C,
#         G=scheme.obj_encoding_matrix,
#         obj_id_to_node_id_map=scheme.obj_id_to_node_id_map,
#         max_repair_set_size=MAX_REPAIR_SET_SIZE,
#     )


#     obj_demands_list = csv.get_obj_demands_list_from_oleg_csv_file(
#         csv_file_path=CSV_FILE_PATH_FOR_OBJ_DEMANDS_LIST,
#     )
#     obj_demand_list = obj_demands_list[0]

#     cum_demand_on_node_0 = 0
#     orig_obj_id_to_demand_map = {}
#     for orig_obj_id in range(service_rate_inspector.k):
#         for repair_set_w_node_ids in service_rate_inspector.orig_obj_id_to_repair_sets_w_node_ids_map[orig_obj_id]:
#             demand = obj_demand_list[orig_obj_id]

#             if 0 in repair_set_w_node_ids:
#                 orig_obj_id_to_demand_map[orig_obj_id] = demand
#                 cum_demand_on_node_0 += demand

#     log(INFO, "",
#         cum_demand_on_node_0=cum_demand_on_node_0,
#         orig_obj_id_to_demand_map=orig_obj_id_to_demand_map,
#         len_orig_obj_id_to_demand_map=len(orig_obj_id_to_demand_map),
#     )
