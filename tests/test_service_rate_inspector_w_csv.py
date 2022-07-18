from src import csv_utils, service_rate, storage_scheme
from src.debug_utils import *

from tests import conftest, node_id_to_objs


def test_w_frac_of_demand_vectors_in_cap_region(input_dict_for_test_w_csv: dict):
    node_id_to_objs_list = csv_utils.get_node_id_to_objs_list_from_oleg_csv_file(
        csv_file_path=input_dict_for_test_w_csv["csv_file_path_for_node_id_to_objs_list"],
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
        max_repair_set_size=1,
    )

    obj_demands_list = csv_utils.get_obj_demands_list_from_oleg_csv_file(
        csv_file_path=input_dict_for_test_w_csv["csv_file_path_for_obj_demands_list"],
    )

    num_in_cap_region = 0
    for i, obj_demand_list in enumerate(obj_demands_list):
        is_in_cap_region = service_rate_inspector.is_in_cap_region(obj_demand_list)
        num_in_cap_region += int(is_in_cap_region)

        max_load = service_rate_inspector.max_load(obj_demand_list)

        log(DEBUG, f"i= {i}",
            # obj_demand_list=obj_demand_list,
            is_in_cap_region=is_in_cap_region,
            max_load=max_load,
        )

    log(DEBUG, "",
        input_dict_for_test_w_csv=input_dict_for_test_w_csv,
        frac_of_demand_vectors_in_cap_region=num_in_cap_region / len(obj_demands_list),
    )
