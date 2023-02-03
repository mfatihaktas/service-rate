from src import csv_utils, service_rate, storage_scheme
from src.utils.debug import *


def test_w_frac_of_demand_vectors_in_cap_region(input_dict_for_test_w_csv: dict):
    node_id_to_objs_list = csv_utils.get_node_id_to_objs_list_from_oleg_csv_file(
        csv_file_path=input_dict_for_test_w_csv[
            "csv_file_path_for_node_id_to_objs_list"
        ],
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
        max_repair_set_size=input_dict_for_test_w_csv["max_repair_set_size"],
    )

    obj_demands_list = csv_utils.get_obj_demands_list_from_oleg_csv_file(
        csv_file_path=input_dict_for_test_w_csv["csv_file_path_for_obj_demands_list"],
    )

    def cum_demand_for_top_objects(
        obj_demand_list: list[float],
        frac_of_objects: float,
    ) -> float:
        n = int(len(obj_demand_list) * frac_of_objects)
        return sum(obj_demand_list[:n])

    num_in_cap_region = 0
    for i, obj_demand_list in enumerate(obj_demands_list):
        is_in_cap_region = service_rate_inspector.is_in_cap_region(obj_demand_list)
        num_in_cap_region += int(is_in_cap_region)

        max_load = service_rate_inspector.max_load(obj_demand_list)

        obj_demand_list.sort(reverse=True)

        log(
            DEBUG,
            f"i= {i}",
            # obj_demand_list=obj_demand_list,
            obj_demand_list_len=len(obj_demand_list),
            is_in_cap_region=is_in_cap_region,
            max_load=max_load,
            cum_demand=sum(obj_demand_list),
            cum_demand_for_top_10_percent_objects=cum_demand_for_top_objects(
                obj_demand_list=obj_demand_list,
                frac_of_objects=0.1,
            ),
            cum_demand_for_top_20_percent_objects=cum_demand_for_top_objects(
                obj_demand_list=obj_demand_list,
                frac_of_objects=0.2,
            ),
            cum_demand_for_top_30_percent_objects=cum_demand_for_top_objects(
                obj_demand_list=obj_demand_list,
                frac_of_objects=0.3,
            ),
        )

    log(
        DEBUG,
        "",
        input_dict_for_test_w_csv=input_dict_for_test_w_csv,
        frac_of_demand_vectors_in_cap_region=num_in_cap_region / len(obj_demands_list),
    )
