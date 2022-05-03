from csv_utils import (
    get_node_id_objs_list_from_oleg_csv_file,
    get_obj_demands_list_from_oleg_csv_file,
)
from debug_utils import *
from service_rate import ServiceRateInspector
from storage_scheme import Obj, StorageScheme, name_to_node_objs_list_map


def run(node_id_objs_list: list[list[Obj]]):
    log(DEBUG, "Started;", node_id_objs_list=node_id_objs_list)

    scheme = StorageScheme(node_id_objs_list)
    log(DEBUG, "", storage_scheme=scheme)

    inspector = ServiceRateInspector(
        m=len(node_id_objs_list),
        C=1,
        G=scheme.obj_encoding_matrix,
        obj_to_node_id_map=scheme.obj_id_to_node_id_map,
    )
    inspector.plot_cap_2d()

    log(DEBUG, "Done.")


def run_w_csv_file_path(
    csv_file_path_for_node_id_objs_list: str,
    csv_file_path_for_obj_demands_list: str,
):
    log(
        DEBUG,
        "Started;",
        csv_file_path_for_node_id_objs_list=csv_file_path_for_node_id_objs_list,
        csv_file_path_for_obj_demands_list=csv_file_path_for_obj_demands_list,
    )

    node_id_objs_list = get_node_id_objs_list_from_oleg_csv_file(
        csv_file_path_for_node_id_objs_list
    )
    log(DEBUG, "", node_id_objs_list=node_id_objs_list)

    scheme = StorageScheme(node_id_objs_list)
    log(DEBUG, "", storage_scheme=scheme)

    inspector = ServiceRateInspector(
        m=len(node_id_objs_list),
        C=1,
        G=scheme.obj_encoding_matrix,
        obj_to_node_id_map=scheme.obj_id_to_node_id_map,
    )

    obj_demands_list = get_obj_demands_list_from_oleg_csv_file(
        csv_file_path_for_obj_demands_list
    )

    num_is_in_cap_region = 0
    for i, obj_demand_list in enumerate(obj_demands_list):
        is_in_cap_region = inspector.is_in_cap_region(obj_demand_list)
        # log(DEBUG, f"demand-vector-{i}: is_in_cap_region= {is_in_cap_region}")

        num_is_in_cap_region += int(is_in_cap_region)

    log(
        DEBUG,
        "Done; ",
        freq_is_in_cap_region=num_is_in_cap_region / len(obj_demands_list),
    )


if __name__ == "__main__":
    # run(name_to_node_objs_list_map["a_b_a+b"])
    # run(name_to_node_objs_list_map["a_a_b_b"])
    # run(name_to_node_objs_list_map["a_a_a_b_a+b_a+2b"])
    # run(name_to_node_objs_list_map["a,b_a,b"])

    csv_file_path_for_node_id_objs_list = "./exp1_rep_12nodes_placement.csv"
    csv_file_path_for_obj_demands_list = "./exp1_rep_12nodes_demand.csv"
    run_w_csv_file_path(
        csv_file_path_for_node_id_objs_list,
        csv_file_path_for_obj_demands_list
    )

    log(DEBUG, "asdasdasd")
