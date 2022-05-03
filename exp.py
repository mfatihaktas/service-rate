from csv_utils import (
    get_node_id_objs_list_from_oleg_csv_file,
    get_obj_demands_list_from_oleg_csv_file,
)
from debug_utils import *
from service_rate import ServiceRateInspector
from storage_scheme import Obj, StorageScheme, name_to_node_objs_list_map
from pathlib import Path
import pandas as pd

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
    outfile = createResultFilePath(csv_file_path_for_obj_demands_list)
    demDF = pd.read_csv(csv_file_path_for_obj_demands_list)
    demDF["inside"] = 0

    num_is_in_cap_region = 0
    for i, obj_demand_list in enumerate(obj_demands_list):
        is_in_cap_region = inspector.is_in_cap_region(obj_demand_list)
        # log(DEBUG, f"demand-vector-{i}: is_in_cap_region= {is_in_cap_region}")
        num_is_in_cap_region += int(is_in_cap_region)
        demDF.loc[i,"inside"] = is_in_cap_region

    demDF.to_csv(outfile,index=False)



    log(
        DEBUG,
        "Done; ",
        freq_is_in_cap_region=num_is_in_cap_region / len(obj_demands_list),
    )

def createResultFilePath(filename):
    expname = Path(filename).name.split('.')
    expname = expname[0] + "_result.csv"
    outfile = Path(filename).parent / expname
    return outfile

if __name__ == "__main__":
    # run(name_to_node_objs_list_map["a_b_a+b"])
    # run(name_to_node_objs_list_map["a_a_b_b"])
    # run(name_to_node_objs_list_map["a_a_a_b_a+b_a+2b"])
    # run(name_to_node_objs_list_map["a,b_a,b"])

    # csv_file_path_for_node_id_objs_list = "csv/exp1_rep_12nodes_placement.csv"
    # csv_file_path_for_obj_demands_list = "csv/exp1_rep_12nodes_demand.csv"

    # csv_file_path_for_node_id_objs_list = "csv/exp2_ec_9nodes_placement.csv"
    # csv_file_path_for_obj_demands_list = "csv/exp2_ec_9nodes_demand.csv"

    # csv_file_path_for_node_id_objs_list = "csv/exp3_rep_6nodes_placement.csv"
    # csv_file_path_for_node_id_objs_list = "csv/exp3_ec_6nodes_placement.csv"
    # csv_file_path_for_obj_demands_list = "csv/exp3_6nodes_demand.csv"

    csv_file_path_for_node_id_objs_list_coding = "csv/SIMRESULT_SERVICE_RATE_CODING_PLACE_PLACEMENT.csv"
    csv_file_path_for_obj_demands_list_coding = "csv/SIMRESULT_SERVICE_RATE_CODING_PLACE_DEMAND.csv"

    run_w_csv_file_path(
        csv_file_path_for_node_id_objs_list_coding,
        csv_file_path_for_obj_demands_list_coding
    )

    csv_file_path_for_node_id_objs_list_replication = "csv/SIMRESULT_SERVICE_RATE_REPLICATION_PLACE_PLACEMENT.csv"
    csv_file_path_for_obj_demands_list_replication = "csv/SIMRESULT_SERVICE_RATE_REPLICATION_PLACE_DEMAND.csv"

    run_w_csv_file_path(
        csv_file_path_for_node_id_objs_list_replication,
        csv_file_path_for_obj_demands_list_replication

    codingResFile = createResultFilePath(csv_file_path_for_obj_demands_list_coding)
    repResFile = createResultFilePath(csv_file_path_for_obj_demands_list_replication)
    df1 = pd.read_csv(codingResFile)
    df2 = pd.read_csv(repResFile)
    df = pd.concat([df1, df2])
    df = df.sort_values(by=list(df.columns[:-3]))
    df.reset_index(inplace=True, drop=True)
    df.to_csv("csv/experiment_output.csv",index=False)
