from pathlib import Path
import pandas as pd
import sys

from src.service_rate import service_rate, storage_scheme
from src.utils import csv
from src.debug_utils import *
from os.path import exists


def run(
    node_id_objs_list: list[list[storage_scheme.Obj]],
    max_repair_set_size: int = None,
    compute_halfspace_intersections=False,
):
    log(
        DEBUG,
        "Started;",
        node_id_objs_list=node_id_objs_list,
        max_repair_set_size=max_repair_set_size,
        compute_halfspace_intersections=compute_halfspace_intersections,
    )

    scheme = storage_scheme.StorageScheme(node_id_objs_list)
    log(DEBUG, "", storage_scheme=scheme)

    inspector = service_rate.ServiceRateInspector(
        m=len(node_id_objs_list),
        C=1,
        G=scheme.obj_encoding_matrix,
        obj_to_node_id_map=scheme.obj_id_to_node_id_map,
        max_repair_set_size=max_repair_set_size,
        compute_halfspace_intersections=compute_halfspace_intersections,
    )
    inspector.plot_cap_2d()

    log(DEBUG, "Done.")


def run_w_csv_file_path(
    csv_file_path_for_node_id_objs_list: str,
    csv_file_path_for_obj_demands_list: str,
    max_repair_set_size=None,
    compute_halfspace_intersections=False,
):
    log(
        DEBUG,
        "Started;",
        csv_file_path_for_node_id_objs_list=csv_file_path_for_node_id_objs_list,
        csv_file_path_for_obj_demands_list=csv_file_path_for_obj_demands_list,
        max_repair_set_size=max_repair_set_size,
        compute_halfspace_intersections=compute_halfspace_intersections,
    )

    node_id_objs_list = csv.get_node_id_to_objs_list_from_oleg_csv_file(
        csv_file_path_for_node_id_objs_list
    )
    # log(DEBUG, "", node_id_objs_list=node_id_objs_list)

    scheme = storage_scheme.StorageScheme(node_id_objs_list)
    log(DEBUG, "", storage_scheme=scheme)

    inspector = service_rate.ServiceRateInspector(
        m=len(node_id_objs_list),
        C=1,
        G=scheme.obj_encoding_matrix,
        obj_to_node_id_map=scheme.obj_id_to_node_id_map,
        compute_halfspace_intersections=compute_halfspace_intersections,
        max_repair_set_size=max_repair_set_size,
    )

    obj_demands_list = csv.get_obj_demands_list_from_oleg_csv_file(
        csv_file_path_for_obj_demands_list
    )
    outfile = createResultFilePath(csv_file_path_for_obj_demands_list)
    demDF = pd.read_csv(csv_file_path_for_obj_demands_list)
    demDF["inside"] = 0
    demDF["mCost"] = 0

    num_is_in_cap_region = 0
    for i, obj_demand_list in enumerate(obj_demands_list):
        is_in_cap_region = inspector.is_in_cap_region(obj_demand_list)
        min_cost = inspector.min_cost(obj_demand_list)
        # min_distance = inspector.min_distance_to_boundary(obj_demand_list)
        # min_distance = inspector.approx_min_distance_to_boundary(obj_demand_list)
        min_distance = inspector.min_distance_to_boundary_w_cvxpy(obj_demand_list)

        log(
            DEBUG,
            f"demand-vector-{i}: \n"
            f"\t obj_demand_list= {obj_demand_list} \n"
            f"\t is_in_cap_region= {is_in_cap_region} \n"
            f"\t min_cost= {min_cost} \n"
            f"\t min_distance= {min_distance}",
        )

        num_is_in_cap_region += int(is_in_cap_region)
        demDF.loc[i, "inside"] = is_in_cap_region
        demDF.loc[i, "mCost"] = min_cost
        demDF.loc[i, "distance"] = min_distance

    demDF.to_csv(outfile, index=False)

    log(
        DEBUG,
        "Done; ",
        freq_is_in_cap_region=num_is_in_cap_region / len(obj_demands_list),
    )


def createResultFilePath(filename):
    expname = Path(filename).name.split(".")
    expname = expname[0] + "_result.csv"
    outfile = Path(filename).parent / expname
    return outfile


def run_w_sim_result_csv_files(basedir="csv"):
    compute_halfspace_intersections = False  # True

    csv_file_path_for_node_id_objs_list_replication = (
        basedir + "/SIMRESULT_SERVICE_RATE_REPLICATION_PLACE_PLACEMENT.csv"
    )
    csv_file_path_for_obj_demands_list_replication = (
        basedir + "/SIMRESULT_SERVICE_RATE_REPLICATION_PLACE_DEMAND.csv"
    )
    max_repair_set_size = 1

    run_w_csv_file_path(
        csv_file_path_for_node_id_objs_list_replication,
        csv_file_path_for_obj_demands_list_replication,
        max_repair_set_size=max_repair_set_size,
        compute_halfspace_intersections=compute_halfspace_intersections,
    )

    csv_file_path_for_node_id_objs_list_coding = (
        basedir + "/SIMRESULT_SERVICE_RATE_CODING_PLACE_PLACEMENT.csv"
    )
    csv_file_path_for_obj_demands_list_coding = (
        basedir + "/SIMRESULT_SERVICE_RATE_CODING_PLACE_DEMAND.csv"
    )

    csv_file_demand_ec_orbit = (
        basedir + "/SIMRESULT_SERVICE_RATE_CODING_PLACE_DEMAND_ORBIT.csv"
    )
    csv_file_demand_rep_orbit = (
        basedir + "/SIMRESULT_SERVICE_RATE_REPLICATION_PLACE_DEMAND_ORBIT.csv"
    )

    csv_file_demand_ec_orbit_place = (
        basedir + "/SIMRESULT_SERVICE_RATE_CODING_PLACE_PLACEMENT_ORBIT.csv"
    )
    csv_file_demand_rep_orbit_place = (
        basedir + "/SIMRESULT_SERVICE_RATE_REPLICATION_PLACE_PLACEMENT_ORBIT.csv"
    )

    max_repair_set_size = 2

    run_w_csv_file_path(
        csv_file_path_for_node_id_objs_list_coding,
        csv_file_path_for_obj_demands_list_coding,
        max_repair_set_size=max_repair_set_size,
        compute_halfspace_intersections=compute_halfspace_intersections,
    )

    repResFile = createResultFilePath(csv_file_path_for_obj_demands_list_replication)
    df1 = pd.read_csv(repResFile)
    codingResFile = createResultFilePath(csv_file_path_for_obj_demands_list_coding)
    df2 = pd.read_csv(codingResFile)
    df = pd.concat([df1, df2])
    df = df.sort_values(by=["iteration", "type"])
    df.reset_index(inplace=True, drop=True)

    if exists(csv_file_demand_ec_orbit) and exists(csv_file_demand_rep_orbit):
        dforb1 = pd.read_csv(csv_file_demand_ec_orbit, index_col=False)
        dforb2 = pd.read_csv(csv_file_demand_rep_orbit, index_col=False)
        dforb = pd.concat([dforb1, dforb2])
        dforb = dforb.sort_values(by=["iteration", "type"])
        dforb.reset_index(inplace=True, drop=True)
        placementEquals = True
        dfPlaceSim = pd.read_csv(csv_file_path_for_node_id_objs_list_replication)
        dfPlaceOrbit = pd.read_csv(csv_file_demand_rep_orbit_place)
        if not dfPlaceSim.equals(dfPlaceOrbit):
            placementEquals = False
        dfPlaceSim = pd.read_csv(csv_file_path_for_node_id_objs_list_coding)
        dfPlaceOrbit = pd.read_csv(csv_file_demand_ec_orbit_place)
        if not dfPlaceSim.equals(dfPlaceOrbit):
            placementEquals = False

        for row in range(dforb.shape[0]):
            if (
                not (dforb.iloc[row, 0:10] == df.iloc[row, 0:10]).all()
                or not placementEquals
            ):
                print("ERROR: ORBIT csv doesn't equal sim")
            else:
                df["orbit"] = dforb["completed"]
                # df["distanceCost"] = dforb["distanceCost"]
                df["orbitCost"] = dforb["serviceCost"]
                df["orbitLatency"] = dforb["latency"]
        # df["adjustedCost"] = dforb["cost"] * dforb["reqsPerUserSec"]
    df["modelCost"] = df["mCost"].fillna(value=0)
    df.drop(columns=["mCost"], inplace=True)
    df.to_csv(basedir + "/experiment_output.csv", index=False)


if __name__ == "__main__":
    # run(name_to_node_objs_list_map["a_b_a+b"])
    # run(name_to_node_objs_list_map["a_a_b_b"])
    # run(name_to_node_objs_list_map["a_a_a_b_a+b_a+2b"])
    # run(name_to_node_objs_list_map["a,b_a,b"])

    # csv_file_path_for_node_id_objs_list = "csv/exp1_rep_12nodes_placement.csv"
    # csv_file_path_for_obj_demands_list = "csv/exp1_rep_12nodes_demand.csv"
    # max_repair_set_size = 1

    # csv_file_path_for_node_id_objs_list = "csv/exp2_ec_9nodes_placement.csv"
    # csv_file_path_for_obj_demands_list = "csv/exp2_ec_9nodes_demand.csv"
    # max_repair_set_size = 2

    # csv_file_path_for_node_id_objs_list = "csv/exp3_rep_6nodes_placement.csv"
    # csv_file_path_for_node_id_objs_list = "csv/exp3_ec_6nodes_placement.csv"
    # csv_file_path_for_obj_demands_list = "csv/exp3_6nodes_demand.csv"

    # run_w_csv_file_path(
    #     csv_file_path_for_node_id_objs_list,
    #     csv_file_path_for_obj_demands_list,
    #     max_repair_set_size,
    # )

    if len(sys.argv) > 1:
        for i in range(1, len(sys.argv)):
            run_w_sim_result_csv_files(sys.argv[i])
    else:
        run_w_sim_result_csv_files()
