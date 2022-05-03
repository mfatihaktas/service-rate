import collections
import csv

from debug_utils import *
from storage_scheme import CodedObj, Obj, PlainObj


def get_node_id_objs_list_from_oleg_csv_file(csv_file_path: str) -> list[list[Obj]]:
    """CSV file is structured as:
    object-0, object-1, node

    Each row represents an object.
    If object-1 is -1: object is a plain copy/replica.
    else: object is an XOR of object-0 and object-1.
    """

    node_id_to_objs_map = collections.defaultdict(list)

    with open(csv_file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue

            [obj_id_0, obj_id_1, node_id] = [int(i) for i in row]
            if obj_id_1 == -1:
                obj = PlainObj(id_str=chr(ord("a") + obj_id_0))
            else:
                obj = CodedObj(
                    coeff_obj_list=[
                        (1, PlainObj(id_str=chr(ord("a") + obj_id_0))),
                        (1, PlainObj(id_str=chr(ord("a") + obj_id_1))),
                    ]
                )

            node_id_to_objs_map[node_id].append(obj)

            line_count += 1

    log(DEBUG, "Done;", line_count=line_count)

    max_node_id = max(node_id for node_id in node_id_to_objs_map)
    return [node_id_to_objs_map[i] for i in range(max_node_id + 1)]


def get_obj_demands_list_from_oleg_csv_file(csv_file_path: str) -> list[list[float]]:
    """CSV file is structured as:
    demand-0, demand-1, ..., demand-(k - 1)

    Each row represents a separate object demand vector.
    """

    obj_demands_list = []

    line_count = 0
    with open(csv_file_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue

            obj_demands_list.append([float(s) for s in row])

            line_count += 1

    log(DEBUG, "Done;", line_count=line_count)

    return obj_demands_list
