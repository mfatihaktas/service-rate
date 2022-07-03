import numpy
import scipy.spatial

from debug_utils import *


def get_T(
    num_objects: int, # k
    num_repair_sets: int,
    obj_to_repair_set_size_map: dict[int, int],
) -> numpy.array:
    T = numpy.zeros((num_objects, num_repair_sets))
    i = 0
    for obj in range(num_objects):
        j = i + obj_to_repair_set_size_map[obj]
        T[obj, i:j] = 1
        i = j
    # print(f"T= {self.T}")

    return T


def get_M(
    num_objects: int, # n
    num_nodes: int,
    repair_set_list: list[set],
    obj_to_node_id_map: dict[int, int],
) -> numpy.array:
    M = numpy.zeros((num_nodes, len(repair_set_list)))

    for obj in range(num_objects):
        for repair_set_index, repair_set in enumerate(repair_set_list):
            if obj in repair_set:
                M[obj_to_node_id_map[obj], repair_set_index] = 1
    return M


def get_halfspaces(
    num_nodes: int,
    num_repair_sets: int,
    node_capacity: float,
    M: numpy.array,
) -> scipy.spatial.HalfspaceIntersection:
    # log(DEBUG, "",
    #     num_nodes=num_nodes,
    #     num_repair_sets=num_repair_sets,
    #     node_capacity=node_capacity,
    #     M=M,
    # )

    halfspaces = numpy.zeros((num_nodes + num_repair_sets, num_repair_sets + 1))
    for r in range(num_nodes):
        halfspaces[r, -1] = -node_capacity

    halfspaces[: num_nodes, :-1] = M
    for r in range(num_nodes, num_nodes + num_repair_sets):
        halfspaces[r, r - num_nodes] = -1
    # log(INFO, "halfspaces= \n{}".format(halfspaces) )

    feasible_point = numpy.array([node_capacity / num_repair_sets] * num_repair_sets)
    return scipy.spatial.HalfspaceIntersection(halfspaces, feasible_point)
