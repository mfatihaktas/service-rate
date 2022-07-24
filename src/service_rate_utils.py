import collections
import cvxpy
import itertools
import joblib
import math
import numpy
import scipy.spatial

from src.debug_utils import *


def get_T(
    num_orig_objects: int, # k
    total_num_repair_sets: int,
    orig_obj_to_repair_set_w_obj_ids_size_map: dict[int, int],
) -> numpy.array:
    T = numpy.zeros((num_orig_objects, total_num_repair_sets))
    i = 0
    for orig_obj_id in range(num_orig_objects):
        j = i + orig_obj_to_repair_set_w_obj_ids_size_map[orig_obj_id]
        T[orig_obj_id, i:j] = 1
        i = j
    # print(f"T= {self.T}")

    return T


def get_M(
    num_objects: int, # n
    num_nodes: int,
    repair_set_w_obj_ids_list: list[set],
    obj_id_to_node_id_map: dict[int, int],
) -> numpy.array:
    M = numpy.zeros((num_nodes, len(repair_set_w_obj_ids_list)))

    for obj_id in range(num_objects):
        for repair_set_index, repair_set in enumerate(repair_set_w_obj_ids_list):
            if obj_id in repair_set:
                M[obj_id_to_node_id_map[obj_id], repair_set_index] = 1
    return M


def get_halfspaces(
    num_nodes: int,
    total_num_repair_sets: int,
    node_capacity: float,
    M: numpy.array,
) -> scipy.spatial.HalfspaceIntersection:
    # log(DEBUG, "",
    #     num_nodes=num_nodes,
    #     total_num_repair_sets=total_num_repair_sets,
    #     node_capacity=node_capacity,
    #     M=M,
    # )

    halfspaces = numpy.zeros((num_nodes + total_num_repair_sets, total_num_repair_sets + 1))
    for r in range(num_nodes):
        halfspaces[r, -1] = -node_capacity

    halfspaces[: num_nodes, :-1] = M
    for r in range(num_nodes, num_nodes + total_num_repair_sets):
        halfspaces[r, r - num_nodes] = -1
    # log(INFO, "halfspaces= \n{}".format(halfspaces) )

    feasible_point = numpy.array([node_capacity / total_num_repair_sets] * total_num_repair_sets)
    return scipy.spatial.HalfspaceIntersection(halfspaces, feasible_point)


def get_orig_obj_id_to_repair_sets_w_node_ids_map(
    orig_obj_id_to_repair_sets_w_obj_ids_map: dict[int, list[set]],
    obj_id_to_node_id_map: dict[int, int],
):
    # log(DEBUG, "", orig_obj_id_to_repair_sets_w_obj_ids_map=orig_obj_id_to_repair_sets_w_obj_ids_map)

    # def node_repair_set_in_str(repair_set):
    #     s = ",".join([f"{obj_id_to_node_id_map[obj_id]}" for obj_id in repair_set])
    #     return f"({s})"

    def repair_set_w_node_ids(repair_set_w_obj_ids):
        return set([obj_id_to_node_id_map[obj_id] for obj_id in repair_set_w_obj_ids])

    obj_id_to_repair_sets_w_node_ids_map = {
        # obj: ", ".join([node_repair_set_in_str(rs) for rs in repair_sets])
        obj: [repair_set_w_node_ids(rs) for rs in repair_sets_w_obj_ids]
        for obj, repair_sets_w_obj_ids in orig_obj_id_to_repair_sets_w_obj_ids_map.items()
    }

    # log(DEBUG, "Repair groups in terms of node id's:", obj_id_to_repair_sets_w_node_ids_map=obj_id_to_repair_sets_w_node_ids_map)

    return obj_id_to_repair_sets_w_node_ids_map


def get_orig_obj_id_to_repair_sets_w_obj_ids_map(
    k: int,
    n: int,
    G: numpy.array,
    max_repair_set_size: int,
) -> dict[int, set]:
    orig_obj_id_to_repair_sets_w_obj_ids_map = {}

    for obj in range(k):
        repair_set_list = []
        y = numpy.array([0] * obj + [1] + [0] * (k - obj - 1)).reshape(
            (k, 1)
        )

        for repair_size in range(1, max_repair_set_size + 1):
            for subset in itertools.combinations(range(n), repair_size):
                subset = set(subset)

                ## Check if subset contains any previously registered smaller repair group
                skip_rg = False
                for rg in repair_set_list:
                    if rg.issubset(subset):
                        skip_rg = True
                        break
                if skip_rg:
                    continue

                l = [G[:, i] for i in subset]
                A = numpy.column_stack(l)

                x, residuals, _, _ = numpy.linalg.lstsq(A, y)
                residuals = y - numpy.dot(A, x)
                # log(INFO, "", A=A, y=y, x=x, residuals=residuals)
                if (
                    numpy.sum(numpy.absolute(residuals)) < 0.0001
                ):  # residuals.size > 0 and
                    repair_set_list.append(subset)
        orig_obj_id_to_repair_sets_w_obj_ids_map[obj] = repair_set_list

    return orig_obj_id_to_repair_sets_w_obj_ids_map


def get_orig_obj_id_to_repair_sets_w_obj_ids_map_w_joblib(
    k: int,
    n: int,
    G: numpy.array,
    max_repair_set_size=None,
) -> dict[int, list[set[int]]]:
    return {
        obj_id: get_repair_sets_for_obj_w_joblib(
            obj_id=obj_id,
            k=k,
            n=n,
            G=G,
            max_repair_set_size=max_repair_set_size,
        )
        for obj_id in range(k)
    }


def is_a_repair_set(
    obj_id: int,
    comb_set: set[int],
    k: int,
    G: numpy.array,
) -> bool:
    # log(INFO, "", comb_set=comb_set)

    y = numpy.array(
        [0] * obj_id + [1] + [0] * (k - obj_id - 1)
    ).reshape((k, 1))

    l = [G[:, i] for i in comb_set]
    A = numpy.column_stack(l)

    x, residuals, _, _ = numpy.linalg.lstsq(A, y)
    residuals = y - numpy.dot(A, x)

    # log(INFO, "", A=A, y=y) # , x=x, residuals=residuals)
    # print(f"residuals= {residuals}")

    return numpy.sum(numpy.absolute(residuals)) < 0.0001 # residuals.size > 0 and


def find_repair_sets_w_range(
    obj_id: int,
    comb_set_list: list[set[int]],
    index_begin: int,
    index_end: int,
    k: int,
    G: numpy.array,
    repair_set_list: list[set[int]],
):
    for comb_set in comb_set_list[index_begin : index_end + 1]:
        is_a_repair_set_ = is_a_repair_set(
            obj_id=obj_id,
            comb_set=comb_set,
            k=k,
            G=G,
        )
        # print(f"is_a_repair_set_= {is_a_repair_set_}")
        # log(INFO,
        #     f"obj_id= {obj_id} \n"
        #     f"comb_set= {comb_set} \n"
        #     f"is_a_repair_set_= {is_a_repair_set_} \n"
        # )

        # if is_a_repair_set(comb_set):
        if is_a_repair_set_:
            repair_set_list.append(comb_set)


def get_repair_sets_for_obj_w_joblib(
    obj_id: int,
    k: int,
    n: int,
    G: numpy.array,
    max_repair_set_size=None,
) -> list[set[int]]:
    # log(DEBUG, f"obj_id= {obj_id}",
    #     k=k,
    #     n=n,
    #     max_repair_set_size=max_repair_set_size
    # )

    if max_repair_set_size is None:
        max_repair_set_size = k
    log(DEBUG, "", max_repair_set_size=max_repair_set_size)

    # Generate all combinations
    comb_set_list = []
    for repair_size in range(1, max_repair_set_size + 1):
        for subset in itertools.combinations(range(n), repair_size):
            comb_set_list.append(set(subset))

    # Find repair sets with joblib
    num_sets = len(comb_set_list)
    num_jobs = min(10, math.ceil(num_sets / 10))
    num_sets_per_job = math.ceil(num_sets / num_jobs)
    log(DEBUG, f"num_jobs= {num_jobs}, num_sets_per_job= {num_sets_per_job}")

    repair_set_list = []
    joblib.Parallel(n_jobs=num_jobs, prefer="threads")(
        joblib.delayed(find_repair_sets_w_range)(
            obj_id=obj_id,
            comb_set_list=comb_set_list,
            index_begin=i * num_sets_per_job,
            index_end=(i + 1) * num_sets_per_job - 1,
            k=k,
            G=G,
            repair_set_list=repair_set_list,
        )
        for i in range(num_jobs)
    )

    # Remove repair sets which are supersets of a smaller repair set
    _repair_set_list = []
    for repair_set in sorted(repair_set_list, key=len):
        # log(DEBUG, f"repair_set= {repair_set}")

        add_as_new_repair_set = True
        for _repair_set in _repair_set_list:
            if _repair_set.issubset(repair_set):
                add_as_new_repair_set = False
                break

        if add_as_new_repair_set:
            _repair_set_list.append(repair_set)

    return _repair_set_list


# TODO: This finds only the repair sets with one XOR'ed copy and one replica.
# It does not find repair sets with two XOR'ed copies.
# E.g., for system [a, b, a+b, a+2b], the complete repair sets would be found as
# orig_obj_id_to_repair_sets_w_obj_ids_map= {
#     0: [{0}, {1, 2}, {1, 3}, {2, 3}],
#     1: [{1}, {0, 2}, {0, 3}, {2, 3}]
# }
# where the repair set consists of the obj id's.
# This function would return
# orig_obj_id_to_repair_sets_w_obj_ids_map= {
#     0: [{0}, {1, 2}, {1, 3}],
#     1: [{1}, {0, 2}, {0, 3}]
# }
def get_orig_obj_id_to_repair_sets_w_obj_ids_map_for_redundancy_w_two_xors(
    n: int,
    G: numpy.array,
) -> dict[int, set]:
    # log(DEBUG, "", G=G)

    # Fill up `orig_obj_id_to_obj_ids_map`
    orig_obj_id_to_obj_ids_map = collections.defaultdict(list)
    for i in range(n):
        nonzero_indices = list(G[:, i].nonzero()[0])

        if len(nonzero_indices) == 1:
            orig_obj_id = nonzero_indices[0]
            orig_obj_id_to_obj_ids_map[orig_obj_id].append(i)

    # Fill up `orig_obj_id_to_repair_sets_w_obj_ids_map`
    orig_obj_id_to_repair_sets_w_obj_ids_map = collections.defaultdict(list)
    for i in range(n):
        nonzero_indices = list(G[:, i].nonzero()[0])
        # log(DEBUG, "", nonzero_indices=nonzero_indices)

        if len(nonzero_indices) == 1:
            orig_obj_id = nonzero_indices[0]
            orig_obj_id_to_repair_sets_w_obj_ids_map[orig_obj_id].append({i})

        elif len(nonzero_indices) == 2:
            [orig_obj_id_1, orig_obj_id_2] = nonzero_indices

            for obj_id_2 in orig_obj_id_to_obj_ids_map[orig_obj_id_2]:
                orig_obj_id_to_repair_sets_w_obj_ids_map[orig_obj_id_1].append({obj_id_2, i})
            for obj_id_1 in orig_obj_id_to_obj_ids_map[orig_obj_id_1]:
                orig_obj_id_to_repair_sets_w_obj_ids_map[orig_obj_id_2].append({obj_id_1, i})

        else:
            raise ValueError(
                "Unexpected recovery group size; \n"
                f"\t len(nonzero_indices)= {len(nonzero_indices)}"
            )

    return orig_obj_id_to_repair_sets_w_obj_ids_map


def solve_prob(
    prob: cvxpy.Problem,
) -> float:
    """Solves the given cvxpy problem and returns the optimal value
    for the problem.
    """

    try:
        prob.solve()
    except cvxpy.SolverError:
        prob.solve(solver="SCS")

    # log(DEBUG, f"prob.status= {prob.status}")
    if prob.status != cvxpy.OPTIMAL:
        # log(WARNING, "Not optimal", prob_status=prob.status)
        return None

    return prob.value
