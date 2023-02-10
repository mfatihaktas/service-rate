import cvxpy
import itertools
import math
import numpy

from typing import Tuple

from src.service_rate import storage_scheme as storage_scheme_module
from src.utils.debug import *


class StorageOptimizer:
    def __init__(self, demand_vector_list: list[list[float]], max_num_nodes_factor: int = 1):
        self.demand_vector_list = demand_vector_list

        self.k = len(self.demand_vector_list[0])
        self.obj_id_set_to_min_span_size_map = self.get_obj_id_subset_to_min_span_size_map()
        log(DEBUG, "", obj_id_set_to_min_span_size_map=self.obj_id_set_to_min_span_size_map)

        max_min_span_size = max(min_span_size for _, min_span_size in self.obj_id_set_to_min_span_size_map.items())
        self.max_num_nodes = max_min_span_size * max_num_nodes_factor

    def get_obj_id_subset_to_min_span_size_map(self) -> dict[Tuple[int], int]:
        obj_id_set_to_min_span_size_map = {}

        for subset_size in range(1, self.k + 1):
            for obj_id_subset in itertools.combinations(list(range(self.k)), subset_size):
                min_span_size = float("-Inf")

                for demand_vector in self.demand_vector_list:
                    min_span_size_ = math.ceil(sum(demand_vector[obj_id] for obj_id in obj_id_subset))
                    min_span_size = max(min_span_size, min_span_size_)

                obj_id_set_to_min_span_size_map[frozenset(obj_id_subset)] = min_span_size

        log(DEBUG, "Done",
            obj_id_set_to_min_span_size_map=obj_id_set_to_min_span_size_map,
        )

        return obj_id_set_to_min_span_size_map


class StorageOptimizerReplication(StorageOptimizer):
    def __init__(self, demand_vector_list: list[list[float]]):
        super().__init__(demand_vector_list=demand_vector_list)

    def get_obj_id_to_node_id_set_map(self) -> dict[int, set[int]]:
        k, n = self.k, self.max_num_nodes
        log(DEBUG, "Started", k=k, n=n)

        x = cvxpy.Variable(shape=(k, n), name="x", boolean=True)
        constraint_list = []

        # Span constraints
        for counter, (obj_id_set, min_span_size) in enumerate(self.obj_id_set_to_min_span_size_map.items()):
            log(DEBUG, f">> counter= {counter}", obj_id_set=obj_id_set, min_span_size=min_span_size)

            if len(obj_id_set) == 1:
                obj_id = next(iter(obj_id_set))
                constraint_list.append(cvxpy.sum(x[obj_id, :]) >= min_span_size)
                continue

            # x v y v z = [x' ^ y' ^ z']' = [(1 - x) ^ (1 - y) ^ (1 - z)]'
            # => |x v y v z| >= s is equivalent to |(1 - x) ^ (1 - y) ^ (1 - z)| <= k - s
            z = cvxpy.Variable(shape=(n, 1), name=f"z_{counter}", boolean=True)

            for i in obj_id_set:
                constraint_list.append(1 - cvxpy.reshape(x[i, :], shape=(n, 1)) >= z)

            num_objs = len(obj_id_set)
            r_i_complement_in_columns = cvxpy.vstack([1 - x[i, :] for i in obj_id_set]).T
            sum_x_i_complement = r_i_complement_in_columns @ numpy.ones((num_objs, 1))
            # log(DEBUG, "", r_i_complement_in_columns=r_i_complement_in_columns, sum_x_i_complement=sum_x_i_complement)
            constraint_list.append(sum_x_i_complement - len(obj_id_set) + 1 <= z)

            constraint_list.append(cvxpy.sum(z) <= n - min_span_size)

        C = numpy.array([[i + 1] for i in range(n)])
        # log(DEBUG, "", C=C, constraint_list=constraint_list)
        # obj = cvxpy.Minimize(cvxpy.sum(x @ C))

        # obj = cvxpy.Minimize(cvxpy.max(x @ C) + cvxpy.max(x))
        # obj = cvxpy.Minimize(cvxpy.sum_squares(cvxpy.hstack(cvxpy.sum(x[:, i]) for i in range(n))))
        # obj = cvxpy.Minimize(sum(cvxpy.sum(x[:, i]) for i in range(n)))
        obj = cvxpy.Minimize(cvxpy.sum(x))

        prob = cvxpy.Problem(obj, constraint_list)
        prob.solve(solver="SCIP")

        log(DEBUG, "", prob_value=prob.value, x=x.value)

        check(prob.status == cvxpy.OPTIMAL, "Solution to optimization problem is NOT optimal!", x=x.value)

        obj_id_to_node_id_set_map = {
            obj_id : set(
                node_id
                for node_id in range(n)
                if x.value[obj_id, node_id] == 1
            )
            for obj_id in range(k)
        }
        return obj_id_to_node_id_set_map


class StorageOptimizerReplicationAndMDS(StorageOptimizer):
    def __init__(
        self,
        demand_vector_list: list[list[float]],
        mds_node_capacity: float = 1,
    ):
        super().__init__(demand_vector_list=demand_vector_list)
        self.mds_node_capacity = mds_node_capacity

    def get_obj_id_to_node_id_set_map(self) -> dict[int, set[int]]:
        k, n = self.k, self.max_num_nodes
        log(DEBUG, "Started", k=k, n=n)

        x = cvxpy.Variable(shape=(k, n), name="x", boolean=True)
        num_mds_nodes = cvxpy.Variable(name="num_mds_nodes", integer=True)
        constraint_list = []

        # Span constraints
        for counter, (obj_id_set, min_span_size) in enumerate(self.obj_id_set_to_min_span_size_map.items()):
            log(DEBUG, f">> counter= {counter}", obj_id_set=obj_id_set, min_span_size=min_span_size)

            if len(obj_id_set) == 1:
                obj_id = next(iter(obj_id_set))
                constraint_list.append(
                    cvxpy.sum(x[obj_id, :]) + num_mds_nodes * self.mds_node_capacity / self.k >= min_span_size
                )
                continue

            # x v y v z = [x' ^ y' ^ z']' = [(1 - x) ^ (1 - y) ^ (1 - z)]'
            # => |x v y v z| >= s is equivalent to |(1 - x) ^ (1 - y) ^ (1 - z)| <= k - s
            z = cvxpy.Variable(shape=(n, 1), name=f"z_{counter}", boolean=True)

            for i in obj_id_set:
                constraint_list.append(1 - cvxpy.reshape(x[i, :], shape=(n, 1)) >= z)

            num_objs = len(obj_id_set)
            r_i_complement_in_columns = cvxpy.vstack([1 - x[i, :] for i in obj_id_set]).T
            sum_x_i_complement = r_i_complement_in_columns @ numpy.ones((num_objs, 1))
            # log(DEBUG, "", r_i_complement_in_columns=r_i_complement_in_columns, sum_x_i_complement=sum_x_i_complement)
            constraint_list.append(sum_x_i_complement - len(obj_id_set) + 1 <= z)

            constraint_list.append(cvxpy.sum(z) <= n - (min_span_size - num_mds_nodes * self.mds_node_capacity / self.k))

        constraint_list.append(num_mds_nodes >= 0)

        # C = numpy.array([[i + 1] for i in range(n)])
        # log(DEBUG, "", C=C, constraint_list=constraint_list)
        # obj = cvxpy.Minimize(cvxpy.sum(x @ C) + num_mds_nodes**2 / 2 + num_mds_nodes / 2)
        number_of_object_copies_in_replicated_storage = sum(cvxpy.sum(x[:, i]) for i in range(n))
        obj = cvxpy.Minimize(number_of_object_copies_in_replicated_storage + 0.7 * num_mds_nodes)
        # obj = cvxpy.Minimize(number_of_object_copies_in_replicated_storage + 0.7 * num_mds_nodes)

        prob = cvxpy.Problem(obj, constraint_list)
        prob.solve(solver="SCIP")

        log(DEBUG, "", prob_value=prob.value, x=x.value, num_mds_nodes=num_mds_nodes.value)

        check(prob.status == cvxpy.OPTIMAL, "Solution to optimization problem is NOT optimal!")

        obj_id_to_node_id_set_map = {
            obj_id : set(
                node_id
                for node_id in range(n)
                if abs(x.value[obj_id, node_id] - 1) < 0.01
            )
            for obj_id in range(k)
        }
        return obj_id_to_node_id_set_map


class StorageOptimizerReplicationAndMDS_wSingleObjPerNode(StorageOptimizer):
    def __init__(
        self,
        demand_vector_list: list[list[float]],
    ):
        super().__init__(demand_vector_list=demand_vector_list)

    def get_node_id_to_objs_list(self) -> list[storage_scheme_module.Obj]:
        num_sys_list, num_mds = self.get_num_sys_and_mds_nodes()
        log(DEBUG, "", num_sys_list=num_sys_list, num_mds=num_mds)

        # Add systematic nodes
        node_id_to_objs_list = []
        for obj_id, num_sys in enumerate(num_sys_list):
            for _ in range(num_sys):
                node_id_to_objs_list.append(
                    [
                        storage_scheme_module.PlainObj(id_str=chr(ord("a") + obj_id))
                    ]
                )

        # Add MDS nodes
        for i in range(num_mds):
            node_id_to_objs_list.append(
                [
                    storage_scheme_module.CodedObj(
                        coeff_obj_list=[
                            (i + obj_id + 1, storage_scheme_module.PlainObj(chr(ord("a") + obj_id)))
                            for obj_id in range(len(num_sys_list))
                        ]
                    )
                ]
            )

        return node_id_to_objs_list

    def get_num_sys_and_mds_nodes(self) -> Tuple[list[int], int]:
        k = self.k
        log(DEBUG, "Started", k=k)

        num_sys = cvxpy.Variable(shape=(k, 1), name="num_sys", integer=True)
        num_mds = cvxpy.Variable(name="num_mds", integer=True)
        constraint_list = []

        # Span constraints
        id_to_num_sys_mds_group_map = {}
        id_to_num_mds_group_map = {}
        for counter, (obj_id_set, min_span_size) in enumerate(self.obj_id_set_to_min_span_size_map.items()):
            log(DEBUG, f">> counter= {counter}", obj_id_set=obj_id_set, min_span_size=min_span_size)

            # Number of recovery groups for `obj_id_set`
            num_sys_mds_group = cvxpy.Variable(name=f"num_sys_mds_group_{counter}", integer=True)
            num_mds_group = cvxpy.Variable(name=f"num_mds_group_{counter}", integer=True)
            id_to_num_sys_mds_group_map[obj_id_set] = num_sys_mds_group
            id_to_num_mds_group_map[obj_id_set] = num_mds_group

            # num_sys_vars_in_columns = cvxpy.hstack([num_sys[i] for i in range(k) if i not in obj_id_set])
            # max_m = cvxpy.max(cvxpy.hstack([id_to_num_sys_mds_group_map[frozenset([i])] for i in obj_id_set]))
            if len(obj_id_set) == 1:
                constraint_list.extend(
                    [
                        (
                            sum(cvxpy.maximum(num_sys_mds_group - num_sys[i], 0)
                                for i in range(k)
                                if i not in obj_id_set
                            )
                            + len(obj_id_set) * num_sys_mds_group
                            <= num_mds
                        ),
                        sum(num_sys[i] for i in obj_id_set) + num_sys_mds_group + num_mds_group >= min_span_size,
                        num_mds_group <= (num_mds - len(obj_id_set) * num_sys_mds_group) / k,
                        num_sys_mds_group >= 0,
                        num_mds_group >= 0,
                    ]
                )

            else:
                # max_m = cvxpy.max(cvxpy.hstack([id_to_num_sys_mds_group_map[frozenset([i])] for i in obj_id_set]))
                ## Not taking recovery groups that live solely in MDS nodes into account.
                # sum_num_sys_mds_group = sum(id_to_num_sys_mds_group_map[frozenset([i])] for i in obj_id_set)
                # constraint_list.extend(
                #     [
                #         sum(cvxpy.maximum(num_sys_mds_group - num_sys[i] + sum_num_sys_mds_group, 0) for i in range(k) if i not in obj_id_set) + len(obj_id_set) * num_sys_mds_group <= num_mds,
                #         sum(num_sys[i] for i in obj_id_set) - sum_num_sys_mds_group + num_sys_mds_group + (num_mds - len(obj_id_set) * num_sys_mds_group) / k >= min_span_size,
                #         num_sys_mds_group >= 0,
                #     ]
                # )

                sum_num_sys_mds_group = sum(id_to_num_sys_mds_group_map[frozenset([i])] for i in obj_id_set)
                sum_num_mds_group = sum(id_to_num_mds_group_map[frozenset([i])] for i in obj_id_set)
                constraint_list.extend(
                    [
                        (
                            sum(cvxpy.maximum(num_sys_mds_group - num_sys[i] + sum_num_sys_mds_group, 0)
                                for i in range(k)
                                if i not in obj_id_set
                            )
                            + len(obj_id_set) * num_sys_mds_group
                            + k * sum_num_mds_group
                            <= num_mds
                        ),
                        # sum(num_sys[i] for i in obj_id_set) - sum_num_sys_mds_group / k + num_sys_mds_group / k >= min_span_size,
                        sum(num_sys[i] for i in obj_id_set) + num_sys_mds_group >= min_span_size,
                        num_mds_group <= (num_mds - len(obj_id_set) * num_sys_mds_group) / k,
                        num_sys_mds_group >= 0,
                        num_mds_group >= 0,
                    ]
                )

        constraint_list.append(num_sys >= 0)
        constraint_list.append(num_mds >= 0)

        obj = cvxpy.Minimize(cvxpy.sum(num_sys) + num_mds)

        prob = cvxpy.Problem(obj, constraint_list)
        prob.solve(solver="SCIP")

        log(DEBUG, "",
            prob_value=prob.value,
            num_sys=num_sys.value,
            num_mds=num_mds.value,
            id_to_num_sys_mds_group_and_num_mds_group__map={
                id_: {
                    "num_sys_mds_group": num_sys_mds_group.value,
                    "num_mds_group": id_to_num_mds_group_map[id_].value,
                }
                for id_, num_sys_mds_group in id_to_num_sys_mds_group_map.items()
            },
        )

        check(prob.status == cvxpy.OPTIMAL, "Solution to optimization problem is NOT optimal!")

        return [round(float(num_sys.value[i])) for i in range(k)], round(float(num_mds.value))


class StorageOptimizerReplicationAnd2XORs(StorageOptimizer):
    def __init__(self, demand_vector_list: list[list[float]], max_num_nodes_factor: int = None):
        super().__init__(demand_vector_list=demand_vector_list, max_num_nodes_factor=max_num_nodes_factor)

    def get_obj_id_to_node_selection_vector_matrix_and_xored_obj_id_set_to_node_selection_vector_map(
        self
    ) -> Tuple[numpy.ndarray, dict[set, numpy.ndarray]]:
        k, n = self.k, self.max_num_nodes
        log(DEBUG, "Started", k=k, n=n)

        def get_frozenset(i: int, j: int):
            return frozenset(sorted([i, j]))

        obj_id_to_node_selection_vector_matrix = cvxpy.Variable(shape=(k, n), name="obj_id_to_node_selection_vector_matrix", boolean=True)
        # Assumption: An XOR'ed copy a + b won't be placed in a node with a or b.
        xored_obj_id_set_to_node_selection_vector_map = {
            get_frozenset(i, j): cvxpy.Variable(n, name=f"x_{i},{j}", boolean=True)
            for i in range(k)
            for j in range(i + 1, k)
        }

        constraint_list = []

        def find_intersection(node_selection_vector_list: list[cvxpy.Variable]) -> cvxpy.Variable:
            node_selection_vector_list = [cvxpy.reshape(v, shape=(n, 1)) for v in node_selection_vector_list]

            z = cvxpy.Variable(shape=(n, 1), boolean=True)  # , name=f"z_obj_id_{obj_id}_other_obj_id_{other_obj_id}"

            for v in node_selection_vector_list:
                constraint_list.append(v >= z)

            num_vectors = len(node_selection_vector_list)
            v_in_columns = cvxpy.hstack(node_selection_vector_list)
            sum_v = cvxpy.reshape(cvxpy.sum(v_in_columns, axis=1), shape=(n, 1))
            constraint_list.append(sum_v - num_vectors + 1 <= z)

            return z

        def find_union(node_selection_vector_list: list[cvxpy.Variable]) -> cvxpy.Variable:
            # x v y v z = [x' ^ y' ^ z']' = [(1 - x) ^ (1 - y) ^ (1 - z)]'
            # => |x v y v z| >= s is equivalent to |(1 - x) ^ (1 - y) ^ (1 - z)| <= k - s
            node_selection_vector_list = [cvxpy.reshape(v, shape=(n, 1)) for v in node_selection_vector_list]

            z = cvxpy.Variable(shape=(n, 1), boolean=True)

            for v in node_selection_vector_list:
                constraint_list.append(1 - v >= z)

            num_vectors = len(node_selection_vector_list)
            v_complement_in_columns = cvxpy.hstack([1 - v for v in node_selection_vector_list])
            sum_v_complement = cvxpy.reshape(cvxpy.sum(v_complement_in_columns, axis=1), shape=(n, 1))
            constraint_list.append(sum_v_complement - num_vectors + 1 <= z)

            return 1 - z

        for xored_obj_id_set, xor_node_selection_vector in xored_obj_id_set_to_node_selection_vector_map.items():
            # a + b must NOT be placed in a node with a or b.
            replica_span = find_union([obj_id_to_node_selection_vector_matrix[i] for i in xored_obj_id_set])
            intersection_between_replica_span_and_xors = find_intersection([replica_span, xor_node_selection_vector])

            constraint_list.append(intersection_between_replica_span_and_xors == 0)

            # Number of a + b must be less than a or b replicas
            min_replica_count = cvxpy.min(cvxpy.hstack(cvxpy.sum(obj_id_to_node_selection_vector_matrix[i]) for i in xored_obj_id_set))
            constraint_list.append(cvxpy.sum(xor_node_selection_vector) <= min_replica_count)

        # Span constraints
        for counter, (obj_id_set, min_span_size) in enumerate(self.obj_id_set_to_min_span_size_map.items()):
            log(DEBUG, f">> counter= {counter}", obj_id_set=obj_id_set, min_span_size=min_span_size)

            if len(obj_id_set) == 1:
                obj_id = next(iter(obj_id_set))

                xor_span = find_union(
                    [
                        xored_obj_id_set_to_node_selection_vector_map[get_frozenset(obj_id, other_obj_id)]
                        for other_obj_id in range(k)
                        if other_obj_id != obj_id
                    ]
                )

                constraint_list.append(
                    cvxpy.sum(obj_id_to_node_selection_vector_matrix[obj_id]) + cvxpy.sum(xor_span) >= min_span_size
                )

                continue

            # len(obj_id_set) > 1
            replica_span = find_union([obj_id_to_node_selection_vector_matrix[i] for i in obj_id_set])

            num_xors_outside_replica_span_list = []
            for other_obj_id in set(range(k)) - obj_id_set:
                for obj_id in obj_id_set:
                    xor_w_other = xored_obj_id_set_to_node_selection_vector_map[get_frozenset(obj_id, other_obj_id)]

                    # TODO: What if XOR's with different other objects are on the same node?
                    intersection_between_replica_span_and_xor_w_other = find_intersection([replica_span, xor_w_other])
                    num_xors_outside_replica_span = cvxpy.sum(xor_w_other) - cvxpy.sum(intersection_between_replica_span_and_xor_w_other)
                    num_xors_outside_replica_span_list.append(num_xors_outside_replica_span)

            constraint_list.append(
                cvxpy.sum(replica_span)
                + sum(num_xors_outside_replica_span_list)
                >= min_span_size
            )

        obj = cvxpy.Minimize(
            cvxpy.sum(obj_id_to_node_selection_vector_matrix)
            + sum(cvxpy.sum(node_selection_vector) for node_selection_vector in xored_obj_id_set_to_node_selection_vector_map.values())
        )

        prob = cvxpy.Problem(obj, constraint_list)
        prob.solve(solver="SCIP")

        xored_obj_id_set_to_node_selection_vector_map = {
            obj_id: node_selection_vector.value
            for obj_id, node_selection_vector in xored_obj_id_set_to_node_selection_vector_map.items()
        }

        log(DEBUG, "",
            prob_value=prob.value,
            obj_id_to_node_selection_vector_matrix=obj_id_to_node_selection_vector_matrix.value,
            xored_obj_id_set_to_node_selection_vector_map=xored_obj_id_set_to_node_selection_vector_map
        )

        check(prob.status == cvxpy.OPTIMAL, "Solution to optimization problem is NOT optimal!",
              obj_id_to_node_selection_vector_matrix=obj_id_to_node_selection_vector_matrix.value
        )

        return obj_id_to_node_selection_vector_matrix.value, xored_obj_id_set_to_node_selection_vector_map

    def get_obj_id_to_node_id_set_map_and_xor_to_node_id_set_map(self) -> Tuple[dict[int, set[int]], dict[int, set[int]]]:
        obj_id_to_node_selection_vector_matrix, xored_obj_id_set_to_node_selection_vector_map = (
            self.get_obj_id_to_node_selection_vector_matrix_and_xored_obj_id_set_to_node_selection_vector_map()
        )

        k, n = self.k, self.max_num_nodes

        obj_id_to_node_id_set_map = {
            obj_id : set(
                node_id
                for node_id in range(n)
                if abs(obj_id_to_node_selection_vector_matrix[obj_id, node_id] - 1) < 0.05
            )
            for obj_id in range(k)
        }

        xor_to_node_id_set_map = {
            " + ".join([str(i) for i in xored_obj_id_set]): set(
                node_id
                for node_id in range(n)
                if abs(node_selection_vector[node_id] - 1) < 0.05
            )
            for xored_obj_id_set, node_selection_vector in xored_obj_id_set_to_node_selection_vector_map.items()
        }

        # Report stats
        node_id_set_list = list(obj_id_to_node_id_set_map.values()) + list(xor_to_node_id_set_map.values())
        node_id_set = set().union(*node_id_set_list)
        log(DEBUG, "",
            num_nodes=len(node_id_set),
            num_replicas=sum(len(node_id_set) for node_id_set in obj_id_to_node_id_set_map.values()),
            num_xors=sum(len(node_id_set) for node_id_set in xor_to_node_id_set_map.values()),
        )

        return obj_id_to_node_id_set_map, xor_to_node_id_set_map
