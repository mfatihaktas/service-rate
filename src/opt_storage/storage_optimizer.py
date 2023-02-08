import cvxpy
import itertools
import math
import numpy

from typing import Tuple

from src.service_rate import storage_scheme as storage_scheme_module
from src.utils.debug import *


class StorageOptimizer:
    def __init__(self, demand_vector_list: list[list[float]], max_num_nodes: int = None):
        self.demand_vector_list = demand_vector_list
        self.max_num_nodes = max_num_nodes

        self.k = len(self.demand_vector_list[0])
        self.obj_id_set_to_min_span_size_map = self.get_obj_id_subset_to_min_span_size_map()
        log(DEBUG, "", obj_id_set_to_min_span_size_map=self.obj_id_set_to_min_span_size_map)

        max_min_span_size = max(min_span_size for _, min_span_size in self.obj_id_set_to_min_span_size_map.items())
        if self.max_num_nodes is None or self.max_num_nodes < max_min_span_size:
            log(INFO, f"Setting max_num_nodes to max_min_span_size= {max_min_span_size}")
            self.max_num_nodes = max_min_span_size

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
            x_i_complement_in_columns = cvxpy.vstack([1 - x[i, :] for i in obj_id_set]).T
            sum_x_i_complement = x_i_complement_in_columns @ numpy.ones((num_objs, 1))
            # log(DEBUG, "", x_i_complement_in_columns=x_i_complement_in_columns, sum_x_i_complement=sum_x_i_complement)
            constraint_list.append(sum_x_i_complement - len(obj_id_set) + 1 <= z)

            constraint_list.append(cvxpy.sum(z) <= n - min_span_size)

        C = numpy.array([[i + 1] for i in range(n)])
        # log(DEBUG, "", C=C, constraint_list=constraint_list)
        # obj = cvxpy.Minimize(cvxpy.sum(x @ C))

        # obj = cvxpy.Minimize(cvxpy.max(x @ C) + cvxpy.max(x))
        # obj = cvxpy.Minimize(cvxpy.sum_squares(cvxpy.hstack(cvxpy.sum(x[:, i]) for i in range(n))))
        obj = cvxpy.Minimize(sum(cvxpy.sum(x[:, i]) for i in range(n)))

        prob = cvxpy.Problem(obj, constraint_list)
        prob.solve(solver="SCIP")

        log(DEBUG, "", prob_value=prob.value, x_value=x.value)

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
            x_i_complement_in_columns = cvxpy.vstack([1 - x[i, :] for i in obj_id_set]).T
            sum_x_i_complement = x_i_complement_in_columns @ numpy.ones((num_objs, 1))
            # log(DEBUG, "", x_i_complement_in_columns=x_i_complement_in_columns, sum_x_i_complement=sum_x_i_complement)
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
                node_id_to_objs_list.append(storage_scheme_module.PlainObj(id_str=chr(ord("a") + obj_id)))

        # Add MDS nodes
        for i in range(num_mds):
            coeff_obj_list = [(i + 1, storage_scheme_module.PlainObj(id_str="a"))]
            for obj_id in range(1, len(num_sys_list)):
                coeff_obj_list.append((1, storage_scheme_module.PlainObj(chr(ord("a") + obj_id))))

            node_id_to_objs_list.append(storage_scheme_module.CodedObj(coeff_obj_list=coeff_obj_list))

        return node_id_to_objs_list

    def get_num_sys_and_mds_nodes(self) -> Tuple[list[int], int]:
        k = self.k
        log(DEBUG, "Started", k=k)

        num_sys = cvxpy.Variable(shape=(k, 1), name="num_sys", integer=True)
        num_mds = cvxpy.Variable(name="num_mds", integer=True)
        constraint_list = []

        # Span constraints
        for counter, (obj_id_set, min_span_size) in enumerate(self.obj_id_set_to_min_span_size_map.items()):
            log(DEBUG, f">> counter= {counter}", obj_id_set=obj_id_set, min_span_size=min_span_size)

            # Number of recovery groups for `obj_id_set`
            m = cvxpy.Variable(name=f"m_{counter}")

            # num_sys_vars_in_columns = cvxpy.hstack([num_sys[i] for i in range(k) if i not in obj_id_set])
            constraint_list.extend(
                [
                    sum(cvxpy.maximum(m - num_sys[i], 0) for i in range(k) if i not in obj_id_set) + len(obj_id_set) * m <= num_mds,
                    sum(num_sys[i] for i in obj_id_set) + m + (num_mds - len(obj_id_set) * m) / k >= min_span_size,
                ]
            )

        obj = cvxpy.Minimize(cvxpy.sum(num_sys) + num_mds)

        prob = cvxpy.Problem(obj, constraint_list)
        prob.solve(solver="SCIP")

        log(DEBUG, "", prob_value=prob.value, num_sys=num_sys.value, num_mds=num_mds.value)

        check(prob.status == cvxpy.OPTIMAL, "Solution to optimization problem is NOT optimal!")

        return [int(num_sys.value[i]) for i in range(k)], int(num_mds.value)
