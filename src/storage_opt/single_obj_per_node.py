import collections
import cvxpy

from typing import Tuple

from src.service_rate import storage_scheme as storage_scheme_module
from src.storage_opt import (
    access_graph as access_graph_module,
    storage_optimizer as storage_optimizer_module,
)

from src.utils.debug import *


class StorageOptimizerReplicationAndMDS_wSingleObjPerNode(storage_optimizer_module.StorageOptimizer):
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


class StorageOptimizerReplicationAndXOR_wSingleObjPerNode(storage_optimizer_module.StorageOptimizer):
    def __init__(
        self,
        demand_vector_list: list[list[float]],
    ):
        super().__init__(demand_vector_list=demand_vector_list)

        self.access_graph = access_graph_module.AccessGraph(k=self.k)
        self.optimize()

    def optimize(self):
        k = self.k
        log(DEBUG, "Started", k=k)

        constraint_list = []

        # Span constraints
        obj_id_to_obj_to_num_touch_vars_map = {}

        for counter, (obj_id_set, min_span_size) in enumerate(self.obj_id_set_to_min_span_size_map.items()):
            log(DEBUG, f">> counter= {counter}", obj_id_set=obj_id_set, min_span_size=min_span_size)

            if len(obj_id_set) == 1:
                obj_id = next(iter(obj_id_set))

                num_touch_list = []
                obj_to_num_touch_vars_map = collections.defaultdict(list)
                for access_edge in self.access_graph.symbol_to_access_edges_map[obj_id]:
                    # log(DEBUG, f"access_edge= {access_edge}")

                    num_touch = cvxpy.Variable(integer=True)
                    num_touch_list.append(num_touch)

                    for touched_obj in access_edge.get_touched_objects():
                        # log(DEBUG, f"touched_obj= {touched_obj}")
                        obj_to_num_touch_vars_map[touched_obj].append(num_touch)

                constraint_list.append(
                    cvxpy.sum(num_touch_list) >= min_span_size
                )

                for obj, num_touch_to_obj_list in obj_to_num_touch_vars_map.items():
                    constraint_list.append(
                        cvxpy.sum(num_touch_to_obj_list) <= self.access_graph.obj_to_num_copies_var_map[obj]
                    )

                constraint_list.append(
                    cvxpy.vstack(num_touch_list) >= 0
                )
                # log(DEBUG, "", obj_to_num_touch_vars_map=obj_to_num_touch_vars_map)
                obj_id_to_obj_to_num_touch_vars_map[obj_id] = obj_to_num_touch_vars_map

                continue

            num_touch_list = []
            obj_to_num_touch_vars_map = collections.defaultdict(list)
            access_edge_list = []
            for obj_id in obj_id_set:
                access_edge_list.extend(self.access_graph.symbol_to_access_edges_map[obj_id])

            for access_edge in access_edge_list:
                num_touch = cvxpy.Variable(integer=True)
                num_touch_list.append(num_touch)

                for touched_obj in access_edge.get_touched_objects():
                    obj_to_num_touch_vars_map[touched_obj].append(num_touch)

            constraint_list.append(
                cvxpy.sum(num_touch_list) >= min_span_size
            )

            for obj, num_touch_to_obj_list in obj_to_num_touch_vars_map.items():
                constraint_list.append(
                    cvxpy.sum(num_touch_to_obj_list) <= self.access_graph.obj_to_num_copies_var_map[obj]
                )

        # All `num_copies_var`'s must be >= 0
        num_copies_var_hstack = cvxpy.hstack(list(self.access_graph.obj_to_num_copies_var_map.values()))
        constraint_list.extend(
            [
                num_copies_var_hstack >= 0,
                num_copies_var_hstack <= max(self.obj_id_set_to_min_span_size_map.values()),
            ]
        )

        # objective = cvxpy.Minimize(cvxpy.sum(list(self.access_graph.obj_to_num_copies_var_map.values())))

        # num_copies_var_list = list(self.access_graph.obj_to_num_copies_var_map.values())
        num_copies_var_list = []
        for obj, num_copies in self.access_graph.obj_to_num_copies_var_map.items():
            num_copies_var_list.append(
                (1 + 0.01 * obj.get_num_symbols()) * num_copies
            )

        num_copies_var_hstack = cvxpy.hstack(num_copies_var_list)
        objective = cvxpy.Minimize(
            # cvxpy.sum(num_copies_var_hstack)
            # Fails with: `SCIP: maximal branching depth level exceeded!`
            cvxpy.sum(num_copies_var_hstack) + cvxpy.max(num_copies_var_hstack)
            # cvxpy.sum(num_copies_var_hstack) + cvxpy.log_sum_exp(num_copies_var_hstack)
            # cvxpy.sum_squares(num_copies_var_hstack)
        )

        prob = cvxpy.Problem(objective, constraint_list)
        prob.solve(solver="SCIP")

        check(prob.status == cvxpy.OPTIMAL, "Solution to optimization problem is NOT optimal!")

        self.access_graph.set_obj_to_num_copies_map_after_optimization()

        log(DEBUG, "",
            prob_value=prob.value,
            obj_id_to_obj_to_num_touch_map={
                obj_id: {
                    obj: sum(var.value for var in num_touch_var_list)
                    for obj, num_touch_var_list in obj_to_num_touch_vars_map.items()
                }
                for obj_id, obj_to_num_touch_vars_map in obj_id_to_obj_to_num_touch_vars_map.items()
            }
        )
