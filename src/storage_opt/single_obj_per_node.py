import abc
import collections
import cvxpy
import dataclasses
import functools
import itertools
import operator

from typing import Tuple

from src.service_rate import storage_scheme as storage_scheme_module
from src.storage_opt import storage_optimizer as storage_optimizer_module
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


@dataclasses.dataclass
class Object:
    @abc.abstractmethod
    def get_num_symbols(self):
        pass

    @abc.abstractmethod
    def get_symbols(self):
        pass


@dataclasses.dataclass
class SysObject(Object):
    symbol: int

    # def __eq__(self, other_obj: Object):
    #     return isinstance(other_obj, SysObject) and self.symbol == other_obj.symbol

    def __cmp__(self, other_obj: Object):
        if self.symbol == other_obj.symbol:
            return 0
        elif self.symbol < other_obj.symbol:
            return -1
        else:
            return 1


    def __hash__(self):
        return hash(self.symbol)

    def get_symbols(self) -> list[int]:
        return [self.symbol]

    def get_xor(self) -> int:
        return self.symbol

    def get_num_symbols(self) -> int:
        return 1


@dataclasses.dataclass
class XORedObject(Object):
    symbols: tuple[int]

    # def __eq__(self, other_obj: Object):
    #     return (
    #         isinstance(other_obj, XORedObject)
    #         and self.symbols == other_obj.symbols
    #     )

    def __cmp__(self, other_obj: Object):
        if self.symbols == other_obj.symbols:
            return 0
        elif self.symbols < other_obj.symbols:
            return -1
        else:
            return 1

    def __hash__(self):
        return hash(self.symbols)

    def get_symbols(self) -> list[int]:
        return self.symbols

    def get_xor(self) -> int:
        return functools.reduce(operator.ixor, self.symbols)

    def get_num_symbols(self) -> int:
        return len(self.symbols)


def get_symbol_recovered_by_object_pair(obj_1: Object, obj_2: Object) -> int:
    if abs(obj_1.get_num_symbols() - obj_2.get_num_symbols()) != 1:
        return None

    if obj_1.get_num_symbols() < obj_2.get_num_symbols():
        obj_1, obj_2 = obj_2, obj_1

    recovered_symbol_set = set(obj_1.get_symbols()) - set(obj_2.get_symbols())
    if len(recovered_symbol_set) > 1:
        return None

    recovered_symbol = next(iter(recovered_symbol_set))
    return recovered_symbol


@dataclasses.dataclass
class AccessEdge:
    @abc.abstractmethod
    def get_touched_objects(self):
        pass


@dataclasses.dataclass
class AccessLoop:
    obj: Object

    def get_touched_objects(self) -> list[Object]:
        return [self.obj]

@dataclasses.dataclass
class RecoveryEdge:
    obj_1: Object
    obj_2: Object

    def get_touched_objects(self) -> list[Object]:
        return [self.obj_1, self.obj_2]


@dataclasses.dataclass
class AccessGraph:
    k: int

    obj_to_num_copies_var_map: dict[Object, cvxpy.Variable] = dataclasses.field(default=None)
    symbol_to_access_edges_map: dict[int, list[AccessEdge]] = dataclasses.field(default=None)

    def __post_init__(self):
        # Construct `obj_to_num_copies_var_map`
        self.obj_to_num_copies_var_map = {}
        # Systematic copies
        for s in range(self.k):
            obj = SysObject(symbol=s)
            self.obj_to_num_copies_var_map[obj] = cvxpy.Variable(integer=True)

        # XOR'ed copies
        for xor_size in range(2, self.k + 1):
            for symbol_combination in itertools.combinations(list(range(self.k)), xor_size):
                obj = XORedObject(symbols=symbol_combination)
                self.obj_to_num_copies_var_map[obj] = cvxpy.Variable(integer=True)

        # Construct `symbol_to_access_edges_map`
        self.symbol_to_access_edges_map = collections.defaultdict(list)
        for symbol in range(self.k):
            self.symbol_to_access_edges_map[symbol].append(AccessLoop(obj=SysObject(symbol=symbol)))

        for (obj_1, obj_2) in itertools.combinations(list(self.obj_to_num_copies_var_map.keys()), 2):
            recovered_symbol = get_symbol_recovered_by_object_pair(obj_1=obj_1, obj_2=obj_2)
            if recovered_symbol is None or not (0 <= recovered_symbol <= self.k):
                # log(DEBUG, "Not a recovery group", recovered_symbol=recovered_symbol, obj_1=obj_1, obj_2=obj_2)
                continue

            self.symbol_to_access_edges_map[recovered_symbol].append(RecoveryEdge(obj_1=obj_1, obj_2=obj_2))

        log(DEBUG, "Constructed",
            obj_to_num_copies_var_map=self.obj_to_num_copies_var_map,
            symbol_to_access_edges_map=self.symbol_to_access_edges_map,
        )


class StorageOptimizerReplicationAndXOR_wSingleObjPerNode(storage_optimizer_module.StorageOptimizer):
    def __init__(
        self,
        demand_vector_list: list[list[float]],
    ):
        super().__init__(demand_vector_list=demand_vector_list)

        self.access_graph = AccessGraph(k=self.k)

    def get_object_to_num_copies_map(self) -> dict[Object, int]:
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
        constraint_list.append(
            cvxpy.vstack(list(self.access_graph.obj_to_num_copies_var_map.values())) >= 0
        )

        # objective = cvxpy.Minimize(cvxpy.sum(list(self.access_graph.obj_to_num_copies_var_map.values())))

        num_copies_list = []
        for obj, num_copies in self.access_graph.obj_to_num_copies_var_map.items():
            num_copies_list.append(
                (1 + 0.01 * obj.get_num_symbols()) * num_copies
            )
        objective = cvxpy.Minimize(cvxpy.sum(num_copies_list))

        prob = cvxpy.Problem(objective, constraint_list)
        prob.solve(solver="SCIP")

        check(prob.status == cvxpy.OPTIMAL, "Solution to optimization problem is NOT optimal!")

        object_to_num_copies_map = {
            obj: round(float(num_copies_var.value))
            for obj, num_copies_var in self.access_graph.obj_to_num_copies_var_map.items()
        }
        log(DEBUG, "",
            prob_value=prob.value,
            obj_id_to_obj_to_num_touch_vars_map=obj_id_to_obj_to_num_touch_vars_map,
            obj_id_to_obj_to_num_touch_map={
                obj_id: {
                    obj: sum(var.value for var in num_touch_var_list)
                    for obj, num_touch_var_list in obj_to_num_touch_vars_map.items()
                }
                for obj_id, obj_to_num_touch_vars_map in obj_id_to_obj_to_num_touch_vars_map.items()
            }
        )

        return object_to_num_copies_map
