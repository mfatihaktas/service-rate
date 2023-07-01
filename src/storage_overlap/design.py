import abc
import collections
import dataclasses
import enum
import itertools
import joblib
import math
import random

from typing import Generator

from src.service_rate import (
    demand_assigner,
    service_rate,
    service_rate_w_replica,
    storage_scheme as storage_scheme_module,
)
from src.utils import storage_object

from src.utils.debug import *


class StrategyToCheckIfDemandCovered(enum.Enum):
    cvxpy = "cvxpy"
    service_choice_union = "service_choice_union"
    demand_assigner = "demand_assigner"


@dataclasses.dataclass
class StorageDesign:
    k: int
    n: int

    strategy_to_check_if_demand_covered: StrategyToCheckIfDemandCovered
    obj_id_to_node_id_set_map: dict[int, set[int]] = dataclasses.field(init=False)

    def __post_init__(self):
        log(WARNING, "", strategy_to_check_if_demand_covered=self.strategy_to_check_if_demand_covered)
        if self.strategy_to_check_if_demand_covered == StrategyToCheckIfDemandCovered.cvxpy:
            self.service_rate_inspector = self.get_service_rate_inspector()

        elif self.strategy_to_check_if_demand_covered == StrategyToCheckIfDemandCovered.demand_assigner:
            self.demand_assigner = demand_assigner.DemandAssigner(
                obj_id_to_node_id_set_map=self.obj_id_to_node_id_set_map,
                demand_delta=0.01,
            )

    def frac_of_demand_vectors_covered(
        self,
        demand_vector_list: list[list[float]],
    ) -> float:
        num_covered = 0

        for demand_vector in demand_vector_list:
            if self.is_demand_vector_covered(demand_vector=demand_vector):
                num_covered += 1

        return num_covered / len(demand_vector_list)

    def frac_of_demand_vectors_covered_w_generator_input(
        self,
        demand_vector_generator: Generator[list[float], None, None],
    ) -> float:
        num_demand_vector = 0
        num_demand_vector_covered = 0
        for demand_vector in demand_vector_generator:
            # log(DEBUG, "", demand_vector=demand_vector)

            if self.is_demand_vector_covered(demand_vector=demand_vector):
                num_demand_vector_covered += 1

            num_demand_vector += 1

        return num_demand_vector_covered / num_demand_vector

    def _get_service_rate_inspector(self) -> service_rate.ServiceRateInspector:
        node_id_to_objs_list = [[] for _ in range(self.n)]
        for obj_id, node_id_set in self.obj_id_to_node_id_set_map.items():
            for node_id in node_id_set:
                node_id_to_objs_list[node_id].append(
                    storage_scheme_module.PlainObj(id_str=f"{obj_id}")
                )

        storage_scheme = storage_scheme_module.StorageScheme(node_id_to_objs_list=node_id_to_objs_list)
        log(DEBUG, "", storage_scheme=storage_scheme)

        return service_rate.ServiceRateInspector(
            m=len(node_id_to_objs_list),
            C=1,
            G=storage_scheme.obj_encoding_matrix,
            obj_id_to_node_id_map=storage_scheme.obj_id_to_node_id_map,
            redundancy_w_two_xors=False,
            max_repair_set_size=1,
        )

    def get_service_rate_inspector(self) -> service_rate_w_replica.ServiceRateInspectorForStorageWithReplicas:
        return service_rate_w_replica.ServiceRateInspectorForStorageWithReplicas(
            obj_id_to_node_id_set_map=self.obj_id_to_node_id_set_map,
        )

    def is_demand_vector_covered(
        self,
        demand_vector: list[float],
        maximal_load: float = 1,
    ) -> bool:
        # log(DEBUG, "Started", demand_vector=demand_vector)

        if self.strategy_to_check_if_demand_covered == StrategyToCheckIfDemandCovered.cvxpy:
            # log(DEBUG, "Will use service_rate_inspector.is_in_cap_region()")
            # load_across_nodes = self.service_rate_inspector.load_across_nodes(demand_vector)
            # log(DEBUG, "", load_across_nodes=load_across_nodes)
            return self.service_rate_inspector.is_in_cap_region(
                obj_demand_list=demand_vector, maximal_load=maximal_load
            )

        elif self.strategy_to_check_if_demand_covered == StrategyToCheckIfDemandCovered.demand_assigner:
            return self.demand_assigner.is_in_cap_region(
                obj_demand_list=demand_vector,
                maximal_load=maximal_load,
            )

        elif self.strategy_to_check_if_demand_covered == StrategyToCheckIfDemandCovered.service_choice_union:
            return self.is_demand_vector_covered_w_service_choice_union(
                demand_vector=demand_vector, maximal_load=maximal_load,
            )

    def is_demand_vector_covered_w_service_choice_union(
        self,
        demand_vector: list[float],
        maximal_load: float,
        max_combination_size: int = float("Inf"),
    ) -> bool:
        nonneg_demand_index_list = []
        for i, d in enumerate(demand_vector):
            if d > 0:
                nonneg_demand_index_list.append(i)

        max_combination_size_ = min(len(nonneg_demand_index_list), max_combination_size)
        for combination_size in range(1, max_combination_size_ + 1):
            if self.is_demand_vector_covered_for_given_combination_size(
                demand_vector=demand_vector,
                combination_size=combination_size,
                maximal_load=maximal_load,
                nonneg_demand_index_list=nonneg_demand_index_list,
            ) is False:
                log(WARNING, "Not covered",
                    storage_design=self,
                    # demand_vector=demand_vector,
                    combination_size=combination_size,
                    # nonneg_demand_index_list=nonneg_demand_index_list,
                )
                return False

        return True

    def is_demand_vector_covered_for_given_combination_size(
        self,
        demand_vector: list[float],
        combination_size: int,
        maximal_load: float,
        nonneg_demand_index_list: list[float] = None,
    ) -> bool:
        """Implements a "looser" version of is_demand_vector_covered().
        Returns True whenever is_demand_vector_covered() returns True.
        Might return True when is_demand_vector_covered() returns False.
        """

        if nonneg_demand_index_list is None:
            nonneg_demand_index_list = [i for i, d in enumerate(demand_vector) if d > 0]

        for index_combination in itertools.combinations(nonneg_demand_index_list, r=combination_size):
            cum_demand = 0
            node_id_set = set()
            for i in index_combination:
                node_id_set |= self.obj_id_to_node_id_set_map[i]
                cum_demand += demand_vector[i]

            if len(node_id_set) * maximal_load < math.ceil(cum_demand):
                log(WARNING, "Not covered",
                    # demand_vector=demand_vector,
                    cum_demand=cum_demand,
                    index_combination=index_combination,
                    node_id_set=node_id_set,
                )
                return False

        return True

    def is_demand_vector_covered_by_splitting_obj_demands_evenly_across_choices(
        self,
        demand_vector: list[float],
        maximal_load: float,
    ) -> bool:
        node_id_to_demand_map = collections.defaultdict(int)

        for obj_id, demand in enumerate(demand_vector):
            if demand == 0:
                continue

            node_id_set = self.obj_id_to_node_id_set_map[obj_id]

            demand_to_assign_to_each_node = demand / len(node_id_set)
            for node_id in node_id_set:
                node_id_to_demand_map[node_id] += demand_to_assign_to_each_node

                if node_id_to_demand_map[node_id] >= maximal_load:
                    return False

        return True

    def is_demand_vector_covered_by_assigning_to_leftmost_choice_first(
        self,
        demand_vector: list[float],
        maximal_load: float,
    ) -> bool:
        node_id_to_demand_map = collections.defaultdict(int)

        for obj_id, demand in enumerate(demand_vector):
            node_id_set = self.obj_id_to_node_id_set_map[obj_id]

            for node_id in node_id_set:
                avail_node_cap = maximal_load - node_id_to_demand_map[node_id]

                if avail_node_cap:
                    demand_to_assign = min(avail_node_cap, demand)

                    node_id_to_demand_map[node_id] += demand_to_assign
                    demand -= demand_to_assign

            if demand > 0:
                return False

        return True

    def is_demand_vector_covered_w_joblib(
        self,
        demand_vector: list[float],
    ) -> bool:
        nonneg_demand_index_list = [i for i, d in enumerate(demand_vector) if d > 0]

        is_demand_vector_covered_for_given_combination_size_list = joblib.Parallel(n_jobs=-1, prefer="threads")(
            joblib.delayed(self.is_demand_vector_covered_for_given_combination_size)(
                demand_vector=demand_vector,
                combination_size=combination_size,
                nonneg_demand_index_list=nonneg_demand_index_list,
            )
            for combination_size in range(1, len(nonneg_demand_index_list) + 1)
        )

        return all(is_demand_vector_covered_for_given_combination_size_list)

    def is_demand_vector_covered_alternative(
        self,
        demand_vector: list[float],
    ) -> bool:
        nonneg_demand_index_list = []
        for i, d in enumerate(demand_vector):
            if d > 0:
                nonneg_demand_index_list.append(i)

        for nonneg_demand_index_list_ in itertools.permutations(nonneg_demand_index_list):
            cum_demand = 0
            node_id_set = set()
            for i in nonneg_demand_index_list_:
                node_id_set |= self.obj_id_to_node_id_set_map[i]
                cum_demand += demand_vector[i]

                if len(node_id_set) < math.ceil(cum_demand):
                    # log(WARNING, "Not covered",
                    #     node_id_set=node_id_set,
                    #     cum_demand=cum_demand,
                    #     demand_vector=demand_vector,
                    #     nonneg_demand_index_list_=nonneg_demand_index_list_,
                    # )
                    return False

        # log(DEBUG, "Done")
        return True

    def get_node_overlap_size_to_counter_map(self) -> dict[int, int]:
        node_overlap_size_to_counter_map = collections.defaultdict(int)

        for obj_id_1 in range(self.k):
            # for obj_id_2 in range(obj_id_1 + 1, self.k):
            for obj_id_2 in range(self.k):
                if obj_id_2 == obj_id_1:
                    continue

                node_id_set_1 = self.obj_id_to_node_id_set_map[obj_id_1]
                node_id_set_2 = self.obj_id_to_node_id_set_map[obj_id_2]
                overlap_size = len(node_id_set_1.intersection(node_id_set_2))
                if overlap_size > 0:
                    node_overlap_size_to_counter_map[overlap_size] += 1

        return node_overlap_size_to_counter_map

    def get_span_size_to_count_map(
        self,
        combination_size: int,
    ) -> dict[int, int, int]:
        span_size_to_count_map = collections.defaultdict(int)

        for obj_id_tuple in itertools.combinations(list(range(self.k)), r=combination_size):
            node_id_set = set()
            for obj_id in obj_id_tuple:
                node_id_set |= self.obj_id_to_node_id_set_map[obj_id]

            span_size = len(node_id_set)
            span_size_to_count_map[span_size] += 1

        return span_size_to_count_map

    def get_span_size_to_count_map_w_monte_carlo(
        self,
        combination_size: int,
        num_samples: int,
    ) -> dict[int, int, int]:
        span_size_to_count_map = collections.defaultdict(int)

        _obj_id_list = list(range(self.k))

        for _ in range(num_samples):
            obj_id_list = random.sample(_obj_id_list, combination_size)

            node_id_set = set()
            for obj_id in obj_id_list:
                node_id_set |= self.obj_id_to_node_id_set_map[obj_id]

            span_size = len(node_id_set)
            span_size_to_count_map[span_size] += 1

        return span_size_to_count_map

    def get_span_size_to_freq_map(
        self,
        combination_size: int,
    ) -> dict[int, int, float]:
        span_size_to_count_map = self.get_span_size_to_count_map(
            combination_size=combination_size,
        )
        total_count = sum(span_size_to_count_map.values())

        return {
            span_size: count / total_count
            for span_size, count in span_size_to_count_map.items()
        }

    def get_span_size_to_freq_map_w_monte_carlo(
        self,
        combination_size: int,
        num_samples: int,
    ) -> dict[int, int, float]:
        span_size_to_count_map = self.get_span_size_to_count_map_w_monte_carlo(
            combination_size=combination_size,
            num_samples=num_samples,
        )
        total_count = sum(span_size_to_count_map.values())

        return {
            span_size: count / total_count
            for span_size, count in span_size_to_count_map.items()
        }


@dataclasses.dataclass(repr=False)
class NoRedundancyDesign(StorageDesign):
    def __post_init__(self):
        self.obj_id_to_node_id_set_map = {
            obj_id: {obj_id % self.n}
            for obj_id in range(self.k)
        }

        super().__post_init__()

    def __repr__(self):
        return (
            "NoRedundancyDesign( \n"
            f"\t k= {self.k} \n"
            f"\t n= {self.n} \n"
            ")"
        )

    def repr_for_plot(self):
        # return f"NoRedundancyDesign(k= {self.k}, n= {self.n})"
        return r"$\textrm{NoRedundancy}$"


@dataclasses.dataclass
class ReplicaDesign(StorageDesign):
    d: int


@dataclasses.dataclass(repr=False)
class ClusteringDesign(ReplicaDesign):
    def __post_init__(self):
        check(self.n % self.d == 0, f"d= {self.d} must divide n= {self.n}")

        self.obj_id_to_node_id_set_map = {}

        for obj_id in range(self.k):
            cluster_id = (obj_id // self.d) % self.n

            self.obj_id_to_node_id_set_map[obj_id] = set(
                range(cluster_id * self.d, (cluster_id + 1) * self.d)
            )

        super().__post_init__()

    def __repr__(self):
        return (
            "ClusteringDesign( \n"
            f"\t k= {self.k} \n"
            f"\t n= {self.n} \n"
            f"\t d= {self.d} \n"
            ")"
        )

    def repr_for_plot(self):
        # return f"Clustering(k= {self.k}, n= {self.n}, d= {self.d})"
        # return r"$\textrm{Clustering}$" + fr", $d= {self.d}$"
        return r"$\textrm{Clustering}$"


@dataclasses.dataclass
class CyclicDesign(ReplicaDesign):
    shift_size: int

    def __post_init__(self):
        # check(self.n % self.d == 0, f"d= {self.d} must divide n= {self.n}")

        self.obj_id_to_node_id_set_map = {
            obj_id: set(
                i % self.n
                for i in range(obj_id, obj_id + self.shift_size * self.d, self.shift_size)
            )
            for obj_id in range(self.k)
        }

        super().__post_init__()

    def __repr__(self):
        return (
            "CyclicDesign( \n"
            f"\t k= {self.k} \n"
            f"\t n= {self.n} \n"
            f"\t d= {self.d} \n"
            f"\t shift_size= {self.shift_size} \n"
            ")"
        )

    def repr_for_plot(self):
        # return f"Cyclic(k= {self.k}, n= {self.n}, d= {self.d}, s= {self.shift_size})"
        # return r"$\textrm{Cyclic}$" + fr", $d={self.d}$"
        return r"$\textrm{Cyclic}$"


@dataclasses.dataclass
class RandomBlockDesign(ReplicaDesign):
    def __post_init__(self):
        self.obj_id_to_node_id_set_map = collections.defaultdict(set)

        num_objs_per_node = (self.k * self.d) // self.n
        # log(DEBUG, f"num_objs_per_node= {num_objs_per_node}")
        node_id_to_obj_id_set_map = collections.defaultdict(set)
        # node_id_to_num_objs_map = {node_id: 0 for node_id in range(self.n)}

        obj_id_queue = collections.deque(
            [
                obj_id
                for _ in range(self.d)
                for obj_id in range(self.k)
            ]
        )

        for obj_id in obj_id_queue:
            # log(DEBUG, f"> obj_id= {obj_id}, rep_id= {rep_id}")

            node_id = random.randint(0, self.n - 1)
            # log(DEBUG, f"node_id= {node_id}")

            counter = 0
            while (
                len(node_id_to_obj_id_set_map[node_id]) == num_objs_per_node
                or node_id in self.obj_id_to_node_id_set_map[obj_id]
            ):
                node_id = (node_id + 1) % self.n
                # log(DEBUG, f"counter= {counter}",
                #     node_id=node_id,
                #     rep_id=rep_id,
                # )

                counter += 1
                if counter == self.n:
                    # check(False, f"Could not find node for obj_id= {obj_id}",
                    #       node_id_to_num_objs_map=node_id_to_num_objs_map,
                    #       obj_id_to_node_id_set_map=self.obj_id_to_node_id_set_map,
                    # )

                    for node_id in range(self.n):
                        if node_id in self.obj_id_to_node_id_set_map[obj_id]:
                            continue

                        obj_id_to_swap_set = node_id_to_obj_id_set_map[node_id]
                        obj_id_to_swap = next(iter(obj_id_to_swap_set))

                        self.obj_id_to_node_id_set_map[obj_id_to_swap].remove(node_id)
                        self.obj_id_to_node_id_set_map[obj_id].add(node_id)

                        obj_id_queue.append(obj_id_to_swap)

            self.obj_id_to_node_id_set_map[obj_id].add(node_id)
            node_id_to_obj_id_set_map[node_id].add(obj_id)

        super().__post_init__()

    def __repr__(self):
        return (
            "RandomBlockDesign( \n"
            f"\t k= {self.k} \n"
            f"\t n= {self.n} \n"
            f"\t d= {self.d} \n"
            ")"
        )

    def repr_for_plot(self):
        # return f"RandomBlockDesign(k= {self.k}, n= {self.n}, d= {self.d})"
        # return r"$\textrm{RandomBlockDesign}$"
        # return r"$\textrm{Block}$" + fr", $d= {self.d}$"
        return r"$\textrm{Block}$"


@dataclasses.dataclass
class RandomExpanderDesign(ReplicaDesign):
    def __post_init__(self):
        node_id_list = list(range(self.n))
        self.obj_id_to_node_id_set_map = {
            obj_id: set(random.sample(node_id_list, self.d))
            for obj_id in range(self.k)
        }

        # log(DEBUG, "Constructed", obj_id_to_node_id_set_map=self.obj_id_to_node_id_set_map)

        super().__post_init__()

    def __repr__(self):
        return (
            "RandomExpanderDesign( \n"
            f"\t k= {self.k} \n"
            f"\t n= {self.n} \n"
            f"\t d= {self.d} \n"
            ")"
        )

    def reset(self):
        node_id_list = list(range(self.n))
        self.obj_id_to_node_id_set_map = {
            obj_id: set(random.sample(node_id_list, self.d))
            for obj_id in range(self.k)
        }

        # log(DEBUG, "Reset `obj_id_to_node_id_set_map`")

    def repr_for_plot(self):
        # return f"RandomExpanderDesign(k= {self.k}, n= {self.n}, d= {self.d})"
        # return r"$\textrm{RandomDesign}$" + fr", $d={self.d}$"
        # return r"$\textrm{Random}$" + fr", $d={self.d}$"
        return r"$\textrm{Random}$"


@dataclasses.dataclass
class RandomExpanderDesign_wClusters(ReplicaDesign):
    num_clusters: int

    def __post_init__(self):
        self.obj_id_to_node_id_set_map = {}

        num_objs_per_cluster = self.k // self.num_clusters
        cluster_size = self.n // self.num_clusters
        for cluster_index in range(self.num_clusters):
            node_id_list = list(
                range(
                    cluster_index * cluster_size,
                    min(self.n, (cluster_index + 1) * cluster_size)
                )
            )

            for obj_id in range(
                cluster_index * num_objs_per_cluster,
                min(self.k, (cluster_index + 1) * num_objs_per_cluster)
            ):
                self.obj_id_to_node_id_set_map[obj_id] = set(random.sample(node_id_list, self.d))

        log(DEBUG, "Constructed", obj_id_to_node_id_set_map=self.obj_id_to_node_id_set_map)

    def __repr__(self):
        return (
            "RandomExpanderDesign_wClusters( \n"
            f"\t k= {self.k} \n"
            f"\t n= {self.n} \n"
            f"\t d= {self.d} \n"
            ")"
        )

    def repr_for_plot(self):
        # return f"RandomExpanderDesign_wClusters(k= {self.k}, n= {self.n}, d= {self.d})"
        return r"$\textrm{RandomExpanderDesign_wClusters}, N_{c}= $" + fr"{self.num_clusters}"


@dataclasses.dataclass
class TwoXORDesign(StorageDesign):
    d: int

    def repr_for_plot(self):
        # return f"TwoXORDesign(k= {self.k}, n= {self.n}, d= {self.d})"
        return r"$\textrm{2-XOR Design}$"

    def __post_init__(self):
        check(self.k == self.n, f"k= {self.k} and n= {self.n} must be equal")

        num_objs_per_node = self.d

        # Place the systematic copies
        self.node_id_to_obj_set_map = {
            node_id: {storage_object.SysObject(symbol=node_id)} for node_id in range(self.n)
        }
        obj_id_to_num_service_choice_map = {obj_id: 1 for obj_id in range(self.k)}
        obj_id_to_cannot_xor_with_set_map = {obj_id: {obj_id} for obj_id in range(self.k)}

        # Place the XOR copies
        node_id = 0
        for obj_id in range(self.k):
            if obj_id_to_num_service_choice_map[obj_id] == self.d:
                continue

            found_other_obj_id = False
            for other_obj_id in range(self.k):
                if obj_id_to_num_service_choice_map[other_obj_id] == self.d:
                    continue
                elif other_obj_id in obj_id_to_cannot_xor_with_set_map[obj_id]:
                    continue

                xor_obj = storage_object.XORedObject(symbols=(obj_id, other_obj_id))
                found_other_obj_id = True

                found_node_for_xor = False
                for _ in range(self.n):
                    node_id = (node_id + 1) % self.n
                    # log(DEBUG, "***", node_id=node_id)

                    obj_set = self.node_id_to_obj_set_map[node_id]
                    if len(obj_set) == num_objs_per_node:
                        continue

                    obj_id_on_node_set = set()
                    for obj in obj_set:
                        for obj_id_ in obj.get_symbols():
                            obj_id_on_node_set.add(obj_id_)
                    # log(DEBUG, "", node_id=node_id, obj_set=obj_set, obj_id_on_node_set=obj_id_on_node_set)

                    skip_node = False
                    for obj_id_ in obj_id_on_node_set:
                        if (
                            obj_id_ in obj_id_to_cannot_xor_with_set_map[obj_id]
                            or obj_id_ in obj_id_to_cannot_xor_with_set_map[other_obj_id]
                        ):
                            skip_node = True
                    if skip_node:
                        continue

                    # log(DEBUG, "Adding", node_id=node_id, xor_obj=xor_obj)
                    # self.node_id_to_obj_set_map[node_id].add(xor_obj)
                    obj_set.add(xor_obj)
                    obj_id_to_num_service_choice_map[obj_id] += 1
                    obj_id_to_num_service_choice_map[other_obj_id] += 1

                    for obj_id_ in obj_id_on_node_set | {obj_id, other_obj_id}:
                        obj_id_to_cannot_xor_with_set_map[obj_id].add(obj_id_)
                        obj_id_to_cannot_xor_with_set_map[other_obj_id].add(obj_id_)

                    found_node_for_xor = True
                    break

                # log(DEBUG, "", node_id=node_id)
                check(found_node_for_xor, f"Could not find node for xor_obj= {xor_obj}",
                      node_id_to_obj_set_map=self.node_id_to_obj_set_map
                )
                break

            check(found_other_obj_id, "Could not find `other_obj_id` for `obj_id`!",
                  obj_id=obj_id,
                  node_id_to_obj_set_map=self.node_id_to_obj_set_map,
                  obj_id_to_cannot_xor_with_set_map=obj_id_to_cannot_xor_with_set_map,
                  obj_id_to_num_service_choice_map=obj_id_to_num_service_choice_map,
            )

        log(DEBUG, "", node_id_to_obj_set_map=self.node_id_to_obj_set_map)

        self.service_rate_inspector = self.get_service_rate_inspector()

    def get_service_rate_inspector(self) -> service_rate.ServiceRateInspector:
        node_id_to_objs_list = [[] for _ in range(self.n)]
        for node_id, obj_set in self.node_id_to_obj_set_map.items():
            for obj in obj_set:
                if isinstance(obj, storage_object.SysObject):
                    obj_ = storage_scheme_module.PlainObj(id_str=f"{obj.symbol}")
                elif isinstance(obj, storage_object.XORedObject):
                    symbol_list = obj.get_symbols()
                    obj_ = storage_scheme_module.CodedObj(
                        coeff_obj_list=[
                            (1, storage_scheme_module.PlainObj(id_str=f"{symbol_list[0]}")),
                            (1, storage_scheme_module.PlainObj(id_str=f"{symbol_list[1]}")),
                        ]
                    )

                node_id_to_objs_list[node_id].append(obj_)

        storage_scheme = storage_scheme_module.StorageScheme(node_id_to_objs_list=node_id_to_objs_list)
        log(DEBUG, "", storage_scheme=storage_scheme)

        return service_rate.ServiceRateInspector(
            m=len(node_id_to_objs_list),
            C=1,
            G=storage_scheme.obj_encoding_matrix,
            obj_id_to_node_id_map=storage_scheme.obj_id_to_node_id_map,
            redundancy_w_two_xors=True,
        )

    def is_demand_vector_covered(
        self,
        demand_vector: list[float],
    ) -> bool:
        return self.service_rate_inspector.is_in_cap_region(demand_vector)
