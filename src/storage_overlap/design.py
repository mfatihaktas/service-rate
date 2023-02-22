import collections
import dataclasses
import itertools
import math
import random

from typing import Generator

from src.model import demand
from src.service_rate import (
    service_rate,
    storage_scheme as storage_scheme_module,
)
from src.utils import (
    data_structures,
    storage_object,
)

from src.utils.debug import *


@dataclasses.dataclass
class StorageDesign:
    k: int
    n: int


@dataclasses.dataclass
class ReplicaDesign(StorageDesign):
    d: int
    use_cvxpy: bool

    obj_id_to_node_id_set_map: dict[int, set[int]] = dataclasses.field(init=False)

    def __post_init__(self):
        log(WARNING, "", use_cvxpy=self.use_cvxpy)

        if self.use_cvxpy:
            self.service_rate_inspector = self.get_service_rate_inspector()

    def get_service_rate_inspector(self) -> service_rate.ServiceRateInspector:
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

    def is_demand_vector_covered(
        self,
        demand_vector: list[float],
    ) -> bool:
        if self.use_cvxpy:
            # log(DEBUG, "Will use service_rate_inspector.is_in_cap_region()")
            return self.service_rate_inspector.is_in_cap_region(demand_vector)

        nonneg_demand_index_list = []
        for i, d in enumerate(demand_vector):
            if d > 0:
                nonneg_demand_index_list.append(i)

        for combination_size in range(1, len(nonneg_demand_index_list) + 1):
            for index_combination in itertools.combinations(nonneg_demand_index_list, r=combination_size):
                cum_demand = 0
                node_id_set = set()
                for i in index_combination:
                    node_id_set |= self.obj_id_to_node_id_set_map[i]
                    cum_demand += demand_vector[i]

                if math.ceil(cum_demand) > len(node_id_set):
                    return False

        # log(DEBUG, "Done")
        return True

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

    def sim_frac_of_demand_vectors_covered(
        self,
        num_popular_obj: int,
        cum_demand: float,
        zipf_tail_index: float,
        num_sample: int,
        num_sim_run: int = 1,
    ) -> list[float]:
        log(DEBUG, "Started",
            num_popular_obj=num_popular_obj,
            cum_demand=cum_demand,
            zipf_tail_index=zipf_tail_index,
            num_sample=num_sample,
            num_sim_run=num_sim_run,
        )

        frac_of_demand_vectors_covered_list = []

        for sim_id in range(num_sim_run):
            log(DEBUG, f"> sim_id= {sim_id}")

            num_covered = 0
            for demand_vector in demand.sample_demand_vectors_w_zipf_law(
                    num_obj=self.k,
                    num_popular_obj=num_popular_obj,
                    cum_demand=cum_demand,
                    zipf_tail_index=zipf_tail_index,
                    num_sample=num_sample,
            ):
                if self.is_demand_vector_covered(demand_vector=demand_vector):
                    num_covered += 1

            frac_of_demand_vectors_covered = num_covered / num_sample
            frac_of_demand_vectors_covered_list.append(frac_of_demand_vectors_covered)

        log(DEBUG, "Done")
        return frac_of_demand_vectors_covered_list


@dataclasses.dataclass(repr=False)
class ClusteringDesign(ReplicaDesign):
    def __post_init__(self):
        check(self.n % self.d == 0, f"d= {self.d} must divide n= {self.n}")

        self.obj_id_to_node_id_set_map = {
            obj_id: set(
                i % self.n for i in range(obj_id * self.d , obj_id * self.d + self.d)
            )
            for obj_id in range(self.k)
        }

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
        return r"$\textrm{Clustering}$"


@dataclasses.dataclass
class CyclicDesign(ReplicaDesign):
    shift_size: int

    def __post_init__(self):
        check(self.n % self.d == 0, f"d= {self.d} must divide n= {self.n}")

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
        return r"$\textrm{Cyclic}$" + fr"$(s={self.shift_size})$"


@dataclasses.dataclass
class RandomDesign(ReplicaDesign):
    def __post_init__(self):
        self.obj_id_to_node_id_set_map = collections.defaultdict(set)

        num_objs_per_node = (self.k * self.d) // self.n
        log(DEBUG, f"num_objs_per_node= {num_objs_per_node}")
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

    def __repr__(self):
        return (
            "RandomDesign( \n"
            f"\t k= {self.k} \n"
            f"\t n= {self.n} \n"
            f"\t d= {self.d} \n"
            ")"
        )

    def repr_for_plot(self):
        # return f"RandomDesign(k= {self.k}, n= {self.n}, d= {self.d})"
        return r"$\textrm{Random}$"


@dataclasses.dataclass
class TwoXORDesign(StorageDesign):
    d: int

    """
    def __post_init__(self):
        check(self.k == self.n, f"k= {self.k} and n= {self.n} must be equal")

        num_objs_per_node = self.d

        # Place the systematic copies
        node_id_to_obj_set_map = {
            node_id: {storage_object.SysObject(symbol=node_id)} for node_id in range(self.n)
        }
        obj_id_to_num_service_choice_map = {obj_id: 1 for obj_id in range(self.k)}
        disjoint_obj_id_sets_for_service_choice = data_structures.DisjointSets(n=self.k)
        disjoint_obj_id_sets_for_node = data_structures.DisjointSets(n=self.k)

        # Place the XOR copies
        node_id = 0
        for obj_id in range(self.k):
            # num_service_choices_for_obj_id = disjoint_obj_id_sets_for_service_choice.get_connected_component_size(obj_id)
            # if num_service_choices_for_obj_id == self.d:
            #     continue
            if obj_id_to_num_service_choice_map[obj_id] == self.d:
                continue

            found_other_obj_id = False
            for other_obj_id in range(self.k):
                # num_service_choices_for_other_obj_id = disjoint_obj_id_sets_for_service_choice.get_connected_component_size(other_obj_id)
                # if num_service_choices_for_other_obj_id == self.d:
                #     continue
                if obj_id_to_num_service_choice_map[other_obj_id] == self.d:
                    continue
                elif disjoint_obj_id_sets_for_service_choice.is_connected(obj_id, other_obj_id):
                    continue
                elif disjoint_obj_id_sets_for_node.is_connected(obj_id, other_obj_id):
                    continue

                xor_obj = storage_object.XORedObject(symbols=(obj_id, other_obj_id))
                found_other_obj_id = True

                found_node_for_xor = False
                for _ in range(self.n):
                    node_id = (node_id + 1) % self.n
                    log(DEBUG, "***", node_id=node_id)

                    obj_set = node_id_to_obj_set_map[node_id]
                    if len(obj_set) == num_objs_per_node:
                        continue

                    obj_id_on_node_set = set()
                    for obj in obj_set:
                        for obj_id_ in obj.get_symbols():
                            obj_id_on_node_set.add(obj_id_)
                    log(DEBUG, "", node_id=node_id, obj_set=obj_set, obj_id_on_node_set=obj_id_on_node_set)

                    if disjoint_obj_id_sets_for_service_choice.do_intersect(
                        x_set=set(xor_obj.get_symbols()),
                        y_set=obj_id_on_node_set,
                    ):
                        continue

                    log(DEBUG, "Adding", node_id=node_id, xor_obj=xor_obj)
                    # node_id_to_obj_set_map[node_id].add(xor_obj)
                    obj_set.add(xor_obj)
                    obj_id_to_num_service_choice_map[obj_id] += 1
                    obj_id_to_num_service_choice_map[other_obj_id] += 1

                    disjoint_obj_id_sets_for_service_choice.union(obj_id, other_obj_id)
                    for obj_id_on_node in obj_id_on_node_set:
                        disjoint_obj_id_sets_for_node.union(obj_id, obj_id_on_node)
                        disjoint_obj_id_sets_for_node.union(other_obj_id, obj_id_on_node)

                    found_node_for_xor = True
                    break

                log(DEBUG, "", node_id=node_id)
                check(found_node_for_xor, f"Could not find node for xor_obj= {xor_obj}",
                      node_id_to_obj_set_map=node_id_to_obj_set_map
                )
                break

            check(found_other_obj_id, "Could not find `other_obj_id` for `obj_id`!",
                  node_id_to_obj_set_map=node_id_to_obj_set_map,
            )

        log(DEBUG, "", node_id_to_obj_set_map=node_id_to_obj_set_map)
    """

    def __post_init__(self):
        check(self.k == self.n, f"k= {self.k} and n= {self.n} must be equal")

        num_objs_per_node = self.d

        # Place the systematic copies
        node_id_to_obj_set_map = {
            node_id: {storage_object.SysObject(symbol=node_id)} for node_id in range(self.n)
        }
        obj_id_to_num_service_choice_map = {obj_id: 1 for obj_id in range(self.k)}
        # obj_id_to_node_id_in_service_choice_set_map = {obj_id: set(obj_id) for obj_id in range(self.k)}
        obj_id_to_cannot_xor_with_set_map = {obj_id: {obj_id} for obj_id in range(self.k)}

        # disjoint_obj_id_sets = data_structures.DisjointSets(n=self.k)

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
                # elif disjoint_obj_id_sets.is_connected(obj_id, other_obj_id):
                #     continue

                xor_obj = storage_object.XORedObject(symbols=(obj_id, other_obj_id))
                found_other_obj_id = True

                found_node_for_xor = False
                for _ in range(self.n):
                    node_id = (node_id + 1) % self.n
                    log(DEBUG, "***", node_id=node_id)

                    obj_set = node_id_to_obj_set_map[node_id]
                    if len(obj_set) == num_objs_per_node:
                        continue

                    obj_id_on_node_set = set()
                    for obj in obj_set:
                        for obj_id_ in obj.get_symbols():
                            obj_id_on_node_set.add(obj_id_)
                    log(DEBUG, "", node_id=node_id, obj_set=obj_set, obj_id_on_node_set=obj_id_on_node_set)

                    # if disjoint_obj_id_sets_for_service_choice.do_intersect(
                    #     x_set=set(xor_obj.get_symbols()),
                    #     y_set=obj_id_on_node_set,
                    # ):
                    #     continue
                    skip_node = False
                    for obj_id_ in obj_id_on_node_set:
                        if (
                            obj_id_ in obj_id_to_cannot_xor_with_set_map[obj_id]
                            or obj_id_ in obj_id_to_cannot_xor_with_set_map[other_obj_id]
                        ):
                            skip_node = True
                    if skip_node:
                        continue

                    log(DEBUG, "Adding", node_id=node_id, xor_obj=xor_obj)
                    # node_id_to_obj_set_map[node_id].add(xor_obj)
                    obj_set.add(xor_obj)
                    obj_id_to_num_service_choice_map[obj_id] += 1
                    obj_id_to_num_service_choice_map[other_obj_id] += 1

                    # disjoint_obj_id_sets_for_service_choice.union(obj_id, other_obj_id)
                    # for obj_id_on_node in obj_id_on_node_set:
                    #     disjoint_obj_id_sets_for_node.union(obj_id, obj_id_on_node)
                    #     disjoint_obj_id_sets_for_node.union(other_obj_id, obj_id_on_node)
                    for obj_id_ in obj_id_on_node_set | {obj_id, other_obj_id}:
                        obj_id_to_cannot_xor_with_set_map[obj_id].add(obj_id_)
                        obj_id_to_cannot_xor_with_set_map[other_obj_id].add(obj_id_)

                    found_node_for_xor = True
                    break

                log(DEBUG, "", node_id=node_id)
                check(found_node_for_xor, f"Could not find node for xor_obj= {xor_obj}",
                      node_id_to_obj_set_map=node_id_to_obj_set_map
                )
                break

            check(found_other_obj_id, "Could not find `other_obj_id` for `obj_id`!",
                  obj_id=obj_id,
                  node_id_to_obj_set_map=node_id_to_obj_set_map,
                  obj_id_to_cannot_xor_with_set_map=obj_id_to_cannot_xor_with_set_map,
                  obj_id_to_num_service_choice_map=obj_id_to_num_service_choice_map,
            )

        log(DEBUG, "", node_id_to_obj_set_map=node_id_to_obj_set_map)
