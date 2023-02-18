import dataclasses
import itertools
import math

from typing import Generator

from src.utils.debug import *


@dataclasses.dataclass
class StorageDesign:
    k: int
    n: int


@dataclasses.dataclass
class ReplicaDesign(StorageDesign):
    d: int

    obj_id_to_node_id_set_map: dict[int, set[int]] = dataclasses.field(default=None)

    def is_demand_vector_covered(
        self,
        demand_vector: list[float],
    ) -> bool:
        k = len(demand_vector)
        for combination_size in range(1, k + 1):
            for index_combination in itertools.combinations(list(range(k)), r=combination_size):
                cum_demand = 0
                node_id_set = set()
                for i in index_combination:
                    node_id_set |= self.obj_id_to_node_id_set_map[i]
                    cum_demand += demand_vector[i]

                if math.ceil(cum_demand) > len(node_id_set):
                    return False

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
        num_popular_objs: int,
        cum_demand: float,
        zipf_tail_index: float,
        num_samples: int,
        num_sim_run: int = 1,
    ) -> list[float]:
        frac_of_demand_vectors_covered_list = []

        for _ in num_sim_run:
            num_covered = 0
            for demand_vector_list in demand.sample_demand_vectors_w_zipf_law(
                    num_objs=self.k,
                    num_popular_objs=num_popular_objs,
                    cum_demand=cum_demand,
                    zipf_tail_index=zipf_tail_index,
                    num_samples=num_samples,
            ):
                if self.is_demand_vector_covered(demand_vector=demand_vector):
                    num_covered += 1

                frac_of_demand_vectors_covered = num_covered / num_samples
                frac_of_demand_vectors_covered_list.append(frac_of_demand_vectors_covered)

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

    def __repr__(self):
        return (
            "ClusteringDesign( \n"
            f"\t k= {self.k} \n"
            f"\t n= {self.n} \n"
            f"\t d= {self.d} \n"
            ")"
        )

    def repr_for_plot(self):
        return f"Clustering(k= {self.k}, n= {self.n}, d= {self.d})"


@dataclasses.dataclass
class CyclicDesign(ReplicaDesign):
    def __post_init__(self):
        check(self.n % self.d == 0, f"d= {self.d} must divide n= {self.n}")

        self.obj_id_to_node_id_set_map = {
            obj_id: set(i % self.n for i in range(obj_id, obj_id + self.d))
            for obj_id in range(self.k)
        }

    def __repr__(self):
        return (
            "CyclicDesign( \n"
            f"\t k= {self.k} \n"
            f"\t n= {self.n} \n"
            f"\t d= {self.d} \n"
            ")"
        )

    def repr_for_plot(self):
        return f"Cyclic(k= {self.k}, n= {self.n}, d= {self.d})"
