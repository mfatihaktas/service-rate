import abc
import collections
import copy
import itertools

from typing import Tuple

from src.service_rate import (
    service_rate,
    storage_scheme as storage_scheme_module,
)

from src.utils.debug import *
from src.utils.misc import *


class StorageSearcher:
    def __init__(self, demand_vector_list: list[list[float]]):
        self.demand_vector_list = demand_vector_list

        self.k = len(self.demand_vector_list[0])

    @abc.abstractmethod
    def search(self):
        pass


class SearchStorageWithReplicasAndTwoXORs(StorageSearcher):
    def __init__(self, demand_vector_list: list[list[float]]):
        super().__init__(demand_vector_list=demand_vector_list)

        self.obj_to_add_list = self.get_obj_to_add_list()
        log(DEBUG, "", obj_to_add_list=self.obj_to_add_list)

    def get_obj_to_add_list(self) -> list[storage_scheme_module.Obj]:
        obj_to_add_list = [
            storage_scheme_module.PlainObj(id_str=f"{get_char(obj_id)}")
            for obj_id in range(self.k)
        ]

        obj_id_list = list(range(self.k))
        for obj_id_pair in itertools.combinations(obj_id_list, r=2):
            obj_to_add_list.append(
                storage_scheme_module.CodedObj(
                    coeff_obj_list=[
                        (1, storage_scheme_module.PlainObj(id_str=f"{get_char(obj_id_pair[0])}")),
                        (1, storage_scheme_module.PlainObj(id_str=f"{get_char(obj_id_pair[1])}")),
                    ]
                )
            )

        return obj_to_add_list

    def get_are_all_demand_vectors_covered_and_min_distance_to_demand_vectors(
        self,
        node_id_to_objs_list: list[list[storage_scheme_module.Obj]],
    ) -> Tuple[bool, float]:
        storage_scheme = storage_scheme_module.StorageScheme(node_id_to_objs_list)
        log(DEBUG, "", storage_scheme=storage_scheme)

        service_rate_inspector = service_rate.ServiceRateInspector(
            m=len(node_id_to_objs_list),
            C=1,
            G=storage_scheme.obj_encoding_matrix,
            obj_id_to_node_id_map=storage_scheme.obj_id_to_node_id_map,
        )

        are_all_demand_vectors_covered = True
        min_distance = float("Inf")
        for demand_vector in self.demand_vector_list:
            in_cap_region_, min_distance_ = service_rate_inspector.get_in_cap_region_and_min_distance_to_boundary_w_cvxpy(
                obj_demand_list=demand_vector,
            )
            log(DEBUG, "",
                node_id_to_objs_list=node_id_to_objs_list,
                demand_vector=demand_vector,
                in_cap_region=in_cap_region_,
                min_distance=min_distance_,
            )

            min_distance = min(min_distance, min_distance_)
            are_all_demand_vectors_covered = (are_all_demand_vectors_covered and in_cap_region_)

        return are_all_demand_vectors_covered, min_distance

    def get_node_id_to_objs_list_w_brute_force(self) -> list[list[storage_scheme_module.Obj]]:
        node_id_to_objs_list = [
            [
                storage_scheme_module.PlainObj(id_str=f"{get_char(obj_id)}")
            ]
            for obj_id in range(self.k)
        ]

        q = collections.deque()

        def append_to_q(node_id_to_objs_list: list[list[storage_scheme_module.Obj]]):
            q.append(storage_scheme_module.copy_node_id_to_objs_list(node_id_to_objs_list=node_id_to_objs_list))

        append_to_q(node_id_to_objs_list)

        count = 0
        while q:
            node_id_to_objs_list = q.popleft()
            count += 1
            log(DEBUG, "Started iteration", count=count, node_id_to_objs_list=node_id_to_objs_list)

            # if len(node_id_to_objs_list) > 4:
            #     return

            are_all_demand_vectors_covered, min_distance = self.get_are_all_demand_vectors_covered_and_min_distance_to_demand_vectors(
                node_id_to_objs_list=node_id_to_objs_list,
            )
            log(DEBUG, "",
                are_all_demand_vectors_covered=are_all_demand_vectors_covered,
                min_distance=min_distance,
                node_id_to_objs_list=node_id_to_objs_list,
            )

            if are_all_demand_vectors_covered:
                log(DEBUG, "Found storage")
                return node_id_to_objs_list

            are_all_demand_vectors_covered = False
            for obj_to_add in self.obj_to_add_list:
                node_id_to_objs_list.append([obj_to_add])
                append_to_q(copy.copy(node_id_to_objs_list))
                node_id_to_objs_list.pop()

        log(DEBUG, "Done")
        return None

    def get_node_id_to_objs_list(self) -> list[list[storage_scheme_module.Obj]]:
        node_id_to_objs_list = [
            [
                storage_scheme_module.PlainObj(id_str=f"{get_char(obj_id)}")
            ]
            for obj_id in range(self.k)
        ]

        count = 0
        while True:
            count += 1
            log(DEBUG, "Started iteration", count=count, node_id_to_objs_list=node_id_to_objs_list)

            are_all_demand_vectors_covered = False
            min_distance = float("Inf")
            obj_to_add = None
            for obj_to_add_ in self.obj_to_add_list:
                node_id_to_objs_list.append([copy.copy(obj_to_add_)])

                are_all_demand_vectors_covered, min_distance_ = self.get_are_all_demand_vectors_covered_and_min_distance_to_demand_vectors(
                    node_id_to_objs_list=node_id_to_objs_list,
                )

                if are_all_demand_vectors_covered:
                    log(DEBUG, "Found storage")
                    return node_id_to_objs_list

                elif min_distance_ < min_distance:
                    min_distance = min_distance_
                    obj_to_add = obj_to_add_
                    log(DEBUG, "Updated",
                        min_distance=min_distance,
                        min_distance_=min_distance_,
                        obj_to_add=obj_to_add,
                        obj_to_add_=obj_to_add_,
                    )

                node_id_to_objs_list.pop()

            node_id_to_objs_list.append([copy.copy(obj_to_add)])

        log(DEBUG, "Failed to find storage")
        return None
