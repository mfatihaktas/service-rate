import itertools
import numpy

from src.service_rate import service_rate

from src.utils.debug import *


class ServiceRateInspectorForStorageWithStripeAndParity(service_rate.ServiceRateInspectorBase):
    def __init__(
        self,
        k: int,
        m: int,
        s: int,
        obj_id_to_node_id_set_map: dict[int, set[int]],
        C: int = 1,
    ):
        super().__init__(k=k, n=None, m=m, C=C)
        self.s = s
        self.obj_id_to_node_id_set_map = obj_id_to_node_id_set_map

        self.obj_id_to_repair_node_id_sets_map = self.get_obj_id_to_repair_node_id_sets_map()

        self.obj_id_to_num_repair_sets_map = {
            obj_id: len(repair_node_id_sets)
            for obj_id, repair_node_id_sets in self.obj_id_to_repair_node_id_sets_map.items()
        }
        self.l = sum(self.obj_id_to_num_repair_sets_map.values())

        self.T = self.get_T()
        self.M = self.get_M()

        log(DEBUG, "",
            obj_id_to_repair_node_id_sets_map=self.obj_id_to_repair_node_id_sets_map,
            obj_id_to_num_repair_sets_map=self.obj_id_to_num_repair_sets_map,
            l=self.l,
            T=self.T,
            M=self.M,
        )

    def get_obj_id_to_repair_node_id_sets_map(self) -> dict[int, list[set[int]]]:
        return {
            obj_id: [
                set(node_id_tuple)
                for node_id_tuple in list(itertools.combinations(node_id_set, r=self.s))
            ]
            for obj_id, node_id_set in self.obj_id_to_node_id_set_map.items()
        }

    def get_T(self) -> numpy.array:
        total_num_repair_sets = self.l
        T = numpy.zeros((self.k, total_num_repair_sets))
        i = 0
        for obj_id in range(self.k):
            j = i + self.obj_id_to_num_repair_sets_map[obj_id]
            T[obj_id, i:j] = 1
            i = j

        return T

    def get_M(self) -> numpy.array:
        M = numpy.zeros((self.m, self.l))

        repair_set_index = 0
        for repair_node_id_sets in self.obj_id_to_repair_node_id_sets_map.values():
            for repair_node_id_set in repair_node_id_sets:
                for node_id in repair_node_id_set:
                    M[node_id, repair_set_index] = 1 / self.s

                repair_set_index += 1

        return M
