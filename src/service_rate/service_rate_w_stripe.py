import itertools
import numpy

from src.service_rate import service_rate

from src.utils.debug import *


class ServiceRateInspectorForStorageWithStripeAndParity(service_rate.ServiceRateInspectorForStorageWithReplicasOrStripes):
    def __init__(
        self,
        obj_id_to_node_id_set_map: dict[int, set[int]],
        s: int,
        C: int = 1,
    ):
        self.s = s
        super().__init__(obj_id_to_node_id_set_map=obj_id_to_node_id_set_map, C=C)

    def get_obj_id_to_repair_node_id_sets_map(self) -> dict[int, list[set[int]]]:
        return {
            obj_id: [
                set(node_id_tuple)
                for node_id_tuple in list(itertools.combinations(node_id_set, r=self.s))
            ]
            for obj_id, node_id_set in self.obj_id_to_node_id_set_map.items()
        }

    def get_M(self) -> numpy.array:
        # log(WARNING, "", m=self.m, l=self.l)

        M = numpy.zeros((self.m, self.l))

        repair_set_index = 0
        # for repair_node_id_sets in self.obj_id_to_repair_node_id_sets_map.values():
        for obj_id, repair_node_id_sets in self.obj_id_to_repair_node_id_sets_map.items():
            for repair_node_id_set in repair_node_id_sets:
                for node_id in repair_node_id_set:
                    # log(WARNING, "", node_id=node_id, repair_set_index=repair_set_index)
                    M[node_id, repair_set_index] = 1 / self.s

                repair_set_index += 1

        return M
