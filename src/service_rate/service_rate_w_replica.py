from src.service_rate import service_rate

from src.utils.debug import *


class ServiceRateInspectorForStorageWithReplicas(service_rate.ServiceRateInspectorForStorageWithReplicasOrStripes):
    def __init__(
        self,
        obj_id_to_node_id_set_map: dict[int, set[int]],
        C: int = 1,
    ):
        super().__init__(obj_id_to_node_id_set_map=obj_id_to_node_id_set_map, C=C)

    def get_obj_id_to_repair_node_id_sets_map(self) -> dict[int, list[set[int]]]:
        return {
            obj_id: [{node_id} for node_id in node_id_set]
            for obj_id, node_id_set in self.obj_id_to_node_id_set_map.items()
        }
