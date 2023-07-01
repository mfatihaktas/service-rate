import dataclasses
import random

from src.utils.debug import *


@dataclasses.dataclass
class DemandAssigner:
    obj_id_to_node_id_set_map: dict[int, set[int]]
    demand_delta: float

    def __post_init__(self):
        self.node_id_set = set()
        for node_id_set_ in self.obj_id_to_node_id_set_map.values():
            self.node_id_set |= node_id_set_
        # log(DEBUG, "", node_id_set=self.node_id_set)

    def is_in_cap_region(
        self,
        obj_demand_list: list[float],
        maximal_load: float = 1,
    ) -> bool:
        node_id_to_assigned_demand_map = {
            node_id: 0
            for node_id in self.node_id_set
        }

        # obj_id_list = list(range(len(obj_demand_list)))
        obj_id_to_demand_map = {
            obj_id: demand
            for obj_id, demand in enumerate(obj_demand_list)
        }

        while len(obj_id_to_demand_map):
            # Select `obj_id`
            obj_id = random.choice(list(obj_id_to_demand_map.keys()))
            demand_to_assign = min(obj_id_to_demand_map[obj_id], self.demand_delta)
            obj_id_to_demand_map[obj_id] -= demand_to_assign

            if obj_id_to_demand_map[obj_id] == 0:
                obj_id_to_demand_map.pop(obj_id)

            # Assign `demand_to_assign`
            node_id_set = self.obj_id_to_node_id_set_map[obj_id]
            assigned_demand_and_node_id_list = sorted(
                [
                    (node_id_to_assigned_demand_map[node_id], node_id)
                    for node_id in node_id_set
                ]
            )
            assigned_demand, node_id = assigned_demand_and_node_id_list[0]
            if assigned_demand + demand_to_assign > maximal_load:
                return False

            node_id_to_assigned_demand_map[node_id] += demand_to_assign

        # log(DEBUG, "", node_id_to_assigned_demand_map=node_id_to_assigned_demand_map)
        return True
