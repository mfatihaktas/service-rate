import cvxpy
import numpy

from src.service_rate import service_rate_utils

from src.utils.debug import *
from src.utils.plot import *


class ServiceRateInspectorForStorageWithStripeAndParity:
    def __init__(
        self,
        k: int,
        n: int,
        s: int,
        obj_id_to_node_id_set_map: dict[int, set[int]],
        C: int = 1,
    ):
        self.k = k
        self.n = n
        self.s = s
        self.obj_id_to_node_id_set_map = obj_id_to_node_id_set_map
        self.C = C

        self.obj_id_to_num_repair_sets_map = {
            obj_id: len(node_id_set)
            for obj_id, node_id_set in obj_id_to_node_id_set_map.items()
        }
        self.l = sum(self.obj_id_to_num_repair_sets_map.values())

        self.obj_id_to_repair_node_id_sets_map = self.get_obj_id_to_repair_node_id_sets_map()

        self.T = self.get_T()
        self.M = self.get_M()

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
        M = numpy.zeros((self.n, self.l))

        repair_set_index = 0
        for repair_node_id_sets in self.obj_id_to_repair_node_id_sets_map.values():
            for repair_node_id_set in repair_node_id_sets:
                for node_id in repair_node_id_set:
                    M[node_id, repair_set_index] = 1

                repair_set_index += 1

        return M

    def is_in_cap_region(self, obj_demand_list: list[float]) -> bool:
        demand_vector = numpy.array(obj_demand_list).reshape((self.k, 1))

        x = cvxpy.Variable(shape=(self.l, 1), name="x")

        # obj = cvxpy.Maximize(numpy.ones((1, self.l))*x)
        obj = cvxpy.Maximize(0)
        constraints = [self.M @ x <= self.C, x >= 0, self.T @ x == demand_vector]

        prob = cvxpy.Problem(obj, constraints)
        opt_value = service_rate_utils.solve_prob(prob)

        return opt_value is not None
