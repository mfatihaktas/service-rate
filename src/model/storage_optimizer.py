import abc
import math

from typing import Tuple

from src.service_rate import storage_scheme as storage_scheme_module

from src.utils.debug import *
from src.utils.misc import *


def get_alpha_beta_gamma_for_trapezoidal_cap_region(
    demand_vector_list: list[list[float]],
) -> Tuple[int, int, int]:
    x_set = set()
    y_set = set()
    for demand_vector in demand_vector_list:
        check(len(demand_vector) == 2, "ONLY defined for k= 2!")

        x_set.add(demand_vector[0])
        y_set.add(demand_vector[0])

    alpha = max(x_set)
    beta = max(y_set)
    gamma = min(x_set) + beta

    return (alpha, beta, gamma)


class StorageOptimizer:
    @abc.abstractmethod
    def get_node_id_to_objs_list(self):
        pass


class StorageOptimizer_wReplicasAndXORs_forTwoObjectsAndTrapezoidalCapRegion:
    def __init__(self, alpha: float, beta: float, gamma: float):
        """Trapezoidal capacity region is defined by the points:
        (alpha, 0)
        (0, beta)
        (alpha, gamma - alpha)
        (gamma - beta, beta)
        where gamma > max{alpha, beta}.
        """

        # check(gamma > alpha and gamma > beta, "`gamma` must be greater `alpha` and `beta`!")

        if gamma < alpha:
            alpha = gamma
        if gamma < beta:
            beta = gamma

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def get_node_id_to_objs_list(self) -> list[storage_scheme_module.Obj]:
        node_id_to_objs_list = []

        num_a = math.ceil((self.alpha - self.beta + self.gamma) / 2)
        num_b = math.ceil((self.beta - self.alpha + self.gamma) / 2)
        num_nodes = math.ceil((self.alpha + self.beta + self.gamma) / 2)
        # num_a_plus_b = math.ceil((self.alpha + self.beta - self.gamma) / 2)
        num_a_plus_b = num_nodes - num_a - num_b

        for _ in range(num_a):
            node_id_to_objs_list.append(
                [
                    storage_scheme_module.PlainObj(id_str=f"{get_char(0)}")
                ]
            )

        for _ in range(num_b):
            node_id_to_objs_list.append(
                [
                    storage_scheme_module.PlainObj(id_str=f"{get_char(1)}")
                ]
            )

        for _ in range(num_a_plus_b):
            node_id_to_objs_list.append(
                [
                    storage_scheme_module.CodedObj(
                        coeff_obj_list=[
                            (1, storage_scheme_module.PlainObj(id_str=f"{get_char(0)}")),
                            (1, storage_scheme_module.PlainObj(id_str=f"{get_char(1)}")),
                        ]
                    )
                ],
            )

        return node_id_to_objs_list
