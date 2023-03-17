"""Assumes the following demand model:
- Exactly `m` objects have non-negative demand, which we refer to as "active" objects.
- Each active object has the same demand.
"""

import abc
import dataclasses
import math

from src.model import allocation_w_complexes

from src.utils.debug import *


@dataclasses.dataclass
class StorageDesignModel:
    k: int
    n: int
    d: int

    @abc.abstractmethod
    def prob_serving_upper_bound(self, m: int, lambda_: int):
        pass

    @abc.abstractmethod
    def prob_serving_lower_bound(self, m: int, lambda_: int):
        pass


@dataclasses.dataclass
class RandomExpanderDesignModel(StorageDesignModel):
    def prob_union_of_m_service_choices_is_larger_than_m_times_lambda(
        self,
        m: int,
        lambda_: float,
    ) -> float:
        service_choice_union_size = math.ceil(m * lambda_)
        return allocation_w_complexes.prob_num_cells_w_zero_particles_eq_c(
            n=self.n, m=m, d=self.d, c=service_choice_union_size
        )

    def prob_serving_upper_bound(self, m: int, lambda_: int) -> float:
        return min(
            self.prob_union_of_m_service_choices_is_larger_than_m_times_lambda(
                m=m_,
                lambda_=lambda_,
            )
            for m_ in range(1, m + 1)
        )

    def prob_serving_lower_bound(self, m: int, lambda_: int) -> float:
        return math.prod(
            [
                self.prob_union_of_m_service_choices_is_larger_than_m_times_lambda(
                    m=m_,
                    lambda_=lambda_,
                )
                for m_ in range(1, m + 1)
            ]
        )
