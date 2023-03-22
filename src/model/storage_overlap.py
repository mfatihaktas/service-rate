"""Assumes the following demand model:
- Exactly `m` objects have non-negative demand, which we refer to as "active" objects.
- Each active object has the same demand.
"""

import abc
import dataclasses
import math

from src.allocation_w_complexes import model as allocation_w_complexes_model

from src.utils.debug import *


@dataclasses.dataclass
class StorageDesignModel:
    k: int
    n: int
    d: int

    @abc.abstractmethod
    def prob_span_is_larger_than_m_times_lambda(
        self,
        m: int,
        lambda_: float,
    ):
        pass

    @abc.abstractmethod
    def prob_serving_upper_bound(self, m: int, lambda_: int):
        pass

    @abc.abstractmethod
    def prob_serving_lower_bound(self, m: int, lambda_: int):
        pass


@dataclasses.dataclass
class RandomExpanderDesignModel(StorageDesignModel):
    def prob_span_is_larger_than_m_times_lambda(
        self,
        m: int,
        lambda_: float,
    ) -> float:
        min_span = math.ceil(m * lambda_)

        return sum(
            allocation_w_complexes_model.prob_num_nonempty_cells_eq_c(
                n=self.n, m=m, d=self.d, c=span
            )
            for span in range(min_span, n + 1)
        )

    def prob_serving(self, m: int, lambda_: int) -> float:
        return min(
            self.prob_span_is_larger_than_m_times_lambda(
                m=m_,
                lambda_=lambda_,
            )
            for m_ in range(1, m + 1)
        )

    def prob_serving_upper_bound(self, m: int, lambda_: int) -> float:
        return min(
            self.prob_span_is_larger_than_m_times_lambda(
                m=m_,
                lambda_=lambda_,
            )
            for m_ in range(1, m + 1)
        )

    def prob_serving_lower_bound(self, m: int, lambda_: int) -> float:
        return math.prod(
            [
                self.prob_span_is_larger_than_m_times_lambda(
                    m=m_,
                    lambda_=lambda_,
                )
                for m_ in range(1, m + 1)
            ]
        )
