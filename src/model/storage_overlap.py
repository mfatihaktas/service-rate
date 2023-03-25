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
        return allocation_w_complexes_model.prob_num_nonempty_cells_geq_c(
            n=self.n, m=m, d=self.d, c=min_span
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

    def prob_serving_upper_bound_(self, m: int, lambda_: int) -> float:
        return min(
            allocation_w_complexes_model.prob_span_of_every_t_complexes_geq_u_upper_bound(
                n=self.n, m=m, d=self.d, t=m_, u=math.ceil(m_ * lambda_)
            )
            for m_ in range(1, m + 1)
        )

    def wrong_prob_serving_lower_bound(self, m: int, lambda_: int) -> float:
        return math.prod(
            [
                self.prob_span_is_larger_than_m_times_lambda(
                    m=m_,
                    lambda_=lambda_,
                )
                for m_ in range(1, m + 1)
            ]
        )

    def _prob_serving_lower_bound(self, m: int, lambda_: int) -> float:
        # return allocation_w_complexes_model.prob_expand_span_by_e_with_each_complex(
        #     n=self.n, m=m, d=self.d, e=lambda_
        # )

        # return allocation_w_complexes_model.prob_expand_span_by_e_with_each_complex(
        #     n=self.n, m=m, d=self.d, e=1
        # )

        # return allocation_w_complexes_model.prob_expand_span_by_at_least_e_with_each_complex(
        #     n=self.n, m=m, d=self.d, e=lambda_ - 1
        # )

        return allocation_w_complexes_model.prob_expand_span_as_necessary_faster(
            n=self.n, m=m, d=self.d, lambda_=lambda_
        )

    def prob_serving_lower_bound(self, m: int, lambda_: int) -> float:
        # return max(
        #     allocation_w_complexes_model.prob_span_of_every_t_complexes_geq_u_lower_bound(
        #         n=self.n, m=m, d=self.d, t=m_, u=math.ceil(m_ * lambda_)
        #     )
        #     for m_ in range(1, m + 1)
        # )

        return math.prod(
            [
                allocation_w_complexes_model.prob_span_of_every_t_complexes_geq_u_alternative(
                    n=self.n, m=m, d=self.d, t=m_, u=math.ceil(m_ * lambda_)
                )
                for m_ in range(1, m + 1)
            ]
        )
