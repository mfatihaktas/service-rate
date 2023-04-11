import abc
import dataclasses
import math
import scipy.stats

from src.allocation_w_complexes import model as allocation_w_complexes_model

from src.utils.debug import *


@dataclasses.dataclass
class StorageDesignModel:
    k: int
    n: int


@dataclasses.dataclass
class NoRedundancyDesignModel(StorageDesignModel):
    b: int

    def prob_serving_downscaling_demand_per_obj(self, p: int, lambda_b_1: float) -> float:
        lambda_ = lambda_b_1 / self.b
        num_active_objs_handled_by_node = math.floor(1 / lambda_)

        prob_sys_can_serve = 1

        for node_id in range(self.n):
            prob_node_can_serve = scipy.stats.binom.cdf(num_active_objs_handled_by_node, self.b, p)
            prob_sys_can_serve *= prob_node_can_serve

        return prob_sys_can_serve

    def prob_serving_downscaling_p_per_obj(self, p: int, lambda_b_1: float) -> float:
        num_active_objs_handled_by_node = math.floor(1 / lambda_b_1)
        p_ = p / self.b

        prob_sys_can_serve = 1

        for node_id in range(self.n):
            prob_node_can_serve = scipy.stats.binom.cdf(num_active_objs_handled_by_node, self.b, p_)
            prob_sys_can_serve *= prob_node_can_serve

        return prob_sys_can_serve


@dataclasses.dataclass
class ReplicaDesignModel(StorageDesignModel):
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
class RandomExpanderDesignModel(ReplicaDesignModel):
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


@dataclasses.dataclass
class ClusteringDesignModel(ReplicaDesignModel):
    def prob_serving(self, p: int, lambda_: int) -> float:
        # log(DEBUG, "Started", p=p, lambda_=lambda_)

        num_clusters = self.n / self.d
        num_active_objs_handled_by_cluster = math.floor(self.d / lambda_)

        prob_single_cluster_is_stable = scipy.stats.binom.cdf(num_active_objs_handled_by_cluster, self.d, p)
        # log(DEBUG, "",
        #     d=self.d,
        #     num_clusters=num_clusters,
        #     num_active_objs_handled_by_cluster=num_active_objs_handled_by_cluster,
        #     prob_single_cluster_is_stable=prob_single_cluster_is_stable
        # )

        return prob_single_cluster_is_stable**num_clusters

    def prob_serving_for_balls_into_bins_upper_bound(self, m: int, lambda_: int) -> float:
        num_bins = self.n / self.d
        num_balls = m
        max_num_balls = math.floor(self.d / lambda_)

        prob_single_node_is_stable = scipy.stats.binom.cdf(max_num_balls, num_balls, 1 / num_bins)

        return prob_single_node_is_stable**num_bins


@dataclasses.dataclass
class CyclicDesignModel(ReplicaDesignModel):
    def prob_serving_lower_bound(self, p: int, lambda_: int) -> float:
        num_nodes_needed_for_active_obj = math.ceil(lambda_)
