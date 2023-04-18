import abc
import dataclasses
import math
import scipy.stats

from mpmath import mp

from src.allocation_w_complexes import model as allocation_w_complexes_model
from src.model import (
    generalized_birthday_problem,
    scan_stats,
)

from src.utils.debug import *


mp.dps = 100


@dataclasses.dataclass
class StorageDesignModel:
    k: int
    n: int


@dataclasses.dataclass
class NoRedundancyDesignModel(StorageDesignModel):
    b: int

    def prob_serving_helper(self, p: int, lambda_: float) -> float:
        num_active_objs_handled_by_node = math.floor(1 / lambda_)

        prob_sys_can_serve = 1
        for node_id in range(self.n):
            prob_node_can_serve = scipy.stats.binom.cdf(num_active_objs_handled_by_node, self.b, p)
            prob_sys_can_serve *= prob_node_can_serve

        return prob_sys_can_serve

    def prob_serving_downscaling_demand_per_obj(self, p: int, lambda_b_1: float) -> float:
        lambda_ = lambda_b_1 / self.b
        return self.prob_serving_helper(p=p, lambda_=lambda_)

    def prob_serving_downscaling_demand_per_obj_(self, p: int, lambda_b_1: float) -> float:
        p_ = p / math.sqrt(self.b)
        lambda_ = lambda_b_1 / math.sqrt(self.b)

        return self.prob_serving_helper(p=p_, lambda_=lambda_)

    def prob_serving_downscaling_p_per_obj(self, p: int, lambda_b_1: float) -> float:
        p_ = p / self.b
        return self.prob_serving_helper(p=p_, lambda_=lambda_b_1)


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
    b: int


@dataclasses.dataclass
class ClusteringDesignModelForBernoulliObjDemands(ClusteringDesignModel):
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

        return prob_single_cluster_is_stable ** num_clusters

    def prob_serving_downscaling_p_per_obj(self, p: int, lambda_: int) -> float:
        p_ = p / self.b

        num_clusters = self.n / self.d
        num_active_objs_handled_by_cluster = math.floor(self.d / lambda_)

        prob_single_cluster_is_stable = scipy.stats.binom.cdf(num_active_objs_handled_by_cluster, self.d * self.b, p_)

        return prob_single_cluster_is_stable**num_clusters

    def prob_serving_upper_bound(self, p: int, lambda_: int) -> float:
        check(self.b == 1, "", b=self.b)

        p_ = mp.mpf(f"{p}")
        lambda_ = mp.mpf(f"{lambda_}")

        d_ = mp.mpf(f"{self.d}")
        n_ = mp.mpf(f"{self.n}")

        def kl_divergence(a, p):
            log(DEBUG, "Started", a=a, p=p)
            # log(DEBUG, "", a_over_p=(a / p), one_minus_a_over_one_minus_p=((1 - a) / (1 - p)))

            # return a * math.log(a / p) + (1 - a) * math.log((1 - a) / (1 - p))

            # a_ = mp.mpf(f"{a}")
            # p_ = mp.mpf(f"{p}")
            # return a_ * mp.log(a_ / p_) + (1 - a_) * mp.log((1 - a_) / (1 - p_))

            try:
                return a * mp.log(a / p) + (1 - a) * mp.log((1 - a) / (1 - p))
            except ZeroDivisionError:
                return None

        num_clusters = n_ / d_
        num_active_objs_handled_by_cluster = mp.floor(d_ / lambda_)

        kl_divergence_ = kl_divergence(num_active_objs_handled_by_cluster / d_, p_)
        if kl_divergence_ is None:
            return None
        prob_single_cluster_is_stable = mp.exp(-d_ * kl_divergence_)

        return mp.power(prob_single_cluster_is_stable, num_clusters)


@dataclasses.dataclass
class ClusteringDesignModelForExpObjDemands(ClusteringDesignModel):
    def prob_serving(self, mean_obj_demand: float) -> float:
        num_clusters = self.n / self.d
        num_objs_in_cluster = self.d * self.b

        prob_single_cluster_is_stable = scipy.stats.erlang.cdf(self.d, a=num_objs_in_cluster, loc=0, scale=mean_obj_demand)

        # log(DEBUG, "",
        #     d=self.d,
        #     num_clusters=num_clusters,
        #     num_active_objs_handled_by_cluster=num_active_objs_handled_by_cluster,
        #     prob_single_cluster_is_stable=prob_single_cluster_is_stable
        # )

        return prob_single_cluster_is_stable ** num_clusters

    def prob_serving_w_downscaling_mean_obj_demand_w_b(self, mean_obj_demand_b_1: float) -> float:
        mean_obj_demand = mean_obj_demand_b_1 / self.b

        return self.prob_serving(mean_obj_demand=mean_obj_demand)

    def prob_serving_lower_bound_w_chernoff(self, mean_obj_demand: float) -> float:
        n, b, d = self.n, self.b, self.d
        num_clusters = n / d
        # num_objs_in_cluster = self.d * self.b

        mu = 1 / mean_obj_demand
        if mu <= b:  # Moment generating function of Exponential is undefined.
            return 0

        prob_single_cluster_is_stable = 1 - math.exp(-d * (mu - b + b * math.log(b / mu)))

        return prob_single_cluster_is_stable ** num_clusters


@dataclasses.dataclass
class ClusteringDesignModelForParetoObjDemands(ClusteringDesignModel):
    def prob_serving(self, min_value: float, tail_index: float) -> float:
        """DEPRECATED: Could not use Sympy to find the CDF of sums of iid random variables.
        """
        import sympy
        import sympy.stats
        # from sympy import stats as sympy_stats

        num_clusters = self.n / self.d
        num_objs_in_cluster = self.d * self.b

        X_list = []
        for i in range(num_objs_in_cluster):
            X = sympy.stats.Pareto(f"X_{i}", min_value, tail_index)
            X_list.append(X)

        # sympy.Sum(X_list[], (i, 1, num_objs_in_cluster))
        sum_X = sum(X_list)
        E_X = sympy.stats.E(sum_X)
        # cdf_X = sympy.stats.cdf(sum_X)
        # pdf_X = sympy.stats.density(sum_X)
        log(DEBUG, "", sum_X=sum_X, E_X=E_X,
            # pdf_X=pdf_X
        )

        # prob_single_cluster_is_stable = sympy.stats.cdf(sum_X)(self.d)
        prob_single_cluster_is_stable = sum_X.pspace.distribution.pdf(1)

        return prob_single_cluster_is_stable ** num_clusters


@dataclasses.dataclass
class ClusteringDesignModelForBallsIntoBinsDemand(ClusteringDesignModel):
    def prob_serving(self, m: int, lambda_: int) -> float:
        num_bins = self.n / self.d
        num_balls = m
        max_num_balls = math.floor(self.d / lambda_)

        prob_single_node_is_stable = scipy.stats.binom.cdf(max_num_balls, num_balls, 1 / num_bins)

        return prob_single_node_is_stable ** num_bins


@dataclasses.dataclass
class CyclicDesignModel(ReplicaDesignModel):
    def prob_serving_w_scan_stats_approx_for_given_k(
        self,
        k: int,
        p: int,
        lambda_: int,
        upper_bound=True,
    ) -> float:
        span = k + self.d - 1
        num_active_objs_handled_by_span = math.floor(span / lambda_)
        if not upper_bound:
            num_active_objs_handled_by_span -= 1

        # k = self.d
        r = num_active_objs_handled_by_span
        if r / k <= p:
            return 0

        return scan_stats.scan_stats_approx_1(n=self.k, p=p, k=k, r=r)
        # return scan_stats.scan_stats_approx_2(n=self.k, p=p, k=k, r=r)

    def prob_serving_w_scan_stats_approx(self, p: int, lambda_: int, upper_bound=True) -> float:
        """Demand can be served if maximum scan statistics for window of size d (M_d) is
        \leq d.

        Note: Uses the approximation presented at the end of page 69 of [1], which assumes
        p < r/k != 1.

        [Revise] This is why we use the downscaled version of the sufficiency condition
        as M \leq 0.9 * d.
        """

        # if upper_bound:
        #     span = 2 * self.d + 1
        # else:
        #     span = 2 * self.d

        # num_active_objs_handled_by_span = math.floor(span / lambda_)

        # span = 2 * self.d + 1
        return self.prob_serving_w_scan_stats_approx_for_given_k(
            k=self.d, p=p, lambda_=lambda_, upper_bound=upper_bound
        )

    def prob_serving_w_scan_stats_approx_improved(self, p: int, lambda_: int, upper_bound=True) -> float:
        prob_list = []
        for k in range(self.d, 3 * self.d):
        # for k in range(1, self.k - self.d):
            prob = self.prob_serving_w_scan_stats_approx_for_given_k(
                k=k, p=p, lambda_=lambda_, upper_bound=upper_bound
            )
            prob_list.append(prob)

        if upper_bound:
            return min(prob_list)
        else:
            return max(prob_list)
