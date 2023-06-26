import abc
import dataclasses
import joblib
import math
import numpy
import scipy.special
import scipy.stats

from mpmath import mp

from src.allocation_w_complexes import model as allocation_w_complexes_model
from src.scan_stats import model as scan_stats_model
from src.sim import random_variable
from src.storage_overlap import design, math_utils

from src.utils.debug import *


# mp.dps = 100


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
        maximal_load: float,
    ) -> float:
        min_span = math.ceil(m * lambda_ / maximal_load)

        return allocation_w_complexes_model.prob_num_nonempty_cells_geq_c_w_joblib(
            n=self.n, m=m, d=self.d, c=min_span
        )

    def prob_serving_upper_bound_for_given_m(
        self,
        m: int,
        lambda_: float,
        maximal_load: float,
    ) -> float:
        if m == 0:
            return 1

        return min(
            self.prob_span_is_larger_than_m_times_lambda(
                m=m_, lambda_=lambda_, maximal_load=maximal_load
            )
            for m_ in range(1, m + 1)
        )

    def prob_serving_upper_bound_for_given_m_w_joblib(
        self,
        m: int,
        lambda_: float,
        maximal_load: float,
    ) -> float:
        if m == 0:
            return 1

        prob_span_is_larger_than_m_times_lambda_list = joblib.Parallel(n_jobs=-1, prefer="processes")(
            joblib.delayed(self.prob_span_is_larger_than_m_times_lambda)(
                m=m_,
                lambda_=lambda_,
                maximal_load=maximal_load,
            )
            for m_ in range(1, m + 1)
        )

        return min(prob_span_is_larger_than_m_times_lambda_list)

    def prob_serving_upper_bound(
        self,
        p: float,
        lambda_: float,
        maximal_load: float,
    ) -> float:
        return sum(
            (
                scipy.stats.binom.pmf(m, self.k, p)
                * self.prob_serving_upper_bound_for_given_m(
                    m=m, lambda_=lambda_, maximal_load=maximal_load
                )
            )
            for m in range(self.k + 1)
        )

        # Em = int(self.k * p)
        # return self.prob_serving_upper_bound_for_given_m(
        #     m=Em, lambda_=lambda_, maximal_load=maximal_load
        # )

    def prob_serving_upper_bound_for_given_m_(self, m: int, lambda_: int) -> float:
        return min(
            allocation_w_complexes_model.prob_span_of_every_t_complexes_geq_u_upper_bound(
                n=self.n, m=m, d=self.d, t=m_, u=math.ceil(m_ * lambda_)
            )
            for m_ in range(1, m + 1)
        )

    def prob_serving_lower_bound_for_given_m(
        self,
        m: int,
        lambda_: float,
        maximal_load: float,
    ) -> float:
        if m == 0:
            return 1

        return math.prod(
            [
                allocation_w_complexes_model.prob_span_of_every_t_complexes_geq_u_alternative(
                    n=self.n, m=m, d=self.d, t=m_, u=math.ceil(m_ * lambda_ / maximal_load)
                )
                for m_ in range(1, m + 1)
            ]
        )

    def prob_serving_lower_bound_for_given_m_w_joblib(
        self,
        m: int,
        lambda_: float,
        maximal_load: float,
    ) -> float:
        if m == 0:
            return 1

        result_list = joblib.Parallel(n_jobs=-1, prefer="processes")(
            joblib.delayed(allocation_w_complexes_model.prob_span_of_every_t_complexes_geq_u_alternative_w_joblib)(
                n=self.n, m=m, d=self.d, t=m_, u=math.ceil(m_ * lambda_ / maximal_load)
            )
            for m_ in range(1, m + 1)
        )

        return math.prod(result_list)

    def prob_serving_lower_bound(
        self,
        p: float,
        lambda_: float,
        maximal_load: float,
    ) -> float:
        return sum(
            (
                scipy.stats.binom.pmf(m, self.k, p)
                * self.prob_serving_lower_bound_for_given_m(
                    m=m, lambda_=lambda_, maximal_load=maximal_load
                )
            )
            for m in range(self.k + 1)
        )

        # Em = int(self.k * p)
        # return self.prob_serving_lower_bound_for_given_m(
        #     m=Em, lambda_=lambda_, maximal_load=maximal_load
        # )


@dataclasses.dataclass
class ClusteringDesignModel(ReplicaDesignModel):
    b: int

    def prob_serving_lower_bound_w_hoeffding(
        self,
        mean_obj_demand: float,
        min_value: float,
        max_value: float,
        maximal_load: float,
    ) -> float:
        log(DEBUG, "Started",
            mean_obj_demand=mean_obj_demand,
            min_value=min_value,
            max_value=max_value,
            maximal_load=maximal_load,
        )
        check(self.b == 1, f"Defined for only b = 1 but b = {self.b}")

        n, d = self.n, self.d
        num_clusters = n / d

        prob_single_cluster_is_stable = (
            1 - math.exp(
                -d * 2 * (
                    (maximal_load - mean_obj_demand) ** 2
                    / (max_value - min_value) ** 2
                )
            )
        )

        return prob_single_cluster_is_stable ** num_clusters


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
    def prob_serving(
        self,
        mean_obj_demand: float,
        maximal_load: float = 1,
    ) -> float:
        num_clusters = self.n / self.d
        num_objs_in_cluster = self.d * self.b

        cluster_capacity = maximal_load * self.d
        prob_single_cluster_is_stable = scipy.stats.erlang.cdf(
            cluster_capacity, a=num_objs_in_cluster, loc=0, scale=mean_obj_demand
        )

        # log(DEBUG, "",
        #     d=self.d,
        #     num_clusters=num_clusters,
        #     num_active_objs_handled_by_cluster=num_active_objs_handled_by_cluster,
        #     prob_single_cluster_is_stable=prob_single_cluster_is_stable
        # )

        return prob_single_cluster_is_stable ** num_clusters

    def prob_serving_w_downscaling_mean_obj_demand_w_b(
        self,
        mean_obj_demand_b_1: float,
        maximal_load: float,
    ) -> float:
        mean_obj_demand = mean_obj_demand_b_1 / self.b

        return self.prob_serving(mean_obj_demand=mean_obj_demand, maximal_load=maximal_load)

    def prob_serving_lower_bound_w_chernoff(
        self,
        mean_obj_demand: float,
        maximal_load: float,
    ) -> float:
        n, b, d = self.n, self.b, self.d
        num_clusters = n / d
        # num_objs_in_cluster = self.d * self.b

        mu = 1 / mean_obj_demand
        if mu <= b:  # Moment generating function of Exponential is undefined.
            return 0

        prob_single_cluster_is_stable = (
            1 - math.exp(
                -d * (
                    mu * maximal_load - b + b * math.log(b / mu / maximal_load)
                )
            )
        )

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

    def prob_serving_sufficient_cond_w_scan_stats_approx_for_given_k(
        self,
        k: int,
        p: int,
        lambda_: float,
        maximal_load: float,
    ) -> float:
        span = maximal_load * (self.d)
        # span = k
        num_active_objs_handled_by_span = math.floor(span / lambda_)
        r = num_active_objs_handled_by_span

        n_ = self.k  # + k - 1
        return scan_stats_model.scan_stats_cdf_approx_by_naus(n=n_, m=k, p=p, k=r)

    def prob_serving_necessary_cond_w_scan_stats_approx_for_given_k(
        self,
        k: int,
        p: int,
        lambda_: float,
        maximal_load: float,
    ) -> float:
        span = maximal_load * (k + self.d - 1)
        num_active_objs_handled_by_span = math.floor(span / lambda_)

        # k = self.d
        r = num_active_objs_handled_by_span
        # if r / k <= p:
        #     return 0

        n_ = self.k + k - 1
        # return scan_stats_model.scan_stats_approx_1(n=self.k, p=p, k=k, r=r)
        # return scan_stats_model.scan_stats_approx_2(n=self.k, p=p, k=k, r=r)
        return scan_stats_model.scan_stats_cdf_approx_by_naus(n=n_, m=k, p=p, k=r)

    def prob_serving_sufficient_cond_w_asymptotic_scan_stats_approx_for_given_k(
        self,
        k: int,
        p: int,
        lambda_: float,
        maximal_load: float,
    ) -> float:
        span = maximal_load * self.d
        num_active_objs_handled_by_span = math.floor(span / lambda_)
        r = num_active_objs_handled_by_span

        return scan_stats_model.scan_stats_approx_2(n=self.k, p=p, k=k, r=r)

    def prob_serving_necessary_cond_w_asymptotic_scan_stats_approx_for_given_k(
        self,
        k: int,
        p: int,
        lambda_: float,
        maximal_load: float,
    ) -> float:
        span = maximal_load * (k + self.d - 1)
        num_active_objs_handled_by_span = math.floor(span / lambda_)

        # k = self.d
        r = num_active_objs_handled_by_span
        # if r / k <= p:
        #     return 0

        return scan_stats_model.scan_stats_approx_2(n=self.k, p=p, k=k, r=r)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  Upper bound  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    def prob_serving_upper_bound_w_scan_stats_approx(
        self,
        p: int,
        lambda_: float,
        maximal_load: float = 1,
        asymptotic=False,
    ) -> float:
        """Demand can be served if maximum scan statistics for window of size d (M_d) is
        \leq d.

        Note: Uses the approximation presented at the end of page 69 of [1], which assumes
        p < r/k != 1.

        [Revise] This is why we use the downscaled version of the sufficiency condition
        as M \leq 0.9 * d.
        """
        if asymptotic:
            return self.prob_serving_necessary_cond_w_asymptotic_scan_stats_approx_for_given_k(
                k=self.d, p=p, lambda_=lambda_, maximal_load=maximal_load
            )

        else:
            return self.prob_serving_necessary_cond_w_scan_stats_approx_for_given_k(
                k=self.d, p=p, lambda_=lambda_, maximal_load=maximal_load
            )

    def prob_serving_upper_bound_w_scan_stats_approx_improved(
        self,
        p: int,
        lambda_: float,
        maximal_load: float,
        asymptotic=False,
    ) -> float:
        prob_list = []

        for k in range(self.d, 3 * self.d):
        # for k in range(1, self.k - self.d):
            if asymptotic:
                prob = self.prob_serving_necessary_cond_w_asymptotic_scan_stats_approx_for_given_k(
                    k=k, p=p, lambda_=lambda_, maximal_load=maximal_load
                )

            else:
                prob = self.prob_serving_necessary_cond_w_scan_stats_approx_for_given_k(
                    k=k, p=p, lambda_=lambda_, maximal_load=maximal_load
                )

            if prob is None:
                prob = float("Inf")

            prob_list.append(prob)

        return min(prob_list)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~  Lower bound  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    def prob_serving_lower_bound_w_scan_stats_approx(
        self,
        p: int,
        lambda_: float,
        maximal_load: float,
        asymptotic=False,
    ) -> float:
        if asymptotic:
            return self.prob_serving_sufficient_cond_w_asymptotic_scan_stats_approx_for_given_k(
                k=self.d, p=p, lambda_=lambda_, maximal_load=maximal_load
            )

        else:
            return self.prob_serving_sufficient_cond_w_scan_stats_approx_for_given_k(
                k=self.d, p=p, lambda_=lambda_, maximal_load=maximal_load
            )

    def prob_serving_lower_bound_w_scan_stats_approx_improved(
        self,
        p: int,
        lambda_: float,
        maximal_load: float,
        asymptotic=False,
    ) -> float:
        prob_list = []

        for k in range(self.d, 3 * self.d):
        # for k in range(self.d, self.k - self.d):
            if asymptotic:
                prob = self.prob_serving_sufficient_cond_w_asymptotic_scan_stats_approx_for_given_k(
                    k=k, p=p, lambda_=lambda_, maximal_load=maximal_load
                )

            else:
                prob = self.prob_serving_sufficient_cond_w_scan_stats_approx_for_given_k(
                    k=k, p=p, lambda_=lambda_, maximal_load=maximal_load
                )

            if prob is None:
                prob = float("-Inf")

            prob_list.append(prob)

        return max(prob_list)


@dataclasses.dataclass
class StorageDesignModelForGivenDemandDistribution(ReplicaDesignModel):
    def prob_cum_demand_leq_cum_supply_for_constant_demand(
        self,
        combination_size: int,
        demand_rv: random_variable.Constant,
        cum_supply: int,
    ) -> float:
        cum_demand = combination_size * demand_rv.value
        return int(demand_rv.value <= self.d and cum_demand <= cum_supply)

    def prob_cum_demand_leq_cum_supply_for_exp_demand(
        self,
        combination_size: int,
        demand_rv: random_variable.Exponential,
        cum_supply: int,
    ) -> float:
        return scipy.stats.erlang.cdf(
            cum_supply, a=combination_size, loc=0, scale=demand_rv.mean()
        )

    def prob_cum_demand_leq_cum_supply_for_bernoulli_demand(
        self,
        combination_size: int,
        demand_rv: random_variable.Bernoulli,
        cum_supply: int,
    ) -> float:
        num_active_objs_rv = scipy.stats.binom(combination_size, demand_rv.p)

        if demand_rv.D > self.d:
            return num_active_objs_rv.pmf(0)

        return sum(
            num_active_objs_rv.pmf(num_active_objs)
            for num_active_objs in range(combination_size + 1)
            if demand_rv.D * num_active_objs <= cum_supply
        )

    def prob_cum_demand_leq_cum_supply_w_sim(
        self,
        combination_size: int,
        demand_rv: random_variable.RandomVariable,
        cum_supply: int,
        num_sim_run: int,
    ) -> float:
        num_success = 0
        for _ in range(num_sim_run):
            demand_list = [
                demand_rv.sample() for _ in range(combination_size)
            ]

            if max(demand_list) > self.d:
                continue

            if sum(demand_list) > cum_supply:
                continue

            num_success += 1

        return num_success / num_sim_run

    def prob_cum_demand_leq_cum_supply(
        self,
        combination_size: int,
        demand_rv: random_variable.RandomVariable,
        span_size: int,
        maximal_load: float,
    ) -> float:
        # log(DEBUG, "Started",
        #     combination_size=combination_size,
        #     demand_rv=demand_rv,
        #     span_size=span_size,
        #     maximal_load=maximal_load,
        # )

        cum_supply = span_size * maximal_load

        if False and isinstance(demand_rv, random_variable.Constant):
            return self.prob_cum_demand_leq_cum_supply_for_constant_demand(
                combination_size=combination_size,
                demand_rv=demand_rv,
                cum_supply=cum_supply,
            )

        elif False and isinstance(demand_rv, random_variable.Exponential):
            return self.prob_cum_demand_leq_cum_supply_for_exp_demand(
                combination_size=combination_size,
                demand_rv=demand_rv,
                cum_supply=cum_supply,
            )

        elif isinstance(demand_rv, random_variable.Bernoulli):
            return self.prob_cum_demand_leq_cum_supply_for_bernoulli_demand(
                combination_size=combination_size,
                demand_rv=demand_rv,
                cum_supply=cum_supply,
            )

        else:
            log(WARNING, "Using numeric integral")
            # return math_utils.prob_cum_demand_leq_cum_supply_w_scipy(
            #     num_demands=combination_size,
            #     demand_pdf=demand_rv.pdf,
            #     d=self.d,
            #     span_size=span_size,
            #     maximal_load=maximal_load,
            # )

            return self.prob_cum_demand_leq_cum_supply_w_sim(
                combination_size=combination_size,
                demand_rv=demand_rv,
                cum_supply=cum_supply,
                num_sim_run=200,
                # num_sim_run=10,
            )

    def prob_serving_upper_bound_for_given_combination_size(
        self,
        demand_rv: random_variable.RandomVariable,
        combination_size: int,
        maximal_load: float = 1,
    ) -> float:
        # log(DEBUG, "Started",
        #     demand_rv=demand_rv,
        #     combination_size=combination_size,
        #     maximal_load=maximal_load,
        # )

        if combination_size <= 2:
            span_size_to_freq_map = self.storage_design.get_span_size_to_freq_map(combination_size)
        else:
            span_size_to_freq_map = self.storage_design.get_span_size_to_freq_map_w_monte_carlo(
                combination_size=combination_size,
                num_samples=1000,
            )
        # log(DEBUG, "",
        #     demand_rv=demand_rv,
        #     d=self.d,
        #     combination_size=combination_size,
        #     span_size_to_freq_map=span_size_to_freq_map,
        # )

        return sum(
            freq * self.prob_cum_demand_leq_cum_supply(
                combination_size=combination_size,
                demand_rv=demand_rv,
                span_size=span_size,
                maximal_load=maximal_load,
            )
            for span_size, freq in span_size_to_freq_map.items()
        )

    def prob_serving_upper_bound(
        self,
        demand_rv: random_variable.RandomVariable,
        num_active_objs: int,
        max_combination_size: int,
        maximal_load: float = 1,
    ) -> float:
        return min(
            self.prob_serving_upper_bound_for_given_combination_size(
                demand_rv=demand_rv,
                combination_size=combination_size,
                maximal_load=maximal_load,
            )
            # ) ** scipy.special.comb(num_active_objs, combination_size)
            # ) ** (num_active_objs // combination_size)
            # ) ** (num_active_objs - combination_size + 1)
            for combination_size in range(2, max_combination_size + 1)
        )

    def prob_serving_lower_bound(
        self,
        demand_rv: random_variable.RandomVariable,
        num_active_objs: int,
        max_combination_size: int,
        maximal_load: float = 1,
    ) -> float:
        return math.prod(
            [
                self.prob_serving_upper_bound_for_given_combination_size(
                    demand_rv=demand_rv,
                    combination_size=combination_size,
                    maximal_load=maximal_load,
                )
                for combination_size in range(2, max_combination_size + 1)
            ]
        )

    def prob_serving_upper_bound_for_bernoulli_demand(
        self,
        demand_rv: random_variable.Bernoulli,
        maximal_load: float = 1,
    ) -> float:
        num_active_objs_rv = scipy.stats.binom(self.k, demand_rv.p)

        if demand_rv.D > self.d:
            return num_active_objs_rv.pmf(0)

        prob = 0
        for num_active_objs in range(self.k + 1):
            span_size_to_freq_map = self.storage_design.get_span_size_to_freq_map_w_monte_carlo(
                combination_size=num_active_objs,
                num_samples=2000,
            )

            cum_demand = demand_rv.D * num_active_objs
            prob += num_active_objs_rv.pmf(num_active_objs) * sum(
                freq
                for cum_supply, freq in span_size_to_freq_map.items()
                if cum_supply >= cum_demand
            )

        return prob

    def prob_serving_upper_bound_for_bernoulli_demand_(
        self,
        demand_rv: random_variable.Bernoulli,
        maximal_load: float = 1,
    ) -> float:
        num_active_objs_rv = scipy.stats.binom(self.k, demand_rv.p)

        if demand_rv.D > self.d:
            return num_active_objs_rv.pmf(0)

        combination_size_to_span_size_to_freq_map = {}
        for combination_size in range(2, self.k + 1):
            combination_size_to_span_size_to_freq_map[combination_size] = (
                self.storage_design.get_span_size_to_freq_map_w_monte_carlo(
                    combination_size=combination_size,
                    num_samples=2000,
                )
            )

        prob = 0
        for num_active_objs in range(self.k + 1):
            prob_for_combination_size_list = []
            for combination_size in range(2, num_active_objs + 1):
                span_size_to_freq_map = combination_size_to_span_size_to_freq_map[combination_size]

                cum_demand = demand_rv.D * combination_size
                prob_for_combination_size = sum(
                    freq
                    for cum_supply, freq in span_size_to_freq_map.items()
                    if cum_supply >= cum_demand
                )
                prob_for_combination_size_list.append(
                    prob_for_combination_size
                    # prob_for_combination_size ** (num_active_objs - combination_size + 1)
                    # prob_for_combination_size ** scipy.special.comb(num_active_objs, combination_size)
                )

            # prob_helper = numpy.mean(prob_for_combination_size_list) if len(prob_for_combination_size_list) else 0
            prob_helper = min(prob_for_combination_size_list) if len(prob_for_combination_size_list) else 0
            # prob_helper = math.prod(prob_for_combination_size_list) if len(prob_for_combination_size_list) else 0
            prob += num_active_objs_rv.pmf(num_active_objs) * prob_helper

        # TODO: Remove.
        if prob > 1:
            return 1

        return prob


@dataclasses.dataclass
class ClusteringDesignModelForGivenDemandDistribution(StorageDesignModelForGivenDemandDistribution):
    def __post_init__(self):
        self.storage_design = design.ClusteringDesign(
            k=self.k, n=self.n, d=self.d, use_cvxpy=False
        )


@dataclasses.dataclass
class CyclicDesignModelForGivenDemandDistribution(StorageDesignModelForGivenDemandDistribution):
    def __post_init__(self):
        self.storage_design = design.CyclicDesign(
            k=self.k, n=self.n, d=self.d, shift_size=1, use_cvxpy=False
        )


@dataclasses.dataclass
class RandomDesignModelForGivenDemandDistribution(StorageDesignModelForGivenDemandDistribution):
    def __post_init__(self):
        self.storage_design = design.RandomExpanderDesign(
            k=self.k, n=self.n, d=self.d, use_cvxpy=False
        )


@dataclasses.dataclass
class BlockDesignModelForGivenDemandDistribution(StorageDesignModelForGivenDemandDistribution):
    def __post_init__(self):
        self.storage_design = design.RandomBlockDesign(
            k=self.k, n=self.n, d=self.d, use_cvxpy=False
        )


@dataclasses.dataclass
class RandomDesignModelForExpDemand(ReplicaDesignModel):
    average_object_demand: float

    def prob_serving_upper_bound_w_complexes(
        self,
        maximal_load: float = 1,
    ) -> float:
        return min(
            self.prob_serving_upper_bound_w_complexes_for_given_num_objs(
                num_objs=num_objs,
                maximal_load=maximal_load,
            )
            for num_objs in range(2, self.k)
        )

    def prob_serving_upper_bound_w_complexes_for_given_num_objs(
        self,
        num_objs: int,
        maximal_load: float = 1,
    ) -> float:
        term_list = []
        return sum(
            (
                self.prob_span(num_objs=num_objs, span=span)
                * self.prob_cum_demand_leq_cum_supply(
                    num_objs=num_objs,
                    cum_supply=(span * maximal_load),
                )
            )
            for span in range(self.d, min(num_objs * self.d, self.n))
        )

        return sum(term_list)

    def prob_span(self, num_objs: int, span: int) -> float:
        """Uses the following results:
        * Let us distribute `N` balls uniformly at random among `n` boxes.
        Let `z` denote the number of boxes which remain empty. Weiss [1] proved:
        If `N(n)/n to alpha > 0` as `n to infty`, then `z` is Gaussian in the limit with
        `E[z] = n * exp{-N/n}`,
        `Var[z] = E[z] * (1 - (1 + N/n)exp{-N/n})`.

        * Let us repeat the above experimenting by distributing `n^` groups of `m` particles
        over `N` cells. Let `z^` be the counterpart of `z`. Sevast’yanov [2] proved:
        `z^ to z` in distribution as `n = n^ * m to infty`.

        Refs:
        [1] Weiss, "Limiting Distributions in Some Occupancy Problems", 1956.
        [2] Sevast’yanov, "Limit Theorems in a Scheme for Allocation of Particles in Cells", 1966.

        Note: Mapping of vars in code to vars in math:
        num_objs -> n^
        d -> m  (group size)
        n -> N
        """

        if num_objs == 0 or span < self.d or span > self.n:
            return 0

        N_over_n = num_objs * self.d / self.n
        exp_N_over_n = math.exp(-N_over_n)
        E_z = self.n * exp_N_over_n
        Var_z = E_z * (1 - (1 + N_over_n) * exp_N_over_n)

        z_rv = random_variable.Normal(mu=E_z, sigma=math.sqrt(Var_z))
        return z_rv.pdf(self.n - span)

    def prob_cum_demand_leq_cum_supply(
        self,
        num_objs: int,
        cum_supply: float,
    ) -> float:
        return scipy.stats.erlang.cdf(cum_supply, a=num_objs, loc=0, scale=self.average_object_demand)
