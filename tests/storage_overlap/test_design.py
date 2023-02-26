import collections
import pytest

from typing import Tuple

from src.model import demand
from src.storage_overlap import design

from src.utils.debug import *


@pytest.fixture(
    scope="session",
    params=[
        # (4, 4, 2),
        (12, 12, 3),
    ],
)
def k_n_d(request) -> Tuple[int, int, int]:
    return request.param


@pytest.fixture
def use_cvxpy() -> bool:
    return False


def test_ClusteringDesign(
    k_n_d: Tuple[int, int, int],
    use_cvxpy: bool,
):
    k, n, d = k_n_d
    clustering_design = design.ClusteringDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy)

    log(INFO, "",
        clustering_design=clustering_design,
        obj_id_to_node_id_set_map=clustering_design.obj_id_to_node_id_set_map,
    )


def test_CyclicDesign(
    k_n_d: Tuple[int, int, int],
    use_cvxpy: bool,
):
    k, n, d = k_n_d

    cyclic_design = design.CyclicDesign(k=k, n=n, d=d, shift_size=1, use_cvxpy=use_cvxpy)

    log(INFO, "",
        cyclic_design=cyclic_design,
        obj_id_to_node_id_set_map=cyclic_design.obj_id_to_node_id_set_map,
    )


def test_TwoXORDesign():
    # k, n = 7, 7
    # k = 106
    k = 121
    # k = 124
    n = k
    xor_design = design.TwoXORDesign(k=k, n=n, d=3)

    log(INFO, "",
        xor_design=xor_design,
    )


def test_get_node_overlap_size_to_counter_map(
    k_n_d: Tuple[int, int, int],
    use_cvxpy: bool,
):
    k, n, d = k_n_d

    clustering_design = design.ClusteringDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy)
    log(INFO, "clustering_design: ",
        clustering_design=clustering_design,
        node_overlap_size_to_counter_map=clustering_design.get_node_overlap_size_to_counter_map()
    )

    cyclic_design = design.CyclicDesign(k=k, n=n, d=d, shift_size=1, use_cvxpy=use_cvxpy)
    log(INFO, "cycling_design: ",
        cyclic_design=cyclic_design,
        node_overlap_size_to_counter_map=cyclic_design.get_node_overlap_size_to_counter_map()
    )

    random_design = design.RandomDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy)
    log(INFO, "random_design: ",
        random_design=random_design,
        node_overlap_size_to_counter_map=random_design.get_node_overlap_size_to_counter_map()
    )


def test_is_demand_vector_covered(use_cvxpy: bool):
    def check_is_demand_vector_covered_equality(replica_design: design.ReplicaDesign):
        num_sample = 100

        for num_popular_obj in range(1, 10):
            for popular_obj_demand in range(2, replica_design.d + 1):
                log(DEBUG, "> ",
                    num_popular_obj=num_popular_obj,
                    popular_obj_demand=popular_obj_demand,
                )

                for demand_vector in demand.sample_demand_vectors_w_zipf_law(
                    num_obj=replica_design.k,
                    num_popular_obj=num_popular_obj,
                    cum_demand=num_popular_obj * popular_obj_demand,
                    zipf_tail_index=0,
                    num_sample=num_sample,
                ):
                    check(
                        (
                            replica_design.is_demand_vector_covered(demand_vector=demand_vector)
                            == replica_design.is_demand_vector_covered_w_joblib(demand_vector=demand_vector)
                            # == replica_design.is_demand_vector_covered_alternative(demand_vector=demand_vector)
                        ),
                        "is_demand_vector_covered()'s returned different results",
                        replica_design=replica_design,
                        obj_id_to_node_id_set_map=replica_design.obj_id_to_node_id_set_map,
                        num_popular_obj=num_popular_obj,
                        popular_obj_demand=popular_obj_demand,
                        demand_vector=demand_vector,
                    )

    k = 12
    n = k
    d = 4

    # replica_design = design.ClusteringDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy)
    # replica_design = design.CyclicDesign(k=k, n=n, d=d, shift_size=1, use_cvxpy=use_cvxpy)
    replica_design = design.RandomDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy)
    check_is_demand_vector_covered_equality(replica_design=replica_design)


def test_combination_size_to_is_demand_covered_map(use_cvxpy: bool):
    def check_combination_size_to_is_demand_covered_map(replica_design: design.ReplicaDesign):
        num_sample = 1000

        for popular_obj_demand in range(2, replica_design.d + 1):
            combination_size_and_is_demand_covered_tuple_to_counter_map = collections.defaultdict(int)

            for num_popular_obj in range(1, 10):
                # log(DEBUG, "> ",
                #     num_popular_obj=num_popular_obj,
                #     popular_obj_demand=popular_obj_demand,
                # )

                for demand_vector in demand.sample_demand_vectors_w_zipf_law(
                    num_obj=replica_design.k,
                    num_popular_obj=num_popular_obj,
                    cum_demand=num_popular_obj * popular_obj_demand,
                    zipf_tail_index=0,
                    num_sample=num_sample,
                ):
                    combination_size_and_is_demand_covered_tuple = tuple(
                        [
                            (
                                combination_size,
                                replica_design.is_demand_vector_covered_for_given_combination_size(
                                    demand_vector=demand_vector,
                                    combination_size=combination_size
                                )
                            )
                            for combination_size in range(2, replica_design.d + 1)
                        ]
                    )
                    combination_size_and_is_demand_covered_tuple_to_counter_map[combination_size_and_is_demand_covered_tuple] += 1

            log(DEBUG, "",
                d=replica_design.d,
                popular_obj_demand=popular_obj_demand,
                combination_size_and_is_demand_covered_tuple_to_counter_map=combination_size_and_is_demand_covered_tuple_to_counter_map,
            )

    k = 120
    n = k
    d = 4

    # replica_design = design.ClusteringDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy)
    replica_design = design.CyclicDesign(k=k, n=n, d=d, shift_size=1, use_cvxpy=use_cvxpy)
    # replica_design = design.RandomDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy)
    check_combination_size_to_is_demand_covered_map(replica_design=replica_design)


def test_get_num_objs_to_span_size_map(use_cvxpy: bool):
    k = 12
    # k = 120
    n = k
    d = 3

    # replica_design = design.ClusteringDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy)
    # replica_design = design.CyclicDesign(k=k, n=n, d=d, shift_size=1, use_cvxpy=use_cvxpy)
    replica_design = design.RandomDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy)

    num_popular_obj = 5
    num_obj_to_span_size_map = replica_design.get_num_obj_to_span_size_map(num_popular_obj=num_popular_obj)
    log(DEBUG, "",
        replica_design=replica_design,
        num_obj_to_span_size_map=num_obj_to_span_size_map
    )
