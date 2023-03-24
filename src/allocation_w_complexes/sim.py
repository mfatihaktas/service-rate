import collections
import itertools
import mpmath as mp
import random

from src.utils.debug import *


def sim_num_nonempty_cells(n: int, m: int, d: int) -> float:
    cell_id_list = list(range(n))

    cell_id_set = set()
    for _ in range(m):
        cell_ids = random.sample(cell_id_list, d)
        for cell_id in cell_ids:
            cell_id_set.add(cell_id)

    return len(cell_id_set)


def sim_mean_num_nonempty_cells_list(
    n: int,
    m: int,
    d: int,
    num_samples: int,
    num_sim_run: int = 1,
) -> list[float]:
    log(DEBUG, "Started",
        n=n,
        m=m,
        d=d,
        num_samples=num_samples,
        num_sim_run=num_sim_run,
    )

    mean_num_empty_cells_list = []
    for _ in range(num_sim_run):
        mean_num_nonempty_cells = numpy.mean(
            [
                sim_num_nonempty_cells(n=n, m=m, d=d)
                for _ in range(num_samples)
            ]
        )

        mean_num_empty_cells_list.append(mean_num_nonempty_cells)

    log(DEBUG, "Done", n=n, m=m, d=d)


def sim_num_nonempty_cells_to_prob_map(
    n: int,
    m: int,
    d: int,
    num_samples: int,
) -> dict[int, float]:
    log(DEBUG, "Started",
        n=n,
        m=m,
        d=d,
        num_samples=num_samples,
    )

    num_nonempty_cells_to_counter_map = collections.defaultdict(int)
    for _ in range(num_samples):
        num_nonempty_cells = sim_num_nonempty_cells(n=n, m=m, d=d)
        num_nonempty_cells_to_counter_map[num_nonempty_cells] += 1

    num_nonempty_cells_to_prob_map = {
        num_nonempty_cells: counter / num_samples
        for num_nonempty_cells, counter in num_nonempty_cells_to_counter_map.items()
    }
    log(DEBUG, "Done", n=n, m=m, d=d, num_nonempty_cells_to_prob_map=num_nonempty_cells_to_prob_map)

    return num_nonempty_cells_to_prob_map


def sim_num_nonempty_cells_to_tail_prob_map(
    n: int,
    m: int,
    d: int,
    num_samples: int,
) -> dict[int, float]:
    log(DEBUG, "Started",
        n=n,
        m=m,
        d=d,
        num_samples=num_samples,
    )

    num_nonempty_cells_to_counter_map = collections.defaultdict(int)
    for _ in range(num_samples):
        num_nonempty_cells = sim_num_nonempty_cells(n=n, m=m, d=d)
        num_nonempty_cells_to_counter_map[num_nonempty_cells] += 1

    num_nonempty_cells_to_tail_prob_map = {
        num_nonempty_cells: sum(
            counter / num_samples
            for num_nonempty_cells_, counter in num_nonempty_cells_to_counter_map.items()
            if num_nonempty_cells_ >= num_nonempty_cells
        )
        for num_nonempty_cells in range(d, n + 1)
    }

    log(DEBUG, "Done", n=n, m=m, d=d, num_nonempty_cells_to_tail_prob_map=num_nonempty_cells_to_tail_prob_map)

    return num_nonempty_cells_to_tail_prob_map


def sim_if_span_of_every_t_complexes_geq_u(
    n: int,
    m: int,
    d: int,
    t: int,
    u: int,
) -> bool:
    cell_id_list = list(range(n))

    complex_list = []
    for _ in range(m):
        complex_ = set(random.sample(cell_id_list, d))

        for complex_combination in itertools.combinations(complex_list, r=(t - 1)):
            union = set.union(*complex_combination, complex_)
            # log(WARNING, "", union=union)

            if len(union) < u:
                return False

        complex_list.append(complex_)

    return True


def sim_if_span_of_every_t_complexes_geq_u_alternative(
    n: int,
    m: int,
    d: int,
    t: int,
    u: int,
) -> bool:
    cell_id_list = list(range(n))

    for m_ in range(1, m):
        complex_ = set(random.sample(cell_id_list, d))

        # num_complex_combinations = int(mp.binomial(m_, t - 1))
        num_complex_combinations = m_
        # log(WARNING, "", num_complex_combinations=num_complex_combinations)

        for _ in range(num_complex_combinations):
            complex_combination = [
                set(random.sample(cell_id_list, d))
                for _ in range(t - 1)
            ]
            union = set.union(*complex_combination, complex_)

            if len(union) < u:
                return False

    return True


def sim_prob_span_of_every_t_complexes_geq_u(
    n: int,
    m: int,
    d: int,
    t: int,
    u: int,
    num_samples: int,
) -> float:
    log(DEBUG, "Started",
        n=n,
        m=m,
        d=d,
        t=t,
        u=u,
        num_samples=num_samples,
    )

    prop = (
        sum(
            sim_if_span_of_every_t_complexes_geq_u(n=n, m=m, d=d, t=t, u=u)
            for _ in range(num_samples)
        )
        / num_samples
    )

    log(DEBUG, "Done", n=n, m=m, d=d, t=t, u=u, prop=prop)

    return prop


def sim_prob_span_of_every_t_complexes_geq_u_alternative(
    n: int,
    m: int,
    d: int,
    t: int,
    u: int,
    num_samples: int,
) -> float:
    log(DEBUG, "Started",
        n=n,
        m=m,
        d=d,
        t=t,
        u=u,
        num_samples=num_samples,
    )

    prop = (
        sum(
            sim_if_span_of_every_t_complexes_geq_u_alternative(n=n, m=m, d=d, t=t, u=u)
            for _ in range(num_samples)
        )
        / num_samples
    )

    log(DEBUG, "Done", n=n, m=m, d=d, t=t, u=u, prop=prop)

    return prop
