import collections
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
    num_sample: int,
    num_sim_run: int = 1,
) -> list[float]:
    log(DEBUG, "Started",
        n=n,
        m=m,
        d=d,
        num_sample=num_sample,
        num_sim_run=num_sim_run,
    )

    mean_num_empty_cells_list = []
    for _ in range(num_sim_run):
        mean_num_nonempty_cells = numpy.mean(
            [
                sim_num_nonempty_cells(n=n, m=m, d=d)
                for _ in range(num_sample)
            ]
        )

        mean_num_empty_cells_list.append(mean_num_nonempty_cells)

    log(DEBUG, "Done", n=n, m=m, d=d)


def sim_num_nonempty_cells_to_prob_map(
    n: int,
    m: int,
    d: int,
    num_sample: int,
) -> dict[int, float]:
    log(DEBUG, "Started",
        n=n,
        m=m,
        d=d,
        num_sample=num_sample,
    )

    num_nonempty_cells_to_counter_map = collections.defaultdict(int)
    for _ in range(num_sample):
        num_nonempty_cells = sim_num_nonempty_cells(n=n, m=m, d=d)
        num_nonempty_cells_to_counter_map[num_nonempty_cells] += 1

    num_nonempty_cells_to_prob_map = {
        num_nonempty_cells: counter / num_sample
        for num_nonempty_cells, counter in num_nonempty_cells_to_counter_map.items()
    }
    log(DEBUG, "Done", n=n, m=m, d=d, num_nonempty_cells_to_prob_map=num_nonempty_cells_to_prob_map)

    return num_nonempty_cells_to_prob_map


def sim_num_nonempty_cells_to_tail_prob_map(
    n: int,
    m: int,
    d: int,
    num_sample: int,
) -> dict[int, float]:
    log(DEBUG, "Started",
        n=n,
        m=m,
        d=d,
        num_sample=num_sample,
    )

    num_nonempty_cells_to_counter_map = collections.defaultdict(int)
    for _ in range(num_sample):
        num_nonempty_cells = sim_num_nonempty_cells(n=n, m=m, d=d)
        num_nonempty_cells_to_counter_map[num_nonempty_cells] += 1

    num_nonempty_cells_to_tail_prob_map = {
        num_nonempty_cells: sum(
            counter / num_sample
            for num_nonempty_cells_, counter in num_nonempty_cells_to_counter_map.items()
            if num_nonempty_cells_ >= num_nonempty_cells
        )
        for num_nonempty_cells in range(d, n + 1)
    }

    log(DEBUG, "Done", n=n, m=m, d=d, num_nonempty_cells_to_tail_prob_map=num_nonempty_cells_to_tail_prob_map)

    return num_nonempty_cells_to_tail_prob_map