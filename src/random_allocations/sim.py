import collections
import random

from src.utils.debug import *


def sim_num_empty_cells(n: int, m: int, d: int) -> float:
    cell_id_list = list(range(n))
    cell_id_set = set()
    for _ in range(m):
        cell_ids = random.sample(cell_id_list, d)
        for cell_id in cell_ids:
            cell_id_set.add(cell_id)

    return n - len(cell_id_set)


def sim_mean_num_empty_cells_list(
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
        mean_num_empty_cells = numpy.mean(
            [
                sim_num_empty_cells(n=n, m=m, d=d)
                for _ in range(num_sample)
            ]
        )

        mean_num_empty_cells_list.append(mean_num_empty_cells)

    log(DEBUG, "Done", n=n, m=m, d=d)


def sim_num_empty_cells_to_prob_map(
    n: int,
    m: int,
    d: int,
    num_sample: int,
) -> list[float]:
    log(DEBUG, "Started",
        n=n,
        m=m,
        d=d,
        num_sample=num_sample,
    )

    num_empty_cells_to_counter_map = collections.defaultdict(int)
    for _ in range(num_sample):
        num_empty_cells = sim_num_empty_cells(n=n, m=m, d=d)
        num_empty_cells_to_counter_map[num_empty_cells] += 1

    num_empty_cells_to_prob_map = {
        num_empty_cells: counter / num_sample
        for num_empty_cells, counter in num_empty_cells_to_counter_map.items()
    }
    log(DEBUG, "Done", n=n, m=m, d=d, num_empty_cells_to_prob_map=num_empty_cells_to_prob_map)

    return num_empty_cells_to_prob_map
