"""`m` balls are thrown into `n` bins, which are chosen at random with replacement.
"""

import collections
import random


def sim_max_num_balls_in_any_bin(n: int, m: int) -> int:
    bin_id_to_num_balls_map = collections.defaultdict(int)
    for _ in range(m):
        bin_id = random.randint(0, n - 1)
        bin_id_to_num_balls_map[bin_id] += 1

    return max(bin_id_to_num_balls_map.values())


def sim_prob_max_num_balls_leq_u(
    n: int,
    m: int,
    u: int,
    num_samples: int,
) -> float:
    return (
        sum(
            sim_max_num_balls_in_any_bin(n=n, m=m) <= u
            for _ in range(num_samples)
        ) / num_samples
    )


def sim_max_num_balls_to_prob_map(
    n: int,
    m: int,
    num_samples: int,
) -> dict[int, float]:
    max_num_balls_list = [
        sim_max_num_balls_in_any_bin(n=n, m=m)
        for _ in range(num_samples)
    ]

    max_num_balls_to_count_map = collections.Counter(max_num_balls_list)
    max_num_balls_to_prob_map = {
        max_num_balls: count / num_samples
        for max_num_balls, count in max_num_balls_to_count_map.items()
    }

    return max_num_balls_to_prob_map
