from src.sim import random_variable


def sim_scan_stats(n: int, m: int, X: random_variable.RandomVariable) -> float:
    x_list = [X.sample() for i in range(n)]

    scan_stat = 0
    for i in range(n - m - 1):
        scan = sum(x_list[i:i + m])
        scan_stat = max(scan_stat, scan)

    return scan_stat


def sim_scan_stats_cdf(
    n: int,
    m: int,
    x: int,
    X: random_variable.RandomVariable,
    num_samples: int,
) -> float:
    num_samples_leq_x = 0
    for _ in range(num_samples):
        scan_stats = sim_scan_stats(n=n, m=m, X=X)
        if scan_stats <= x:
            num_samples_leq_x += 1

    return num_samples_leq_x / num_samples
