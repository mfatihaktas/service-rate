import itertools


def get_demand_vectors(
    num_objs: int,
    demand_ordered_for_most_popular_objs: list[float],
) -> list[list[float]]:
    num_popular_objs = len(demand_ordered_for_most_popular_objs)
    base_demand_vector = (
        demand_ordered_for_most_popular_objs
        + (num_objs - num_popular_objs) * [0]
    )

    return [list(p) for p in itertools.permutations(base_demand_vector)]
