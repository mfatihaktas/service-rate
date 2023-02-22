from src.utils.debug import *


class DisjointSets():
    """Ref:
    - https://plainenglish.io/blog/union-find-data-structure-in-python-8e55369e2a4f
    """

    def __init__(self, n: int):
        # Initially, all elements are single element subsets
        self._parents = [node for node in range(n)]
        self._num_nodes = [1 for _ in range(n)]

    def find(self, u: int) -> int:
        while u != self._parents[u]:
            # path compression technique
            self._parents[u] = self._parents[self._parents[u]]
            u = self._parents[u]

        return u

    def is_connected(self, u: int, v: int) -> bool:
        return self.find(u) == self.find(v)

    def get_connected_component_size(self, u: int) -> int:
        return self._num_nodes[self.find(u)]

    def do_intersect(self, x_set: set[int], y_set: set[int]) -> bool:
        # log(DEBUG, "Started", x_set=x_set, y_set=y_set)

        for x in x_set:
            for y in y_set:
                if self.is_connected(x, y):
                    return True

        return False

    def union(self, u: int, v: int):
        # Union by rank optimization
        root_u, root_v = self.find(u), self.find(v)

        if root_u == root_v:
            return True
        if self._num_nodes[root_u] > self._num_nodes[root_v]:
            self._parents[root_v] = root_u
        elif self._num_nodes[root_v] > self._num_nodes[root_u]:
            self._parents[root_u] = root_v
        else:
            self._parents[root_u] = root_v
            self._num_nodes[root_v] += 1

        return False
