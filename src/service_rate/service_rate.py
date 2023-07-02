import cvxpy
import numpy
import scipy.spatial
import string
import sys

from typing import Tuple, Union

from src.service_rate import service_rate_utils

from src.utils.debug import *
from src.utils.plot import *


class ServiceRateInspectorBase:
    def __init__(
        self,
        k: int,
        n: int,
        m: int,
        C: int = 1,
    ):
        # Number of original objects
        self.k = k
        # Number of object copies stored
        self.n = n
        # Number of nodes
        self.m = m
        # Capacity of each node
        self.C = C

    def is_in_cap_region(
        self,
        obj_demand_list: list[float],
        maximal_load: float = 1,
    ) -> bool:
        """
        demand_vector = numpy.array(obj_demand_list).reshape((self.k, 1))

        x = cvxpy.Variable(shape=(self.l, 1), name="x")

        obj = cvxpy.Maximize(cvxpy.sum(x))
        # obj = cvxpy.Maximize(0)
        # constraints = [self.M @ x <= self.C, x >= 0, self.T @ x == demand_vector]
        constraints = [self.M @ x <= self.C, x >= 0, self.T @ x - demand_vector <= 0.01]

        prob = cvxpy.Problem(obj, constraints)
        opt_value = service_rate_utils.solve_prob(prob)
        # log(DEBUG, "", opt_value=opt_value)

        return opt_value is not None
        """
        max_load = self.max_load(obj_demand_list)
        # log(DEBUG, "", max_load=max_load, obj_demand_list=obj_demand_list)
        if max_load is None:
            return False

        return max_load <= (maximal_load + 0.001)

    def max_load(self, obj_demand_list: list[float]) -> float:
        """Returns the load at the the maximally loaded node (i.e., maximal load)
        after the object demand is distributed across the nodes in a way that minimizes
        the maximal load.
        """

        demand_vector = numpy.array(obj_demand_list).reshape((self.k, 1))
        x = cvxpy.Variable(shape=(self.l, 1), name="x")

        obj = cvxpy.Minimize(cvxpy.max(self.M @ x))
        constraints = [x >= 0, self.T @ x == demand_vector]

        prob = cvxpy.Problem(obj, constraints)
        opt_value = service_rate_utils.solve_prob(prob)

        # blog(x_val=x.value)
        return opt_value / self.C if opt_value else None

    def find_max_demand_for_obj(self, obj_id: int) -> float:
        x = cvxpy.Variable(shape=(self.l, 1), name="x")

        log(DEBUG, f"T[{obj_id}]= {self.T[obj_id]}")
        obj = cvxpy.Maximize(self.T[obj_id] @ x)
        constraints = [self.M @ x <= self.C, x >= 0]

        prob = cvxpy.Problem(obj, constraints)
        opt_value = service_rate_utils.solve_prob(prob)

        return opt_value

    def find_vertices_on_cap_region_boundary(
        self,
        obj_id_list: list[int],
        num_points_on_each_axis_to_query_boundary: int = 5,
    ) -> list[numpy.array]:
        # log(DEBUG, "Started", obj_id_list=obj_id_list, num_points_on_each_axis_to_query_boundary=num_points_on_each_axis_to_query_boundary)

        # check(self.k == 2, "Only defined for k= 2")

        obj_id_to_max_demand_plus_epsilon_map = {
            obj_id: 1.1 * self.find_max_demand_for_obj(obj_id=obj_id)
            for obj_id in range(self.k)
        }
        log(DEBUG, "", obj_id_to_max_demand_plus_epsilon_map=obj_id_to_max_demand_plus_epsilon_map)

        def find_vertex(obj_id: int, obj_demand: float) -> numpy.array:
            obj_demand_vector = numpy.zeros(shape=(self.k, 1))
            for i in obj_id_list:
                obj_demand_vector[i, 0] = obj_id_to_max_demand_plus_epsilon_map[i]
            obj_demand_vector[obj_id, 0] = obj_demand
            log(DEBUG, f"> obj_id= {obj_id}, obj_demand= {obj_demand}", obj_demand_vector=obj_demand_vector)

            x = cvxpy.Variable(shape=(self.l, 1), name="x")

            obj = cvxpy.Minimize(cvxpy.norm(self.T @ x - obj_demand_vector))
            constraints = [self.M @ x <= self.C, x >= 0]

            prob = cvxpy.Problem(obj, constraints)
            opt_value = service_rate_utils.solve_prob(prob)
            check(opt_value is not None, "Optimization problem could not be solved!",
                  obj_id=obj_id, obj_demand=obj_demand
            )

            return numpy.matmul(self.T, x.value)

        vertex_list = [[0] * len(obj_id_list)]
        for obj_id in obj_id_list:
            max_demand = obj_id_to_max_demand_plus_epsilon_map[obj_id]
            for obj_demand in numpy.linspace(start=0, stop=max_demand, num=num_points_on_each_axis_to_query_boundary, endpoint=True):
                vertex = find_vertex(obj_id=obj_id, obj_demand=obj_demand)
                vertex = numpy.concatenate(vertex).ravel().tolist()
                vertex = [vertex[i] for i in obj_id_list]
                log(DEBUG, f"> obj_demand= {obj_demand}", vertex=vertex)
                vertex_list.append(vertex)

        # log(DEBUG, "Done", vertex_list=vertex_list)
        return sorted(vertex_list)

    def load_across_nodes(self, obj_demand_list: list[float]) -> list[float]:
        """Returns the load at the nodes after the object demand is distributed across
        the nodes in a way that minimizes the maximal load.
        """

        demand_vector = numpy.array(obj_demand_list).reshape((self.k, 1))
        x = cvxpy.Variable(shape=(self.l, 1), name="x")

        obj = cvxpy.Minimize(cvxpy.max(self.M @ x))
        constraints = [x >= 0, self.T @ x == demand_vector]

        prob = cvxpy.Problem(obj, constraints)
        _ = service_rate_utils.solve_prob(prob)

        # log(DEBUG, "", x_val=x.value)
        load_across_nodes = numpy.matmul(self.M, x.value)

        return load_across_nodes / self.C

    def load_across_nodes_when_obj_demands_distributed_evenly_across_repair_sets(
        self, obj_demand_list: list[float]
    ) -> list[float]:
        """Returns the load at the nodes after each object demand is distributed evenly across
        its repair sets.
        """

        log(DEBUG, f"m= {self.m}")

        load_across_nodes = [0] * self.m

        for orig_obj_id, obj_demand in enumerate(obj_demand_list):
            repair_set_list = self.orig_obj_id_to_repair_sets_w_obj_ids_map[orig_obj_id]

            obj_demand_per_repair_set = obj_demand / len(repair_set_list)

            for repair_set in repair_set_list:
                for obj_id in repair_set:
                    node_id = self.obj_id_to_node_id_map[obj_id]
                    load_across_nodes[node_id] += obj_demand_per_repair_set

        return load_across_nodes

    def load_across_nodes_when_obj_demands_replicated_to_repair_sets(
        self, obj_demand_list: list[float]
    ) -> list[float]:
        """Returns the load at the nodes after each object demand is replicated to all
        its repair sets.

        Note: This load replication is only for test purposes, it obviously does not implement a real
        load distribution logic.
        """

        log(DEBUG, f"m= {self.m}")

        load_across_nodes = [0] * self.m

        for orig_obj_id, obj_demand in enumerate(obj_demand_list):
            repair_set_list = self.orig_obj_id_to_repair_sets_w_obj_ids_map[orig_obj_id]

            for repair_set in repair_set_list:
                for obj_id in repair_set:
                    node_id = self.obj_id_to_node_id_map[obj_id]
                    load_across_nodes[node_id] += obj_demand

        return load_across_nodes


class ServiceRateInspector(ServiceRateInspectorBase):
    """Defines a storage system that distributes n objects across m
    nodes. Input arguments and class variables are described in the README.
    """

    def __init__(
        self,
        m: int,
        C: float,
        G: numpy.ndarray,
        obj_id_to_node_id_map: dict,
        redundancy_w_two_xors: bool = False,
        max_repair_set_size: int = None,
        compute_halfspace_intersections: bool = False,
    ):
        # Object composition matrix
        self.G = G
        # Map from object id's to node id's
        self.obj_id_to_node_id_map = obj_id_to_node_id_map
        self.compute_halfspace_intersections = compute_halfspace_intersections

        numpy.set_printoptions(threshold=sys.maxsize)

        # log(DEBUG, f"G= \n{self.G}")
        super().__init__(k=G.shape[0], n=G.shape[1], m=m, C=C)

        # Note: Repair sets are given in terms of obj ids,
        # in the sense of columns of G or keys of obj_id_to_node_id_map

        if redundancy_w_two_xors:
            self.orig_obj_id_to_repair_sets_w_obj_ids_map = service_rate_utils.get_orig_obj_id_to_repair_sets_w_obj_ids_map_for_redundancy_w_two_xors(
                n=self.n,
                G=self.G,
            )

        else:
            if max_repair_set_size is None:
                max_repair_set_size = self.k
            # log(DEBUG, "", max_repair_set_size=max_repair_set_size)

            self.orig_obj_id_to_repair_sets_w_obj_ids_map = service_rate_utils.get_orig_obj_id_to_repair_sets_w_obj_ids_map(
                k=self.k, n=self.n, G=self.G, max_repair_set_size=max_repair_set_size,
            )
            # self.orig_obj_id_to_repair_sets_w_obj_ids_map = service_rate_utils.get_orig_obj_id_to_repair_sets_w_obj_ids_map_w_joblib(
            #     k=self.k,
            #     n=self.n,
            #     G=self.G,
            #     max_repair_set_size=max_repair_set_size,
            # )

        # log(DEBUG, "", orig_obj_id_to_repair_sets_w_obj_ids_map=self.orig_obj_id_to_repair_sets_w_obj_ids_map)

        self.orig_obj_id_to_repair_sets_w_node_ids_map = service_rate_utils.get_orig_obj_id_to_repair_sets_w_node_ids_map(
            orig_obj_id_to_repair_sets_w_obj_ids_map=self.orig_obj_id_to_repair_sets_w_obj_ids_map,
            obj_id_to_node_id_map=self.obj_id_to_node_id_map,
        )
        # log(DEBUG, "",
        #     obj_id_to_node_id_map=self.obj_id_to_node_id_map,
        #     orig_obj_id_to_repair_sets_w_node_ids_map=self.orig_obj_id_to_repair_sets_w_node_ids_map,
        # )

        # Repair set list
        repair_set_w_obj_ids_list = []
        for obj in range(self.k):
            repair_set_w_obj_ids_list.extend(
                self.orig_obj_id_to_repair_sets_w_obj_ids_map[obj]
            )

        self.l = len(repair_set_w_obj_ids_list)

        # T
        self.T = service_rate_utils.get_T(
            num_orig_objects=self.k,
            total_num_repair_sets=self.l,
            orig_obj_to_repair_set_w_obj_ids_size_map={
                obj: len(repair_set)
                for obj, repair_set in self.orig_obj_id_to_repair_sets_w_obj_ids_map.items()
            },
        )
        # log(DEBUG, f"T= \n{self.T}")

        # M
        self.M = service_rate_utils.get_M(
            num_objects=self.n,
            num_nodes=self.m,
            repair_set_w_obj_ids_list=repair_set_w_obj_ids_list,
            obj_id_to_node_id_map=self.obj_id_to_node_id_map,
        )
        # log(DEBUG, f"M= \n{self.M}")

        if self.compute_halfspace_intersections:
            self.hull, self.boundary_points_in_rows = self.get_convex_hull_and_boundary_points()

    def __repr__(self):
        return (
            "ServiceRateInspector( \n"
            f"\t m= {self.m} \n"
            f"\t C= {self.C} \n"
            f"\t G= \n{self.G} \n"
            # f"\t obj_id_to_node_id_map= {self.obj_id_to_node_id_map} \n"
            f"\t M= \n{self.M} \n"
            f"\t T= \n{self.T} \n"
            ")"
        )

    def to_sysrepr(self):
        sym_list = string.ascii_lowercase[: self.k]
        node_list = [[] for _ in range(self.m)]
        for obj in range(self.n):
            ni = self.obj_id_to_node_id_map[obj]
            l = []
            for r in range(self.k):
                if self.G[r, obj] != 0:
                    num = int(self.G[r, obj])
                    l.append(
                        "{}{}".format(num, sym_list[r])
                        if num != 1
                        else "{}".format(sym_list[r])
                    )
            node_list[ni].append("+".join(l))

        return str(node_list)

    def get_convex_hull_and_boundary_points(self) -> Tuple[scipy.spatial.ConvexHull, numpy.array]:
        log(DEBUG, "Started")

        halfspaces = service_rate_utils.get_halfspaces(
            num_nodes=self.m,
            total_num_repair_sets=self.l,
            node_capacity=self.C,
            M=self.M,
        )

        boundary_point_list = [
            list(numpy.matmul(self.T, p)) for p in halfspaces.intersections
        ]
        boundary_points_in_rows = numpy.array(
            boundary_point_list
        ).reshape((len(boundary_point_list), self.k))

        log(DEBUG, "Calling scipy.spatial.ConvexHull", boundary_points_in_rows_shape=boundary_points_in_rows.shape)
        hull = scipy.spatial.ConvexHull(boundary_points_in_rows)
        log(DEBUG, "scipy.spatial.ConvexHull is done.")

        return hull, boundary_points_in_rows

    def min_cost(self, obj_demand_list: list[float]):
        # log(DEBUG, "Started;", obj_demand_list=obj_demand_list)

        demand_vector = numpy.array(obj_demand_list).reshape((self.k, 1))

        x = cvxpy.Variable(shape=(self.l, 1), name="x")

        cost_coeff_list = []
        for obj in range(self.k):
            repair_set_list = self.orig_obj_id_to_repair_sets_w_obj_ids_map[obj]
            for repair_set in repair_set_list:
                cost_coeff_list.append(len(repair_set))
        log(DEBUG, "", cost_coeff_list=cost_coeff_list)

        cost_coeff_row_vector = numpy.asarray(cost_coeff_list).reshape((1, self.l))
        obj = cvxpy.Minimize(cost_coeff_row_vector @ x)
        constraints = [self.M @ x <= self.C, x >= 0, self.T @ x == demand_vector]

        prob = cvxpy.Problem(obj, constraints)
        opt_value = service_rate_utils.solve_prob(prob)
        if opt_value is None:
            log(
                WARNING,
                "Object demand vector is not in the capacity region, "
                "thus cannot compute the cost.",
                prob_status=prob.status,
                obj_demand_list=obj_demand_list,
            )
            return None

        allocation_list = []
        for v_list in x.value.tolist():
            v = v_list[0]
            allocation_list.append(v if abs(v) > 0.001 else 0)
        log(DEBUG, f"allocation_list= {allocation_list}")

        cost = cost_coeff_row_vector @ x.value
        cost = cost.tolist()[0][0]
        # cost = cost / numpy.sum(x.value)
        return cost

    def get_in_cap_region_and_min_distance_to_boundary_w_cvxpy(
        self,
        obj_demand_list: list[float],
    ) -> Tuple[bool, float]:
        """Returns the min distance from obj_demand_list to the service rate boundary."""

        demand_vector = numpy.array(obj_demand_list).reshape((self.k, 1))
        x = cvxpy.Variable(shape=(self.l, 1), name="x")

        obj = cvxpy.Minimize(cvxpy.sum_squares(self.T @ x - demand_vector))
        in_cap_region = self.is_in_cap_region(obj_demand_list)
        if in_cap_region:
            constraints = [self.M @ x >= self.C, x >= 0]
        else:
            constraints = [self.M @ x <= self.C, x >= 0]

        prob = cvxpy.Problem(obj, constraints)
        opt_value = service_rate_utils.solve_prob(prob)
        # check(opt_value is not None, "Optimization program failed",
        #       obj_demand_list=obj_demand_list,
        #       in_cap_region=in_cap_region,
        #       self=self
        # )
        if opt_value is not None:
            return in_cap_region, float("Inf")

        # blog(x_val=x.value)
        point_closest_to_demand_vector = numpy.matmul(self.T, x.value)

        distance = numpy.sqrt(
            numpy.sum((point_closest_to_demand_vector - demand_vector) ** 2)
        )
        return in_cap_region, distance

    def min_distance_to_boundary_w_cvxpy_for_oleg(
        self, obj_demand_list: list[Union[float, str]]
    ) -> float:
        """Wrapper around `min_distance_to_boundary_w_cvxpy()` to accomodate the input
        and output requirements needed by Oleg's use case.
        """

        obj_demand_list = [float(i) for i in obj_demand_list]

        in_cap_region = self.is_in_cap_region(numpy.array(obj_demand_list))
        min_distance = self.min_distance_to_boundary_w_cvxpy(obj_demand_list)

        return -min_distance if in_cap_region else min_distance

    def min_distance_to_boundary_w_convex_hull(
        self, obj_demand_list: list[float]
    ) -> float:
        """Returns the min distance from obj_demand_list to the service rate boundary.

        Relies on
        self.halfspaces = scipy.spatial.HalfspaceIntersection(halfspaces, feasible_point)
        self.hull = scipy.spatial.ConvexHull(self.boundary_points_in_rows)
        which take very long to complete when the number of objects > ~30.

        TODO: Investigate why these two functions take so long to complete,
        and if we can make them run faster, or if we can find a different faster
        way to get the same results.

        TODO: This function does not work correctly for points that are inside the
        service rate region.
        """

        if self.compute_halfspace_intersections is False:
            log(
                WARNING,
                "Cannot compute min distance to service capacity boundary",
                compute_halfspace_intersection=self.compute_halfspace_intersections,
            )
            return None

        # https://stackoverflow.com/questions/42248202/find-the-projection-of-a-point-on-the-convex-hull-with-scipy
        def min_distance(p, v1, v2):
            """Finds projection p' of point p on the closest edge formed by points p1 and p2,
            and returns the distance from p to p'.
            """

            # blog(p=p, v1=v1, v2=v2)
            # v2 = v2 - v1
            l = numpy.sum(
                (v2 - v1) ** 2
            )  # compute the squared distance between the 2 vertices
            # blog(l=l, dot=numpy.dot(p-v1, v2-v1)[0] )
            t = numpy.max(
                [0.0, numpy.min([1.0, numpy.dot(p - v1, v2 - v1) / l])]
            )  # numpy.min([1., numpy.dot(p-v1, v2-v1)[0]/l] )
            # blog(dot=numpy.dot(p-v1, v2-v1), t=t)
            proj = v1 + t * (v2 - v1)
            return numpy.sqrt(numpy.sum((proj - p) ** 2))

        x = numpy.array(obj_demand_list).reshape((self.k, 1))
        min_dist = float("Inf")
        for i in range(len(self.hull.vertices)):
            m = min_distance(
                x.T,
                self.boundary_points_in_rows[self.hull.vertices[i]],
                self.boundary_points_in_rows[
                    self.hull.vertices[(i + 1) % len(self.hull.vertices)]
                ],
            )
            if m < min_dist:
                min_dist = m

        return min_dist

    def approx_min_distance_to_boundary(self, obj_demand_list: list[float]) -> float:
        """Returns the approximate min distance from obj_demand_list to the service rate boundary.

        Finds the approximate min distance as follows:
        - Draws a line L from the point of interest P (obj_demand_list) to origin.
        - If P is inside the capacity region, scales P by 2 (along L) until P lies
        outside the capacity region.
        - Performs binary search between origin and P to find the point at which
        L intersects with the capacity region boundary.

        Reason to have this approximation is min_distance_to_boundary() relies on computing
        self.halfspaces = scipy.spatial.HalfspaceIntersection(halfspaces, feasible_point)
        self.hull = scipy.spatial.ConvexHull(self.boundary_points_in_rows)
        , which take very long to complete when the number of objects > ~30.
        """

        def dist(x: numpy.array, y: numpy.array):
            return numpy.sqrt(numpy.sum(numpy.square(x - y)))

        obj_demand_list = [float(i) for i in obj_demand_list]
        u = numpy.array(obj_demand_list) + 0.001
        while self.is_in_cap_region(u):
            u *= 2

        l = 0
        while dist(l, u) > 0.05:
            m = (l + u) / 2
            if self.is_in_cap_region(m):
                l = m
            else:
                u = m

        point_on_boundary = (l + u) / 2
        log(DEBUG, "", point_on_boundary=point_on_boundary)
        return dist(point_on_boundary, numpy.array(obj_demand_list))


class ServiceRateInspectorForStorageWithReplicasOrStripes(ServiceRateInspectorBase):
    def __init__(
        self,
        obj_id_to_node_id_set_map: dict[int, set[int]],
        C: int = 1,
    ):
        k = len(obj_id_to_node_id_set_map)
        node_id_set = set(
            node_id
            for node_id_set in obj_id_to_node_id_set_map.values()
            for node_id in node_id_set
        )
        m = max(node_id_set) + 1

        super().__init__(k=k, n=None, m=m, C=C)
        self.obj_id_to_node_id_set_map = obj_id_to_node_id_set_map

        self.obj_id_to_repair_node_id_sets_map = self.get_obj_id_to_repair_node_id_sets_map()

        self.obj_id_to_num_repair_sets_map = {
            obj_id: len(repair_node_id_sets)
            for obj_id, repair_node_id_sets in self.obj_id_to_repair_node_id_sets_map.items()
        }
        self.l = sum(self.obj_id_to_num_repair_sets_map.values())

        self.T = self.get_T()
        self.M = self.get_M()

        # log(DEBUG, "",
        #     obj_id_to_repair_node_id_sets_map=self.obj_id_to_repair_node_id_sets_map,
        #     obj_id_to_num_repair_sets_map=self.obj_id_to_num_repair_sets_map,
        #     l=self.l,
        #     T=self.T,
        #     M=self.M,
        # )

    def get_T(self) -> numpy.array:
        total_num_repair_sets = self.l
        T = numpy.zeros((self.k, total_num_repair_sets))
        i = 0
        for obj_id in range(self.k):
            j = i + self.obj_id_to_num_repair_sets_map[obj_id]
            T[obj_id, i:j] = 1
            i = j

        return T

    def get_M(self) -> numpy.array:
        M = numpy.zeros((self.m, self.l))

        repair_set_index = 0
        for repair_node_id_sets in self.obj_id_to_repair_node_id_sets_map.values():
            for repair_node_id_set in repair_node_id_sets:
                for node_id in repair_node_id_set:
                    M[node_id, repair_set_index] = 1

                repair_set_index += 1

        return M
