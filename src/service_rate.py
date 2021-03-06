import collections
import cvxpy
import itertools
import mpl_toolkits
import numpy
import scipy.spatial
import string

from typing import Union

from src import service_rate_utils

from src.debug_utils import *
from src.plot_utils import *


class ServiceRateInspector:
    """Defines a storage system that distributes n objects across m
    nodes. Input arguments and class variables are described in the README.
    """

    def __init__(
        self,
        m: int,
        C: float,
        G: numpy.ndarray,
        obj_id_to_node_id_map: dict,
        redundancy_w_two_xors: bool = True,
        compute_halfspace_intersections: bool = False,
        max_repair_set_size: int = None,
    ):
        ## Number of buckets
        self.m = m
        ## Capacity of each node
        self.C = C
        ## Object composition matrix
        self.G = G
        ## Map from object id's to node id's
        self.obj_id_to_node_id_map = obj_id_to_node_id_map
        self.compute_halfspace_intersections = compute_halfspace_intersections

        numpy.set_printoptions(threshold=sys.maxsize)

        # log(DEBUG, f"G= \n{self.G}")

        self.k = G.shape[0]
        self.n = G.shape[1]

        ## Note: Repair sets are given in terms of obj ids,
        ## in the sense of columns of G or keys of obj_id_to_node_id_map

        if redundancy_w_two_xors:
            self.orig_obj_id_to_repair_sets_w_obj_ids_map = service_rate_utils.get_orig_obj_id_to_repair_sets_w_obj_ids_map_for_redundancy_w_two_xors(
                n=self.n, G=self.G,
            )

        else:
            if max_repair_set_size is None:
                max_repair_set_size = self.k
            # log(DEBUG, "", max_repair_set_size=max_repair_set_size)

            # self.orig_obj_id_to_repair_sets_w_obj_ids_map = service_rate_utils.get_orig_obj_id_to_repair_sets_w_obj_ids_map(
            #     k=self.k, n=self.n, G=self.G, max_repair_set_size=max_repair_set_size,
            # )
            self.orig_obj_id_to_repair_sets_w_obj_ids_map = service_rate_utils.get_orig_obj_id_to_repair_sets_w_obj_ids_map_w_joblib(
                k=self.k, n=self.n, G=self.G, max_repair_set_size=max_repair_set_size,
            )

        # log(DEBUG, "", orig_obj_id_to_repair_sets_w_obj_ids_map=self.orig_obj_id_to_repair_sets_w_obj_ids_map)

        self.orig_obj_id_to_repair_sets_w_node_ids_map = service_rate_utils.get_orig_obj_id_to_repair_sets_w_node_ids_map(
            orig_obj_id_to_repair_sets_w_obj_ids_map=self.orig_obj_id_to_repair_sets_w_obj_ids_map,
            obj_id_to_node_id_map=self.obj_id_to_node_id_map,
        )

        ## Repair set list
        repair_set_w_obj_ids_list = []
        for obj in range(self.k):
            repair_set_w_obj_ids_list.extend(self.orig_obj_id_to_repair_sets_w_obj_ids_map[obj])

        self.l = len(repair_set_w_obj_ids_list)

        ## T
        self.T = service_rate_utils.get_T(
            num_orig_objects=self.k,
            total_num_repair_sets=self.l,
            orig_obj_to_repair_set_w_obj_ids_size_map={
                obj: len(repair_set)
                for obj, repair_set in self.orig_obj_id_to_repair_sets_w_obj_ids_map.items()
            }
        )
        # log(DEBUG, f"T= \n{self.T}")

        ## M
        self.M = service_rate_utils.get_M(
            num_objects=self.n,
            num_nodes=self.m,
            repair_set_w_obj_ids_list=repair_set_w_obj_ids_list,
            obj_id_to_node_id_map=self.obj_id_to_node_id_map,
        )
        # log(DEBUG, f"M= \n{self.M}")

        ## Halfspaces
        if compute_halfspace_intersections:
            self.halfspaces = service_rate_utils.get_halfspaces(
                num_nodes=self.m,
                total_num_repair_sets=self.l,
                node_capacity=self.C,
                M=self.M,
            )

            self.boundary_point_list = [list(numpy.matmul(self.T, p)) for p in self.halfspaces.intersections]
            self.boundary_points_in_rows = numpy.array(self.boundary_point_list).reshape((len(self.boundary_point_list), self.k))

            self.hull = scipy.spatial.ConvexHull(self.boundary_points_in_rows)
            log(DEBUG, "scipy.spatial.ConvexHull is done.")

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

    def is_in_cap_region(self, obj_demand_list: list[float]) -> bool:
        demand_vector = numpy.array(obj_demand_list).reshape((self.k, 1))

        x = cvxpy.Variable(shape=(self.l, 1), name="x")

        # obj = cvxpy.Maximize(numpy.ones((1, self.l))*x)
        obj = cvxpy.Maximize(0)
        constraints = [self.M @ x <= self.C, x >= 0, self.T @ x == demand_vector]

        prob = cvxpy.Problem(obj, constraints)
        opt_value = service_rate_utils.solve_prob(prob)

        return opt_value is not None

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
            log(WARNING,
                "Object demand vector is not in the capacity region, "
                "thus cannot compute the cost.",
                prob_status=prob.status,
                obj_demand_list=obj_demand_list,
            )
            return None

        allocation_list = []
        for v_list in x.value.tolist():
            v = v_list[0]
            allocation_list.append(
                v if abs(v) > 0.001 else 0
            )
        log(DEBUG, f"allocation_list= {allocation_list}")

        cost = cost_coeff_row_vector @ x.value
        cost = cost.tolist()[0][0]
        # cost = cost / numpy.sum(x.value)
        return cost

    def min_distance_to_boundary_w_cvxpy(self, obj_demand_list: list[float]) -> float:
        """ Returns the min distance from obj_demand_list to the service rate boundary.
        """

        demand_vector = numpy.array(obj_demand_list).reshape((self.k, 1))
        x = cvxpy.Variable(shape=(self.l, 1), name="x")

        obj = cvxpy.Minimize(cvxpy.sum_squares(self.T @ x - demand_vector))
        in_cap_region = self.is_in_cap_region(numpy.array(obj_demand_list))
        if in_cap_region:
            constraints = [self.M @ x >= self.C, x >= 0]
        else:
            constraints = [self.M @ x <= self.C, x >= 0]

        prob = cvxpy.Problem(obj, constraints)
        opt_value = service_rate_utils.solve_prob(prob)
        if opt_value is None:
            return None

        # blog(x_val=x.value)
        point_closest_to_demand_vector = numpy.matmul(self.T, x.value)

        distance = numpy.sqrt(numpy.sum((point_closest_to_demand_vector - demand_vector)**2))
        return distance

    def min_distance_to_boundary_w_cvxpy_for_oleg(self, obj_demand_list: list[Union[float, str]]) -> float:
        """Wrapper around `min_distance_to_boundary_w_cvxpy()` to accomodate the input
        and output requirements needed by Oleg's use case.
        """

        obj_demand_list = [float(i) for i in obj_demand_list]

        in_cap_region = self.is_in_cap_region(numpy.array(obj_demand_list))
        min_distance = self.min_distance_to_boundary_w_cvxpy(obj_demand_list)

        return (
            -min_distance
            if in_cap_region
            else
            min_distance
        )

    def min_distance_to_boundary_w_convex_hull(self, obj_demand_list: list[float]) -> float:
        """ Returns the min distance from obj_demand_list to the service rate boundary.

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
            log(WARNING,
                "Cannot compute min distance to service capacity boundary",
                compute_halfspace_intersection=self.compute_halfspace_intersections
            )
            return None

        # https://stackoverflow.com/questions/42248202/find-the-projection-of-a-point-on-the-convex-hull-with-scipy
        def min_distance(p, v1, v2):
          """ Finds projection p' of point p on the closest edge formed by points p1 and p2,
          and returns the distance from p to p'.
          """

          # blog(p=p, v1=v1, v2=v2)
          # v2 = v2 - v1
          l = numpy.sum((v2 - v1)**2) # compute the squared distance between the 2 vertices
          # blog(l=l, dot=numpy.dot(p-v1, v2-v1)[0] )
          t = numpy.max([0., numpy.min([1., numpy.dot(p - v1, v2 - v1) / l])]) # numpy.min([1., numpy.dot(p-v1, v2-v1)[0]/l] )
          # blog(dot=numpy.dot(p-v1, v2-v1), t=t)
          proj = v1 + t*(v2 - v1)
          return numpy.sqrt(numpy.sum((proj - p)**2))

        x = numpy.array(obj_demand_list).reshape((self.k, 1))
        min_dist = float('Inf')
        for i in range(len(self.hull.vertices)):
          m = min_distance(
              x.T,
              self.boundary_points_in_rows[self.hull.vertices[i]],
              self.boundary_points_in_rows[self.hull.vertices[(i+1) % len(self.hull.vertices)]]
          )
          if m < min_dist:
            min_dist = m

        return min_dist

    def approx_min_distance_to_boundary(self, obj_demand_list: list[float]) -> float:
        """ Returns the approximate min distance from obj_demand_list to the service rate boundary.

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
        return (
            opt_value / self.C
            if opt_value
            else
            None
        )

    def load_across_nodes(self, obj_demand_list: list[float]) -> list[float]:
        """Returns the load at the nodes after the object demand is distributed across
        the nodes in a way that minimizes the maximal load.
        """

        demand_vector = numpy.array(obj_demand_list).reshape((self.k, 1))
        x = cvxpy.Variable(shape=(self.l, 1), name="x")

        obj = cvxpy.Minimize(cvxpy.max(self.M @ x))
        constraints = [x >= 0, self.T @ x == demand_vector]

        prob = cvxpy.Problem(obj, constraints)
        opt_value = service_rate_utils.solve_prob(prob)

        # blog(x_val=x.value)
        load_across_nodes = numpy.matmul(self.M, x.value)

        return load_across_nodes / self.C

    def load_across_nodes_when_obj_demands_distributed_evenly_across_repair_sets(self, obj_demand_list: list[float]) -> list[float]:
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

    def load_across_nodes_when_obj_demands_replicated_to_repair_sets(self, obj_demand_list: list[float]) -> list[float]:
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
