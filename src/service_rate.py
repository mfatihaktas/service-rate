import collections
import cvxpy
import itertools
import mpl_toolkits
import numpy
import scipy.spatial
import string

from src import plot_polygon, service_rate_utils

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
        obj_to_node_id_map: dict,
        compute_halfspace_intersections: bool = False,
        max_repair_set_size: int = None,
        redundancy_w_two_xors: bool = False,
    ):
        ## Number of buckets
        self.m = m
        ## Capacity of each node
        self.C = C
        ## Object composition matrix
        self.G = G
        ## Map from object id's to node id's
        self.obj_to_node_id_map = obj_to_node_id_map
        self.compute_halfspace_intersections = compute_halfspace_intersections

        numpy.set_printoptions(threshold=sys.maxsize)

        self.k = G.shape[0]
        self.n = G.shape[1]

        ## Note: Repair sets are given in terms of obj ids,
        ## in the sense of columns of G or keys of obj_to_node_id_map

        if redundancy_w_two_xors:
            self.obj_to_repair_sets_map = service_rate_utils.get_obj_to_repair_sets_map_for_redundancy_w_two_xors(
                n=self.n, G=self.G,
            )

        else:
            if max_repair_set_size is None:
                max_repair_set_size = self.k
            log(DEBUG, "", max_repair_set_size=max_repair_set_size)

            # self.obj_to_repair_sets_map = service_rate_utils.get_obj_to_repair_sets_map(
            #     k=self.k, n=self.n, G=self.G, max_repair_set_size=max_repair_set_size,
            # )
            self.obj_to_repair_sets_map = service_rate_utils.get_obj_to_repair_sets_map_w_joblib(
                k=self.k, n=self.n, G=self.G, max_repair_set_size=max_repair_set_size,
            )

        log(DEBUG, "", obj_to_repair_sets_map=self.obj_to_repair_sets_map)

        ## Repair set list
        repair_set_list = []
        for obj in range(self.k):
            repair_set_list.extend(self.obj_to_repair_sets_map[obj])

        self.l = len(repair_set_list)

        ## T
        self.T = service_rate_utils.get_T(
            num_objects=self.k,
            num_repair_sets=self.l,
            obj_to_repair_set_size_map={
                obj: len(repair_set)
                for obj, repair_set in self.obj_to_repair_sets_map.items()
            }
        )
        log(DEBUG, f"T= \n{self.T}")

        ## M
        self.M = service_rate_utils.get_M(
            num_objects=self.n,
            num_nodes=self.m,
            repair_set_list=repair_set_list,
            obj_to_node_id_map=self.obj_to_node_id_map,
        )
        log(DEBUG, f"M= \n{self.M}")

        ## Halfspaces
        if compute_halfspace_intersections:
            self.halfspaces = service_rate_utils.get_halfspaces(
                num_nodes=self.m,
                num_repair_sets=self.l,
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
            f"\t G=\n {self.G} \n"
            # f"\t obj_to_node_id_map= {self.obj_to_node_id_map} \n"
            f"\t M= {self.M} \n"
            f"\t T= {self.T} \n"
            ")"
        )

    def to_sysrepr(self):
        sym_list = string.ascii_lowercase[: self.k]
        node_list = [[] for _ in range(self.m)]
        for obj in range(self.n):
            ni = self.obj_to_node_id_map[obj]
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
        try:
            prob.solve()
        except cvxpy.SolverError:
            prob.solve(solver="SCS")

        # log(DEBUG, f"prob.status= {prob.status}")
        # blog(x_val=x.value)
        return prob.status == "optimal"

    def min_cost(self, obj_demand_list: list[float]):
        # log(DEBUG, "Started;", obj_demand_list=obj_demand_list)

        demand_vector = numpy.array(obj_demand_list).reshape((self.k, 1))

        x = cvxpy.Variable(shape=(self.l, 1), name="x")

        cost_coeff_list = []
        for obj in range(self.k):
            repair_set_list = self.obj_to_repair_sets_map[obj]
            for repair_set in repair_set_list:
                cost_coeff_list.append(len(repair_set))
        log(DEBUG, "", cost_coeff_list=cost_coeff_list)

        cost_coeff_row_vector = numpy.asarray(cost_coeff_list).reshape((1, self.l))
        obj = cvxpy.Minimize(cost_coeff_row_vector @ x)
        constraints = [self.M @ x <= self.C, x >= 0, self.T @ x == demand_vector]

        prob = cvxpy.Problem(obj, constraints)
        try:
            prob.solve()
        except cvxpy.SolverError:
            prob.solve(solver="SCS")

        if prob.status != "optimal":
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

        # log(DEBUG, f"prob.status= {prob.status}")
        if prob.status == cvxpy.OPTIMAL:
            cost = cost_coeff_row_vector @ x.value
            cost = cost.tolist()[0][0]
            # cost = cost / numpy.sum(x.value)
            return cost
        else:
            return None

    def min_distance_to_boundary(self, obj_demand_list: list[float]) -> float:
        """ Returns the min distance from obj_demand_list to the service rate boundary.

        Relies on
        self.halfspaces = scipy.spatial.HalfspaceIntersection(halfspaces, feasible_point)
        self.hull = scipy.spatial.ConvexHull(self.boundary_points_in_rows)
        which take very long to complete when the number of objects > ~30.

        TODO: Investigate why these two functions takes so long to complete,
        and if we can make them run faster, or if we can find a different faster
        way to get the same results.
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

        u = numpy.array(obj_demand_list)
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

    def plot_cap(self):
        if self.k == 2:
            self.plot_cap_2d()
        elif self.k == 3:
            self.plot_cap_3d()
        else:
            log(ERROR, "Not implemented for k= {}".format(self.k))

    def plot_cap_2d(self):
        check(self.compute_halfspace_intersections,
              "To plot capacity region, `compute_halfspace_intersections` should have been set")

        # print("hs.intersections= \n{}".format(self.hs.intersections) )
        x_l, y_l = [], []
        for x in self.halfspaces.intersections:
            y = numpy.matmul(self.T, x)
            x_l.append(y[0])
            y_l.append(y[1])

        points = numpy.column_stack((x_l, y_l))
        hull = scipy.spatial.ConvexHull(points)
        for simplex in hull.simplices:
            simplex = numpy.append(
                simplex, simplex[0]
            )  # https://stackoverflow.com/questions/27270477/3d-convex-hull-from-point-cloud
            plot.plot(points[simplex, 0], points[simplex, 1], c=NICE_BLUE, marker="o")

        plot.legend()
        fontsize = 18
        plot.xlabel(r"$\rho_a$", fontsize=fontsize)
        plot.xlim(xmin=0)
        plot.ylabel(r"$\rho_b$", fontsize=fontsize)
        plot.ylim(ymin=0)
        title = (
            r"$k= {}$, $m= {}$, $C= {}$, ".format(self.k, self.m, self.C)
            + "Volume= {0:.2f} \n".format(hull.volume)
            + self.to_sysrepr()
        )
        plot.title(title, fontsize=fontsize, y=1.05)
        plot.savefig("plot_cap_2d.png", bbox_inches="tight")
        plot.gcf().clear()
        log(INFO, "Done.")

    def plot_cap_2d_when_k_g_2(self):
        fontsize = 18

        xi_yi_l = list(itertools.combinations(range(self.k), 2))
        fig, axs = plot.subplots(len(xi_yi_l), 1, sharex="col")
        figsize = [4, len(xi_yi_l) * 4]
        for i, (xi, yi) in enumerate(xi_yi_l):
            print(">> i= {}, (xi, yi)= ({}, {})".format(i, xi, yi))
            x_l, y_l = [0], [0]
            for _p in self.hs.intersections:
                p = numpy.matmul(self.T, _p)
                include = True
                for j in range(self.k):
                    if j == xi or j == yi:
                        continue
                    if p[j] > 0.01:
                        include = False
                        break
                if include:
                    x_l.append(p[xi])
                    y_l.append(p[yi])

            ax = axs[i]
            plot.sca(ax)

            # print("x_l= {}".format(x_l) )
            # print("y_l= {}".format(y_l) )
            # print("Right before plotting red cross; i= {}".format(i) )
            # plot.plot([1], [1], c='red', marker='x', ms=12)
            # print("Right after plotting red cross; i= {}".format(i) )
            # plot.plot(x_l, y_l, c=NICE_BLUE, marker='o', ls='None')

            points = numpy.column_stack((x_l, y_l))
            # print("points= {}".format(points) )
            hull = scipy.spatial.ConvexHull(points)
            for simplex in hull.simplices:
                simplex = numpy.append(
                    simplex, simplex[0]
                )  # https://stackoverflow.com/questions/27270477/3d-convex-hull-from-point-cloud
                plot.plot(
                    points[simplex, 0], points[simplex, 1], c=NICE_BLUE, marker="o"
                )
            plot.xlim(xmin=0)
            plot.ylim(ymin=0)
            plot.xlabel("{}".format(chr(ord("a") + xi)), fontsize=fontsize)
            plot.ylabel("{}".format(chr(ord("a") + yi)), fontsize=fontsize)
            plot.title("i= {}".format(i))

        suptitle = (
            r"$k= {}$, $n= {}$, $C= {}$ \n".format(self.k, self.m, self.C)
            + self.to_sysrepr()
        )
        plot.suptitle(suptitle, fontsize=fontsize)  # , y=1.05
        fig.set_size_inches(figsize[0], figsize[1])
        plot.savefig("plot_cap_2d_when_k_g_2.png", bbox_inches="tight")
        plot.gcf().clear()
        log(INFO, "done.")

    def plot_cap_3d(self):
        ax = plot.axes(projection="3d")

        x_l, y_l, z_l = [], [], []
        for x in self.hs.intersections:
            y = numpy.matmul(self.T, x)
            x_l.append(y[0])
            y_l.append(y[1])
            z_l.append(y[2])

        points = numpy.column_stack((x_l, y_l, z_l))
        hull = scipy.spatial.ConvexHull(points)
        for simplex in hull.simplices:
            simplex = numpy.append(
                simplex, simplex[0]
            )  # https://stackoverflow.com/questions/27270477/3d-convex-hull-from-point-cloud
            ax.plot3D(
                points[simplex, 0],
                points[simplex, 1],
                points[simplex, 2],
                c=NICE_BLUE,
                marker="o",
            )

        faces = hull.simplices
        verts = points
        triangles = []
        for s in faces:
            sq = [
                (verts[s[0], 0], verts[s[0], 1], verts[s[0], 2]),
                (verts[s[1], 0], verts[s[1], 1], verts[s[1], 2]),
                (verts[s[2], 0], verts[s[2], 1], verts[s[2], 2]),
            ]
            triangles.append(sq)

        new_faces = plot_polygon.simplify(triangles)
        for sq in new_faces:
            f = mpl_toolkits.mplot3d.art3d.Poly3DCollection([sq])
            # f.set_color(matplotlib.colors.rgb2hex(scipy.rand(20) ) )
            f.set_color(next(dark_color_c))
            f.set_edgecolor("k")
            f.set_alpha(0.15)  # 0.2
            ax.add_collection3d(f)

        plot.legend()
        fontsize = 18
        ax.set_xlabel(r"$\rho_a$", fontsize=fontsize)
        ax.set_xlim(xmin=0)
        ax.set_ylabel(r"$\rho_b$", fontsize=fontsize)
        ax.set_ylim(ymin=0)
        ax.set_zlabel(r"$\rho_c$", fontsize=fontsize)
        ax.set_zlim(zmin=0)
        ax.view_init(20, 30)
        # plot.title(r'$k= {}$, $m= {}$, $C= {}$'.format(self.k, self.m, self.C) + ', Volume= {0:.2f}'.format(hull.volume) \
        #   + '\n{}'.format(self.to_sysrepr() ), fontsize=fontsize, y=1.05)
        plot.title(r"$k= {}$, $n= {}$".format(self.k, self.m), fontsize=fontsize)
        plot.gcf().set_size_inches(7, 5)
        plot.savefig(
            "plot_cap_3d_{}.png".format(self.to_sysrepr()), bbox_inches="tight"
        )
        plot.gcf().clear()
        log(INFO, "done.")
