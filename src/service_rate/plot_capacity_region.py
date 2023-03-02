import numpy
import scipy

from src.service_rate import service_rate
from src.utils import plot_polygon

from src.utils.debug import *
from src.utils.plot import *


def plot_capacity_region(
    service_rate_inspector: service_rate.ServiceRateInspector,
    file_name_suffix: str,
):
    k = service_rate_inspector.k
    if k == 2:
        plot_cap_2d(
            service_rate_inspector=service_rate_inspector,
            file_name_suffix=file_name_suffix,
        )
    elif k == 3:
        plot_cap_3d(
            service_rate_inspector=service_rate_inspector,
            file_name_suffix=file_name_suffix,
        )
    else:
        log(ERROR, "Not implemented for k= {}".format(k))


# def add_popularity_heatmap():
#     ## Popularity heatmap
#     if popmodel is not None:
#       if heatmap_grid_xmax_ymax is None:
#         heatmap_grid, xmax, ymax = gen_pop_heatmap_grid_xmax_ymax(popmodel)
#       else:
#         heatmap_grid, xmax, ymax = heatmap_grid_xmax_ymax

#       plot.imshow(np.rot90(heatmap_grid), cmap=plot.cm.gist_earth_r, extent=[0, xmax, 0, ymax] )
#       covered_mass = self.integrate_overcaphyperlane(popmodel.joint_pdf)
#       plot.text(0.75, 0.9, 'Covered mass= {}'.format(covered_mass),
#         horizontalalignment='center', verticalalignment='center', transform = plot.gca().transAxes)


def plot_capacity_region_2d(
    service_rate_inspector: service_rate.ServiceRateInspector,
    file_name_suffix: str = None,
):
    log(DEBUG, "Started", service_rate_inspector=service_rate_inspector)

    check(
        service_rate_inspector.compute_halfspace_intersections,
        "To plot capacity region, `compute_halfspace_intersections` should have been set",
    )

    hull = service_rate_inspector.hull
    points = service_rate_inspector.boundary_points_in_rows
    # # log(INFO, "", halfspaces_intersections=service_rate_inspector.halfspaces.intersections)
    # x_l, y_l = [], []
    # for x in service_rate_inspector.halfspaces.intersections:
    #     y = numpy.matmul(service_rate_inspector.T, x)
    #     x_l.append(y[0])
    #     y_l.append(y[1])
    # # log(DEBUG, "", x_l=x_l, y_l=y_l)
    # log(DEBUG, "Will call scipy.spatial.ConvexHull", len_x_l=len(x_l), len_y_l=len(y_l))
    # points = numpy.column_stack((x_l, y_l))
    # hull = scipy.spatial.ConvexHull(points)

    # for simplex in hull.simplices:
    #     simplex = numpy.append(
    #         simplex, simplex[0]
    #     )  # https://stackoverflow.com/questions/27270477/3d-convex-hull-from-point-cloud
    #     plot.plot(points[simplex, 0], points[simplex, 1], c=NICE_BLUE, marker="o")
    plot.fill(
        points[hull.vertices, 0], points[hull.vertices, 1], c=NICE_BLUE, alpha=0.5
    )

    if file_name_suffix is None:
        return

    fontsize = 24

    # Oleg plot config
    ax = plot.gca()
    yticks = [0, 0.5, 1, 1.5, 2, 2.5]
    xticks = [0.5, 1, 1.5, 2, 2.5]
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xlim([0, 2.5])
    ax.set_ylim([0, 2.5])
    plot.xticks(fontsize=fontsize)
    plot.yticks(fontsize=fontsize)

    plot.xlabel(r"$\lambda_a$", fontsize=fontsize)
    # plot.xlim(xmin=0)
    plot.ylabel(r"$\lambda_b$", fontsize=fontsize)
    # plot.ylim(ymin=0)

    # title = (
    #     r"$k= {}$, $m= {}$, $C= {}$, ".format(
    #         service_rate_inspector.k, service_rate_inspector.m, service_rate_inspector.C
    #     )
    #     + "Volume= {0:.2f} \n".format(hull.volume)
    #     + service_rate_inspector.to_sysrepr()
    # )
    # plot.title(title, fontsize=fontsize, y=1.05)
    plot.gcf().set_size_inches(4, 3)
    plot.savefig(f"plots/plot_capacity_region_{file_name_suffix}.png", bbox_inches="tight")
    plot.gcf().clear()
    log(INFO, "Done.")


def plot_capacity_region_2d_alternative(
    service_rate_inspector: service_rate.ServiceRateInspector,
    file_name_suffix: str = None,
):
    log(DEBUG, "Started", service_rate_inspector=service_rate_inspector)


    plot.fill(
        points[hull.vertices, 0], points[hull.vertices, 1], c=NICE_BLUE, alpha=0.5
    )

    fontsize = 24
    plot.xlabel(r"$\lambda_a$", fontsize=fontsize)
    # plot.xlim(xmin=0)
    plot.ylabel(r"$\lambda_b$", fontsize=fontsize)
    # plot.ylim(ymin=0)

    # title = (
    #     r"$k= {}$, $m= {}$, $C= {}$, ".format(
    #         service_rate_inspector.k, service_rate_inspector.m, service_rate_inspector.C
    #     )
    #     + "Volume= {0:.2f} \n".format(hull.volume)
    #     + service_rate_inspector.to_sysrepr()
    # )
    # plot.title(title, fontsize=fontsize, y=1.05)
    plot.gcf().set_size_inches(4, 3)
    plot.savefig(f"plots/plot_capacity_region_{file_name_suffix}.png", bbox_inches="tight")
    plot.gcf().clear()
    log(INFO, "Done.")


def plot_capacity_region_2d_when_k_g_2(
    service_rate_inspector: service_rate.ServiceRateInspector,
    file_name_suffix: str,
):
    fontsize = 18

    xi_yi_l = list(itertools.combinations(range(service_rate_inspector.k), 2))
    fig, axs = plot.subplots(len(xi_yi_l), 1, sharex="col")
    figsize = [4, len(xi_yi_l) * 4]
    for i, (xi, yi) in enumerate(xi_yi_l):
        print(">> i= {}, (xi, yi)= ({}, {})".format(i, xi, yi))
        x_l, y_l = [0], [0]
        for _p in service_rate_inspector.hs.intersections:
            p = numpy.matmul(service_rate_inspector.T, _p)
            include = True
            for j in range(service_rate_inspector.k):
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
            plot.plot(points[simplex, 0], points[simplex, 1], c=NICE_BLUE, marker="o")
        plot.xlim(xmin=0)
        plot.ylim(ymin=0)
        plot.xlabel("{}".format(chr(ord("a") + xi)), fontsize=fontsize)
        plot.ylabel("{}".format(chr(ord("a") + yi)), fontsize=fontsize)
        plot.title("i= {}".format(i))

    suptitle = (
        r"$k= {}$, $n= {}$, $C= {}$ \n".format(
            service_rate_inspector.k, service_rate_inspector.m, service_rate_inspector.C
        )
        + service_rate_inspector.to_sysrepr()
    )
    plot.suptitle(suptitle, fontsize=fontsize)  # , y=1.05
    fig.set_size_inches(figsize[0], figsize[1])
    plot.savefig("plots/plot_cap_2d_when_k_g_2.png", bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")


def plot_capacity_region_3d(
    service_rate_inspector: service_rate.ServiceRateInspector,
    file_name_suffix: str,
):
    ax = plot.axes(projection="3d")

    x_l, y_l, z_l = [], [], []
    for x in service_rate_inspector.hs.intersections:
        y = numpy.matmul(service_rate_inspector.T, x)
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
    plot.title(
        r"$k= {}$, $m= {}$, $C= {}$".format(
            service_rate_inspector.k, service_rate_inspector.m, service_rate_inspector.C
        )
        + ", Volume= {0:.2f}".format(hull.volume)
        + f"\n{service_rate_inspector.to_sysrepr()}",
        fontsize=fontsize,
        y=1.05,
    )

    plot.title(r"$k= {}$, $n= {}$".format(self.k, self.m), fontsize=fontsize)
    plot.gcf().set_size_inches(7, 5)
    plot.savefig(f"plots/plot_cap_3d_{file_name_suffix}.png", bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")
