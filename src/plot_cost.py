import numpy

from src import service_rate
from src.debug_utils import *
from src.plot_utils import *


def plot_cost_2d(
    service_rate_inspector: service_rate.ServiceRateInspector,
    file_name_suffix: str,
):
    check(
        service_rate_inspector.k == 2,
        "Defined only for k = 2",
        service_rate_inspector_k=service_rate_inspector.k,
    )

    x_max = max([p[0] for p in service_rate_inspector.boundary_point_list])
    y_max = max([p[1] for p in service_rate_inspector.boundary_point_list])
    X, Y = numpy.mgrid[0:x_max:10j, 0:y_max:10j]
    log(DEBUG, "", X=X, Y=Y, X_shape=X.shape, Y_shape=Y.shape)

    x_l, y_l = X.ravel(), Y.ravel()
    cost_list = [
        service_rate_inspector.min_cost(obj_demand_list=[x, y])
        for x, y in zip(x_l, y_l)
    ]

    cost_matrix = numpy.array(cost_list).reshape(X.shape)
    log(DEBUG, "", cost_matrix=cost_matrix)

    fig = plot.gcf()
    # ax = plot.axes(projection='3d')
    ax = plot.gca()

    cost_matrix = cost_matrix.astype(float)
    cost_matrix = numpy.ma.masked_where(cost_matrix is None, cost_matrix)
    log(DEBUG, "After masking", cost_matrix=cost_matrix)

    color_map = plot.cm.Reds
    color_map.set_bad(color="white")
    # ax.plot_surface(x_l, y_l, cost_list, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    # ax.scatter(x_l, y_l, cost_list, c=cost_list, cmap='viridis', lw=0.5)
    c = ax.pcolormesh(X, Y, cost_matrix, cmap=color_map)  # , vmin=0, vmax=1.75)
    fig.colorbar(c, ax=ax)

    # Service cap
    for simplex in service_rate_inspector.hull.simplices:
        plot.plot(
            service_rate_inspector.boundary_points_in_rows[simplex, 0],
            service_rate_inspector.boundary_points_in_rows[simplex, 1],
            c=NICE_BLUE,
            marker="o",
            ls="-",
            lw=3,
        )

    plot.xlim((0, 2.5))
    plot.ylim((0, 2.5))
    # prettify(plot.gca() )
    # ax.set_title('{}'.format(service_rate_inspector.to_sysrepr() ) )
    ax.set_xlabel(r"$\lambda_a$", fontsize=24)
    ax.set_ylabel(r"$\lambda_b$", fontsize=24)

    fig = plot.gcf()
    fig.set_size_inches(5, 3.5)
    plot.savefig(
        f"plot_cost_2d_{file_name_suffix}.png", bbox_inches="tight"
    )  # , bbox_extra_artists=[ax],
    fig.clear()

    log(INFO, "Done")
