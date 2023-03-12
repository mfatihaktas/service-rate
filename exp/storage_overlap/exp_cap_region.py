import dataclasses

from src.service_rate import (
    plot_capacity_region,
    service_rate_w_stripe,
)

from src.utils.debug import *
from src.utils.plot import *


def get_service_rate_inspector(
    d: int,
    s: int,
    overlap_size: int,
) -> service_rate_w_stripe.ServiceRateInspectorForStorageWithStripeAndParity:
    m = 2 * d - overlap_size

    obj_id_to_node_id_set_map = {
        0: set(range(d)),
        1: set(range(d - overlap_size, m)),
    }

    return service_rate_w_stripe.ServiceRateInspectorForStorageWithStripeAndParity(
        k=2,
        m=m,
        s=s,
        obj_id_to_node_id_set_map=obj_id_to_node_id_set_map,
    )


def plot_capacity_region_w_varying_overlap_size():
    @dataclasses.dataclass
    class Conf:
        d: int
        s: int

    conf_list = []
    for d in range(2, 6):
        for s in range(1, d + 1):
            conf = Conf(d=d, s=s)
            conf_list.append(conf)
    log(DEBUG, "", conf_list=conf_list)

    overlap_size_list = [1, 2, 3, 4, 5]

    num_rows = len(overlap_size_list)
    num_columns = len(conf_list)
    fig_size = (num_columns * 5, num_rows * 5)
    fig, ax_list = plot.subplots(num_rows, num_columns, figsize=fig_size)

    def plot_(column_index: int):
        conf = conf_list[column_index]
        log(DEBUG, "Started", column_index=column_index, conf=conf)

        for row_index, overlap_size in enumerate(overlap_size_list):
            if overlap_size > conf.d:
                continue

            service_rate_inspector = get_service_rate_inspector(
                d=conf.d,
                s=conf.s,
                overlap_size=overlap_size,
            )

            ax = ax_list[row_index][column_index]
            plot.sca(ax)
            plot_capacity_region.plot_capacity_region_2d_alternative_w_hull(
                service_rate_inspector=service_rate_inspector,
                obj_id_list=[0, 1],
            )

            fontsize = 14
            plot.xlabel(r"$\lambda_a$", fontsize=fontsize)
            plot.ylabel(r"$\lambda_b$", fontsize=fontsize)

            title = (
                fr"$d= {conf.d}$, "
                fr"$s= {conf.s}$, "
                r"$n_{overlap}=$" + fr" ${overlap_size}$"
            )
            plot.title(title, fontsize=fontsize)

    for i in range(num_columns):
        plot_(column_index=i)

    plot.savefig("plots/plot_capacity_region_w_varying_overlap_size.png", bbox_inches="tight")
    plot.gcf().clear()
    log(INFO, "Done")


if __name__ == "__main__":
    plot_capacity_region_w_varying_overlap_size()
