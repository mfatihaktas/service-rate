import copy
import dataclasses

from src.service_rate import (
    plot_capacity_region,
    service_rate,
    storage_scheme as storage_scheme_module
)

from src.utils.debug import *
from src.utils.misc import *
from src.utils.plot import *


def get_node_id_to_objs_list_for_sys_and_mds(
    k: int,
    num_sys: int,
    num_mds: int,
    field_size: int,
) -> list[list[storage_scheme_module.Obj]]:
    node_id_to_objs_list = []

    sys_obj_cycle = itertools.cycle(
        [
            storage_scheme_module.PlainObj(id_str=f"{get_char(obj_id)}")
            for obj_id in range(k)
        ]
    )

    for _ in range(num_sys):
        obj = copy.copy(next(sys_obj_cycle))
        node_id_to_objs_list.append([obj])

    mds_obj_cycle = itertools.cycle(
        [
            storage_scheme_module.CodedObj(
                coeff_obj_list=[
                    ((obj_id + 1)**i, storage_scheme_module.PlainObj(id_str=f"{get_char(obj_id)}"))
                    for obj_id in range(k)
                ]
            )
            for i in range(field_size)
        ]
    )

    for _ in range(num_mds):
        obj = copy.copy(next(mds_obj_cycle))
        node_id_to_objs_list.append([obj])

    return node_id_to_objs_list


def plot_capacity_region_for_sys_and_mds_w_field_size(
    k: int,
    num_nodes: int,
):
    log(DEBUG, "Started", k=k, num_nodes=num_nodes)

    @dataclasses.dataclass
    class Conf:
        num_sys: int
        num_mds: int

    conf_list = []
    for num_sys in range(k, num_nodes, k):
        num_mds = num_nodes - num_sys
        conf = Conf(num_sys=num_sys, num_mds=num_mds)
        conf_list.append(conf)
    log(DEBUG, "", conf_list=conf_list)

    field_size_list = list(range(1, k + 2))

    num_rows = len(field_size_list)
    num_columns = len(conf_list)
    fig_size = (num_columns * 5, num_rows * 5)
    fig, ax_matrix = plot.subplots(num_rows, num_columns, figsize=fig_size)

    def plot_(plot_index: int):
        conf = conf_list[plot_index]
        log(DEBUG, "Started", plot_index=plot_index, conf=conf)

        for row_index, field_size in enumerate(field_size_list):
            node_id_to_objs_list = get_node_id_to_objs_list_for_sys_and_mds(
                k=k,
                num_sys=conf.num_sys,
                num_mds=conf.num_mds,
                field_size=field_size,
            )

            storage_scheme = storage_scheme_module.StorageScheme(node_id_to_objs_list)
            log(DEBUG, "", storage_scheme=storage_scheme)

            service_rate_inspector = service_rate.ServiceRateInspector(
                m=len(node_id_to_objs_list),
                C=1,
                G=storage_scheme.obj_encoding_matrix,
                obj_id_to_node_id_map=storage_scheme.obj_id_to_node_id_map,
                max_repair_set_size=k,
            )

            ax = ax_matrix[row_index][plot_index]
            plot.sca(ax)
            plot_capacity_region.plot_capacity_region_2d_alternative(
                service_rate_inspector=service_rate_inspector,
                obj_id_list=[0, 1],
            )

            fontsize = 14
            plot.xlabel(r"$\lambda_a$", fontsize=fontsize)
            plot.ylabel(r"$\lambda_b$", fontsize=fontsize)

            title = (
                r"$n_{sys}= $" + fr"${conf.num_sys}$, "
                r"$n_{mds}= $" + fr"${conf.num_mds}$, "
                # r"$n_{\textrm{mds}}= $" + fr"${conf.num_mds}$, "
                fr"$q= {field_size}$"
            )
            plot.title(title, fontsize=fontsize)

    for i in range(num_columns):
        plot_(plot_index=i)

    plot.savefig(f"plots/plot_capacity_region_for_a_b_mds_w_field_size_num_nodes_{num_nodes}.png", bbox_inches="tight")
    plot.gcf().clear()
    log(INFO, "Done")


def test_service_rate_inspector():
    node_id_to_objs_list = get_node_id_to_objs_list_for_sys_and_mds(
        num_a=1,
        num_b=1,
        num_mds=4,
        field_size=2,
    )

    storage_scheme = storage_scheme_module.StorageScheme(node_id_to_objs_list)
    log(DEBUG, "", storage_scheme=storage_scheme, G=storage_scheme.obj_encoding_matrix)

    service_rate_inspector = service_rate.ServiceRateInspector(
        m=len(node_id_to_objs_list),
        C=1,
        G=storage_scheme.obj_encoding_matrix,
        obj_id_to_node_id_map=storage_scheme.obj_id_to_node_id_map,
        max_repair_set_size=2,
    )


if __name__ == "__main__":
    # test_service_rate_inspector()
    plot_capacity_region_for_sys_and_mds_w_field_size(k=3, num_nodes=20)
