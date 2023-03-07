"""Incomplete. Just copied and left it as it is.
"""


import copy
import dataclasses

from src.service_rate import (
    plot_capacity_region,
    service_rate,
    storage_scheme as storage_scheme_module
)

from src.utils.debug import *
from src.utils.plot import *


def get_node_id_to_objs_list_for_a_b_mds(
    num_a: int,
    num_b: int,
    num_mds: int,
    field_size: int,
) -> list[list[storage_scheme_module.Obj]]:
    node_id_to_objs_list = []

    for _ in range(num_a):
        node_id_to_objs_list.append([storage_scheme_module.PlainObj(id_str="a")])

    for _ in range(num_b):
        node_id_to_objs_list.append([storage_scheme_module.PlainObj(id_str="b")])

    obj_cycle = itertools.cycle(
        [
            storage_scheme_module.CodedObj(
                coeff_obj_list=[
                    (1, storage_scheme_module.PlainObj(id_str="a")),
                    (i + 1, storage_scheme_module.PlainObj(id_str="b")),
                ]
            )
            for i in range(field_size)
        ]
    )

    for _ in range(num_mds):
        obj = copy.copy(next(obj_cycle))
        node_id_to_objs_list.append([obj])

    # # Unbalanced MDS nodes
    # obj = copy.copy(next(obj_cycle))
    # for _ in range(num_mds // 2):
    #     node_id_to_objs_list.append([obj])

    # for _ in range(num_mds // 2):
    #     obj = copy.copy(next(obj_cycle))
    #     node_id_to_objs_list.append([obj])

    return node_id_to_objs_list


def plot_capacity_region_for_a_b_mds_w_field_size(
    num_nodes: int,
):
    log(DEBUG, "Started", num_nodes=num_nodes)

    @dataclasses.dataclass
    class Conf:
        num_a: int
        num_b: int
        num_mds: int

    conf_list = []
    # for num_a in range(1, num_nodes - 1):
    #     for num_b in range(1, num_nodes - num_a):
    #         num_mds = num_nodes - num_a - num_b
    #         conf = Conf(num_a=num_a, num_b=num_b, num_mds=num_mds)
    #         conf_list.append(conf)
    for num_sys in range(2, num_nodes, 2):
        num_mds = num_nodes - num_sys
        conf = Conf(num_a=num_sys // 2, num_b=num_sys // 2, num_mds=num_mds)
        conf_list.append(conf)
    log(DEBUG, "", conf_list=conf_list)

    field_size_list = [2, 3]

    num_rows = len(field_size_list)
    num_columns = len(conf_list)
    fig_size = (num_columns * 5, num_rows * 5)
    fig, ax_matrix = plot.subplots(num_rows, num_columns, figsize=fig_size)

    def plot_(plot_index: int):
        conf = conf_list[plot_index]
        log(DEBUG, "Started", plot_index=plot_index, conf=conf)

        for row_index, field_size in enumerate(field_size_list):
            node_id_to_objs_list = get_node_id_to_objs_list_for_a_b_mds(
                num_a=conf.num_a,
                num_b=conf.num_b,
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
                max_repair_set_size=2,
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
                fr"$n_a= {conf.num_a}$, "
                fr"$n_b= {conf.num_b}$, "
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
    node_id_to_objs_list = get_node_id_to_objs_list_for_a_b_mds(
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
    plot_capacity_region_for_a_b_mds_w_field_size(num_nodes=20)
