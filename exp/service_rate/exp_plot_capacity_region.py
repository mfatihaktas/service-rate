from src.service_rate import (
    plot_capacity_region,
    service_rate,
    storage_scheme as storage_scheme_module
)

from src.utils.debug import *
from src.utils.plot import *


def get_node_id_to_objs_list(
    num_a: int,
    num_b: int,
    num_a_plus_b: int,
    num_a_minus_b: int,
) -> list[list[storage_scheme_module.Obj]]:
    node_id_to_objs_list = []

    for _ in range(num_a):
        node_id_to_objs_list.append([storage_scheme_module.PlainObj(id_str="a")])

    for _ in range(num_b):
        node_id_to_objs_list.append([storage_scheme_module.PlainObj(id_str="b")])

    for _ in range(num_a_plus_b):
        node_id_to_objs_list.append(
            [
                storage_scheme_module.CodedObj(
                    coeff_obj_list=[
                        (1, storage_scheme_module.PlainObj(id_str="a")),
                        (1, storage_scheme_module.PlainObj(id_str="b")),
                    ]
                )
            ],
        )

    for _ in range(num_a_minus_b):
        node_id_to_objs_list.append(
            [
                storage_scheme_module.CodedObj(
                    coeff_obj_list=[
                        (1, storage_scheme_module.PlainObj(id_str="a")),
                        (2, storage_scheme_module.PlainObj(id_str="b")),
                    ]
                )
            ],
        )

    return node_id_to_objs_list


def plot_capacity_region_sequence_for_a__b__a_plus_b__a_minus_b(
    num_a: int,
    num_b: int,
    num_coded: int,
):
    log(DEBUG, "Started",
        num_a=num_a,
        num_b=num_b,
        num_coded=num_coded,
    )

    num_a_plus_b_and_num_a_minus_b_list = []
    for num_a_plus_b in range(1, num_coded):
        num_a_minus_b = num_coded - num_a_plus_b
        num_a_plus_b_and_num_a_minus_b_list.append((num_a_plus_b, num_a_minus_b))

    num_plots = len(num_a_plus_b_and_num_a_minus_b_list)
    fig_size = (num_plots * 5, 5)
    fig, ax_list = plot.subplots(1, num_plots, figsize=fig_size)

    def plot_(plot_index: int):
        log(DEBUG, "Started", plot_index=plot_index)

        num_a_plus_b, num_a_minus_b = num_a_plus_b_and_num_a_minus_b_list[plot_index]

        node_id_to_objs_list = get_node_id_to_objs_list(
            num_a=num_a,
            num_b=num_b,
            num_a_plus_b=num_a_plus_b,
            num_a_minus_b=num_a_minus_b,
        )

        scheme = storage_scheme_module.StorageScheme(node_id_to_objs_list)
        # log(DEBUG, "", storage_scheme=scheme)

        service_rate_inspector = service_rate.ServiceRateInspector(
            m=len(node_id_to_objs_list),
            C=1,
            G=scheme.obj_encoding_matrix,
            obj_id_to_node_id_map=scheme.obj_id_to_node_id_map,
            compute_halfspace_intersections=True,
            max_repair_set_size=2,
        )

        ax = ax_list[plot_index]
        plot.sca(ax)
        plot_capacity_region.plot_capacity_region_2d(
            service_rate_inspector=service_rate_inspector,
        )

        fontsize = 14
        plot.xlabel(r"$\lambda_a$", fontsize=fontsize)
        plot.ylabel(r"$\lambda_b$", fontsize=fontsize)

        title = (
            fr"$n_a= {num_a}$, "
            fr"$n_b= {num_b}$, "
            r"$n_{a+b}= $" + fr"${num_a_plus_b}$, "
            r"$n_{a-b}= $" + fr"${num_a_minus_b}$, "
        )
        plot.title(title, fontsize=fontsize)

    for i in range(num_plots):
        plot_(plot_index=i)

    plot.savefig(f"plots/plot_capacity_region_{num_a}_a__{num_b}_b__{num_coded}_coded.png", bbox_inches="tight")
    plot.gcf().clear()
    log(INFO, "Done")


if __name__ == "__main__":
    plot_capacity_region_sequence_for_a__b__a_plus_b__a_minus_b(
        num_a=3,
        num_b=3,
        num_coded=5,
    )
