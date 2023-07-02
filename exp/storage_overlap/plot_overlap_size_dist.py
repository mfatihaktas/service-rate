import collections
import numpy

from src.storage_overlap import design

from src.utils.debug import *
from src.utils.plot import *


def get_node_overlap_size_to_E_and_std_frac_map(
    n: int,
    d: int,
    num_samples: int,
):
    overlap_size_to_frac_list_map = collections.defaultdict(list)
    for _ in range(num_samples):
        storage_design = design.RandomBlockDesign(
            k=n,
            n=n,
            d=d,
            strategy_to_check_if_demand_covered=design.StrategyToCheckIfDemandCovered.cvxpy,
        )

        overlap_size_to_count_map = storage_design.get_node_overlap_size_to_counter_map()
        total_count = sum(overlap_size_to_count_map.values())
        for overlap_size, count in overlap_size_to_count_map.items():
            overlap_size_to_frac_list_map[overlap_size].append(count / total_count)

    return {
        overlap_size: (numpy.mean(frac_list), numpy.std(frac_list))
        for overlap_size, frac_list in overlap_size_to_frac_list_map.items()
    }


def plot_overlap_size_dist_for_random_block_design(
    n: int,
    d: int,
    num_samples: int,
):
    log(INFO, "Started",
        n=n,
        d=d,
        num_samples=num_samples,
    )

    overlap_size_to_E_and_std_frac_map = get_node_overlap_size_to_E_and_std_frac_map(
        n=n,
        d=d,
        num_samples=num_samples,
    )
    log(INFO, "",
        overlap_size_to_E_and_std_frac_map=overlap_size_to_E_and_std_frac_map,
    )

    bar_width = 0.5
    # x_list = [overlap_size + bar_width / 2 for overlap_size in overlap_size_to_E_and_std_frac_map]
    x_list = [overlap_size for overlap_size in overlap_size_to_E_and_std_frac_map]
    E_and_std_frac_list = list(overlap_size_to_E_and_std_frac_map.values())
    y_list = [E_and_std_frac[0] for E_and_std_frac in E_and_std_frac_list]
    yerr_list = [E_and_std_frac[1] for E_and_std_frac in E_and_std_frac_list]
    plot.bar(x_list, y_list, yerr=yerr_list, color=NICE_BLUE, width=bar_width, alpha=0.5)

    plot.xticks(x_list)

    fontsize = 16
    # plot.legend(fontsize=14, framealpha=0.5, loc="upper left", bbox_to_anchor=(1, 1))
    plot.ylabel("Fraction of overlaps", fontsize=fontsize)
    plot.xlabel("Overlap size", fontsize=fontsize)

    plot.title(
        (
            fr"$n= {n}$, "
            fr"$d= {d}$"
        ),
        fontsize=fontsize,
    )

    # Save the plot
    # plot.gcf().set_size_inches(8, 6)
    plot.gcf().set_size_inches(6, 4)
    file_name = (
        "plots/plot_overlap_size_dist_for_random_block_design"
        + f"_n_{n}"
        + f"_d_{d}"
        + ".pdf"
    )
    plot.savefig(file_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")


if __name__ == "__main__":
    plot_overlap_size_dist_for_random_block_design(
        n=1000,
        d=5,
        num_samples=50,
    )
