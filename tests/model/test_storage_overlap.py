import pytest

from typing import Tuple

from src.model import storage_overlap as storage_overlap_model

from src.utils.debug import *
from src.utils.plot import *


@pytest.fixture(
    scope="session",
    params=[
        # (10, 2),
        # (10, 3),
        # (20, 3),
        # (20, 4),
        # (30, 4),
        # (40, 4),
        (40, 3),
        # (40, 4),
    ],
)
def k_d(request) -> Tuple[int, int]:
    return request.param


def test_RandomExpanderDesignModel(k_d: Tuple[int, int]):
    k, d = k_d
    lambda_ = d

    log(DEBUG, "", k=k, d=d, lambda_=lambda_)

    model = storage_overlap_model.RandomExpanderDesignModel(k=k, n=k, d=d)

    m_list = []
    prob_serving_lower_bound_list = []
    prob_serving_upper_bound_list = []
    for m in range(1, k // lambda_ + 1):
        m_list.append(m)

        prob_serving_lower_bound = model.prob_serving_lower_bound(m=m, lambda_=lambda_)
        prob_serving_upper_bound = model.prob_serving_upper_bound(m=m, lambda_=lambda_)

        log(DEBUG, f"> m= {m}",
            prob_serving_lower_bound=prob_serving_lower_bound,
            prob_serving_upper_bound=prob_serving_upper_bound,
        )

        prob_serving_lower_bound_list.append(prob_serving_lower_bound)
        prob_serving_upper_bound_list.append(prob_serving_upper_bound)

    plot.plot(m_list, prob_serving_lower_bound_list, label="LB", color=next(dark_color_cycle), marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
    plot.plot(m_list, prob_serving_upper_bound_list, label="UB", color=next(dark_color_cycle), marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    fontsize = 14
    plot.legend(fontsize=fontsize)
    # plot.yscale("log")
    plot.ylabel("Fraction of demand vectors covered", fontsize=fontsize)
    plot.xlabel("Number of popular objects", fontsize=fontsize)

    plot.title(
        fr"$k= {k}$, "
        fr"$d= {d}$, "
        r"$\lambda= $" + fr"${lambda_}$"
    )

    # Save the plot
    plot.gcf().set_size_inches(8, 6)
    file_name = (
        "plots/plot_model"
        + f"_k_{k}"
        + f"_d_{d}"
        + f"_lambda_{lambda_}"
        + ".png"
    )
    plot.savefig(file_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")
