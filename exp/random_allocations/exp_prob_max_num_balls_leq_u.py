import joblib
import math
import numpy

from src.random_allocations import sim

from src.utils.debug import *
from src.utils.plot import *


def plot_prob_max_num_balls_leq_u(
    n: int,
    m: int,
    num_samples: int = 300,
):
    log(INFO, "Started",
        n=n,
        m=m,
        num_samples=num_samples,
    )

    max_num_balls_to_prob_map = sim.sim_max_num_balls_to_prob_map(n=n, m=m, num_samples=num_samples)
    log(INFO, "", max_num_balls_to_prob_map=max_num_balls_to_prob_map)

    max_num_balls_and_prob_list = sorted(list(max_num_balls_to_prob_map.items()))
    max_num_balls_list = [e[0] for e in max_num_balls_and_prob_list]
    prob_list = [e[1] for e in max_num_balls_and_prob_list]
    cdf_list = numpy.cumsum(prob_list)

    plot.plot(max_num_balls_list, cdf_list, label=f"n={n}, m={m}", color=next(dark_color_cycle), marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    fontsize = 14
    plot.legend(fontsize=fontsize)
    plot.ylabel("Pr{Max num balls <= u}", fontsize=fontsize)
    plot.xlabel(r"$u$", fontsize=fontsize)

    plot.title(f"Number of samples= {num_samples}")

    log(INFO, "Done")


def manage_plot_prob_max_num_balls_leq_u():
    log(INFO, "Started")

    # joblib.Parallel(n_jobs=-1, prefer="processes")(
    #     joblib.delayed(plot_prob_max_num_balls_leq_u)(
    #         n=n,
    #         m=m,
    #         num_samples=10**3,
    #     )
    #     for n in [40]
    #     for m in [40]
    # )

    num_samples = 10**3
    for n, m in [
        (40, 40),
        (100, 100),
    ]:
        plot_prob_max_num_balls_leq_u(
            n=n,
            m=m,
            num_samples=num_samples,
        )

    # Save the plot
    plot.gcf().set_size_inches(8, 6)
    file_name = (
        "plots/plot_prob_max_num_balls_leq_u"
        + ".png"
    )
    plot.savefig(file_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")


def plot_prob_max_num_balls_leq_u_alternative(
    n: int,
    lambda_: float,
    num_samples: int = 300,
):
    log(INFO, "Started",
        n=n,
        lambda_=lambda_,
        num_samples=num_samples,
    )

    u = math.floor(1 / lambda_)

    m_list = list(range(2, n))
    prob_list = []
    prob_upper_bound_w_b_2_list = []
    prob_sqrt_upper_bound_w_b_2_list = []
    prob_upper_bound_w_b_3_list = []
    prob_cubed_upper_bound_w_b_3_list = []
    for m in m_list:
        prob = sim.sim_prob_max_num_balls_leq_u(n=n, m=m, u=u, num_samples=num_samples)
        prob_list.append(prob)

        prob_upper_bound_w_b_2 = sim.sim_prob_max_num_balls_leq_u(n=n, m=(2 * m), u=(2 * u), num_samples=num_samples)
        prob_upper_bound_w_b_2_list.append(prob_upper_bound_w_b_2)
        prob_sqrt_upper_bound_w_b_2 = math.pow(prob, 1 / 2)
        prob_sqrt_upper_bound_w_b_2_list.append(prob_sqrt_upper_bound_w_b_2)

        prob_upper_bound_w_b_3 = sim.sim_prob_max_num_balls_leq_u(n=n, m=(3 * m), u=(3 * u), num_samples=num_samples)
        prob_upper_bound_w_b_3_list.append(prob_upper_bound_w_b_3)
        prob_cubed_upper_bound_w_b_3 = math.pow(prob, 1 / 3)
        prob_cubed_upper_bound_w_b_3_list.append(prob_cubed_upper_bound_w_b_3)

        log(DEBUG, f"> m= {m}",
            prob=prob,
            prob_upper_bound_w_b_2=prob_upper_bound_w_b_2,
            prob_sqrt_upper_bound_w_b_2=prob_sqrt_upper_bound_w_b_2,
            prob_upper_bound_w_b_3=prob_upper_bound_w_b_3,
            prob_cubed_upper_bound_w_b_3=prob_cubed_upper_bound_w_b_3,
        )

    plot.plot(m_list, prob_list, label=r"$\mathcal{P}_{m, \lambda}$", color=next(dark_color_cycle), marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    color = next(dark_color_cycle)
    plot.plot(m_list, prob_upper_bound_w_b_2_list, label=r"$\mathcal{P}_{2m, \lambda/2}$", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
    plot.plot(m_list, prob_sqrt_upper_bound_w_b_2_list, label=r"$\mathcal{P}_{m, \lambda}^{1/2}$", color=color, marker=".", linestyle="dotted", lw=2, mew=3, ms=5)

    color = next(dark_color_cycle)
    plot.plot(m_list, prob_upper_bound_w_b_3_list, label=r"$\mathcal{P}_{3m, \lambda/3}$", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
    plot.plot(m_list, prob_cubed_upper_bound_w_b_3_list, label=r"$\mathcal{P}_{m, \lambda}^{1/3}$", color=color, marker=".", linestyle="dotted", lw=2, mew=3, ms=5)

    fontsize = 14
    plot.legend(fontsize=fontsize)
    # plot.ylabel(r"$\mathcal{P}_{m, \lambda}$", fontsize=fontsize)
    # plot.ylabel(r"$\textrm{Pr}\{\textrm{System is stable}\}$", fontsize=fontsize)
    plot.ylabel("Probability of system stability", fontsize=fontsize)
    plot.xlabel(r"$m$", fontsize=fontsize)

    plot.title(
        fr"$n= {n}$, "
        fr"$\lambda= {lambda_}$, "
        r"$N_{sim}= $" + fr"${num_samples}$"
    )

    # Save the plot
    plot.gcf().set_size_inches(8, 6)
    file_name = (
        "plots/plot_prob_max_num_balls_leq_u"
        + f"_n_{n}"
        + f"_u_{n}"
        + ".png"
    )
    plot.savefig(file_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")


def manage_plot_prob_max_num_balls_leq_u_alternative():
    log(INFO, "Started")

    num_samples = 10**3
    for n, lambda_ in [
        # (10, 0.3),
        # (100, 0.3),
        (1000, 0.3),
        # (100, 20, 6),
        # (40, 4, 4),
        # (100, 100),
    ]:
        plot_prob_max_num_balls_leq_u_alternative(
            n=n,
            lambda_=lambda_,
            num_samples=num_samples,
        )

    log(INFO, "Done")


if __name__ == "__main__":
    # manage_plot_prob_max_num_balls_leq_u()
    manage_plot_prob_max_num_balls_leq_u_alternative()
