import joblib

from src.allocation_w_complexes import model, sim

from src.utils.debug import *
from src.utils.plot import *


def plot_prob_span_of_every_t_complexes_geq_u(
    n: int,
    m: int,
    d: int,
    t: int,
    num_samples: int = 300,
):
    log(INFO, "Started",
        n=n,
        m=m,
        d=d,
        t=t,
        num_samples=num_samples,
    )

    sim_prob_list = []
    prob_list = []

    u_list = list(range(d, t * d))
    for u in u_list:
        log(DEBUG, f"> u= {u}")

        sim_prob = sim.sim_prob_span_of_every_t_complexes_geq_u(
            n=n, m=m, d=d, t=t, u=u, num_samples=num_samples
        )
        sim_prob_list.append(sim_prob)

        prob = model.prob_span_of_every_t_complexes_geq_u_alternative(
            n=n, m=m, d=d, t=t, u=u
        )
        prob_list.append(prob)

        log(INFO, "",
            u=u,
            sim_prob=sim_prob,
            prob=prob,
        )

    color = next(dark_color_cycle)
    plot.plot(u_list, sim_prob_list, label=f"Sim, t= {t}", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)
    plot.plot(u_list, prob_list, label=f"Model, t= {t}", color=color, marker=next(marker_cycle), linestyle="dotted", lw=2, mew=3, ms=5)

    fontsize = 14
    plot.legend(fontsize=fontsize)
    # plot.yscale("log")
    plot.ylabel("Pr{Span of every t-complex >= u}", fontsize=fontsize)
    plot.xlabel(r"$u$", fontsize=fontsize)

    plot.title(
        fr"$n= {n}$, "
        fr"$m= {m}$, "
        fr"$d= {d}$"
    )

    log(INFO, "Done")


def plot_prob_span_of_complexes_geq_u(
    n: int,
    m: int,
    d: int,
    num_samples: int = 300,
):
    t_list = list(range(2, m + 1))

    log(INFO, "Started",
        n=n,
        m=m,
        d=d,
        t_list=t_list,
        num_samples=num_samples,
    )

    for t in t_list:
        plot_prob_span_of_every_t_complexes_geq_u(
            n=n,
            m=m,
            d=d,
            t=t,
            num_samples=num_samples,
        )

    # Save the plot
    plot.gcf().set_size_inches(8, 6)
    file_name = (
        "plots/plot_prob_span_of_complexes_geq_u"
        + f"_n_{n}"
        + f"_m_{m}"
        + f"_d_{d}"
        + ".png"
    )
    plot.savefig(file_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")


def manage_plot_prob_span_of_complexes_geq_u_w_joblib():
    log(INFO, "Started")

    joblib.Parallel(n_jobs=-1, prefer="processes")(
        joblib.delayed(plot_prob_span_of_complexes_geq_u)(
            n=n,
            m=m,
            d=d,
            num_samples=10**4,
        )
        # for n in [10]
        for n in [100]
        for m in [10, 20]
        for d in range(2, 7)
    )

    log(INFO, "Done")


if __name__ == "__main__":
    manage_plot_prob_span_of_complexes_geq_u_w_joblib()
