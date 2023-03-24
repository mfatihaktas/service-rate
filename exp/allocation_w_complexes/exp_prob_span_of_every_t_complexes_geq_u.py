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

    u_list = list(range(d, d * t))
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


def plot_frac_demand_vectors_covered_vs_num_popular_objs(
    d: int,
    demand_for_popular: int,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    k = 120
    # k = 30
    n = k
    use_cvxpy = False

    # num_popular_obj_list = [2, 5, 10] + [int(k * frac) for frac in [0.1, 0.15, 0.2, 0.25]]
    # num_popular_obj_list = [2, 5, 10] + [int(k * frac) for frac in [0.1, 0.6, 0.8]]
    # num_popular_obj_list = [2, 5, 10]
    num_popular_obj_list = list(range(1, 10))
    # num_popular_obj_list = list(range(1, 20))
    # num_popular_obj_list = list(range(1, 10)) + [int(k * frac) for frac in [0.1, 0.6, 0.8]]
    # num_popular_obj_list = [2, 5]

    log(INFO, "Started",
        num_popular_obj_list=num_popular_obj_list,
        demand_for_popular=demand_for_popular,
        num_samples=num_samples,
        num_sim_run=num_sim_run,
    )

    for storage_design, storage_design_model in [
        (
            design.RandomExpanderDesign(k=k, n=n, d=d, use_cvxpy=use_cvxpy),
            storage_overlap_model.RandomExpanderDesignModel(k=k, n=n, d=d)
        ),
    ]:
        plot_frac_demand_vectors_covered_vs_num_popular_objs_for_storage_design(
            storage_design=storage_design,
            storage_design_model=storage_design_model,
            demand_for_popular=demand_for_popular,
            num_popular_obj_list=num_popular_obj_list,
            num_samples=num_samples,
            num_sim_run=num_sim_run,
        )

    # Save the plot
    plot.gcf().set_size_inches(8, 6)
    file_name = (
        "plots/exp_random_design_plot_frac_demand_vectors_covered_vs_num_popular_objs"
        + f"_k_{k}"
        + f"_d_{d}"
        + f"_lambda_{demand_for_popular}"
        # + f"_cvxpy_{use_cvxpy}"
        + ".png"
    )
    plot.savefig(file_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")


def manage_plot_frac_demand_vectors_covered_vs_num_popular_objs():
    log(INFO, "Started")

    def plot_(d: int):
        for demand_for_popular in range(1, d + 1):
            plot_frac_demand_vectors_covered_vs_num_popular_objs(
                d=d,
                demand_for_popular=demand_for_popular,
                num_samples=300,
                # num_samples=1000,
                num_sim_run=3,
            )

    # plot_(d=2)
    plot_(d=3)
    # plot_(d=4)
    # plot_(d=5)
    # plot_(d=6)

    log(INFO, "Done")


def manage_plot_frac_demand_vectors_covered_vs_num_popular_objs_w_joblib():
    log(INFO, "Started")

    joblib.Parallel(n_jobs=-1, prefer="processes")(
        joblib.delayed(plot_frac_demand_vectors_covered_vs_num_popular_objs)(
            d=d,
            demand_for_popular=demand_for_popular,
            num_samples=300,
            num_sim_run=3,
            # num_samples=5000,
            # num_sim_run=5,
        )
        # for d in range(1, 2)
        # for d in range(2, 3)
        # for d in range(3, 4)
        # for d in range(2, 5)
        for d in range(2, 7)
        for demand_for_popular in range(1, d + 1)
        # for d in [4]
        # for demand_for_popular in [3]
        # for demand_for_popular in [4]
        # for d in [6]
        # for demand_for_popular in [5]
    )

    log(INFO, "Done")


if __name__ == "__main__":
    # manage_plot_frac_demand_vectors_covered_vs_num_popular_objs()
    manage_plot_frac_demand_vectors_covered_vs_num_popular_objs_w_joblib()
