import joblib

from src.utils.plot import *

from src.utils import exp_utils


def plot_P(
    d_list: list[int],
    num_active_objs: int,
    maximal_load: float,
    demand_dist: exp_utils.DemandDistribution,
    plot_ub: bool,
    num_samples: int = 300,
    num_sim_run: int = 3,
):
    k = 120
    if num_active_objs is None:
        num_active_objs = k

    for d in d_list:
        exp_utils.plot_P_for_given_params(
            k=k,
            d=d,
            num_active_objs=num_active_objs,
            maximal_load=maximal_load,
            demand_dist=demand_dist,
            plot_ub=plot_ub,
            num_samples=num_samples,
            num_sim_run=num_sim_run,
        )

    fontsize = 16
    plot.legend(fontsize=14, loc="upper right", bbox_to_anchor=(1.35, 0.75))
    plot.ylabel(r"$\mathcal{P}$", fontsize=fontsize)

    # Save the plot
    plot.gcf().set_size_inches(6, 4)
    file_name = (
        f"plots/plot_P_{demand_dist.value}_demand"
        + f"_k_{k}"
        + f"_d_{d}"
        + f"_m_{maximal_load}"
        # + f"_n_active_{num_active_objs}"
        + ".pdf"
    )
    plot.savefig(file_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")


def manage_plot_P_w_joblib():
    log(INFO, "Started")

    joblib.Parallel(n_jobs=-1, prefer="processes")(
        joblib.delayed(plot_P)(
            d_list=[3],
            # d_list=[10],
            # d_list=[2, 3, 4],
            # d_list=[5, 6],
            # d_list=[2, 3, 4, 5],
            num_active_objs=None,
            maximal_load=1,
            demand_dist=demand_dist,
            plot_ub=False,  # False
            num_samples=300,  # 300,
            # num_samples=1000,
            num_sim_run=3,
        )
        for demand_dist in [
            exp_utils.DemandDistribution.Bernoulli,
            # exp_utils.DemandDistribution.Exp,
            # exp_utils.DemandDistribution.Pareto,
        ]
    )

    log(INFO, "Done")


if __name__ == "__main__":
    manage_plot_P_w_joblib()
