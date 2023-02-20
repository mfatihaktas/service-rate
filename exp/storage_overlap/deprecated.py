def plot_frac_demand_vectors_covered_vs_num_popular_objs_and_demand_for_popular():
    d = 3
    num_popular_obj_list = list(range(1, 10))
    demand_for_popular_list = list(range(1, d + 1))

    num_sample = 300
    num_sim_run = 3

    log(INFO, "Started",
        d=d,
        num_popular_obj_list=num_popular_obj_list,
        demand_for_popular_list=demand_for_popular_list,
        num_sample=num_sample,
        num_sim_run=num_sim_run,
    )

    ax = plot.axes(projection="3d")

    def plot_(replica_design: design.ReplicaDesign):
        log(INFO, f">> replica_design= {replica_design}")

        E_frac_of_demand_vectors_covered_list = []
        std_frac_of_demand_vectors_covered_list = []

        x_l, y_l, z_l = [], [], []
        for demand_for_popular in demand_for_popular_list:
            for num_popular_obj in num_popular_obj_list:
                frac_of_demand_vectors_covered_list = replica_design.sim_frac_of_demand_vectors_covered(
                    num_popular_obj=num_popular_obj,
                    cum_demand=demand_for_popular * num_popular_obj,
                    zipf_tail_index=0,
                    num_sample=num_sample,
                    num_sim_run=num_sim_run,
                )

                log(INFO, f"> demand_for_popular= {demand_for_popular}, num_popular_obj= {num_popular_obj}",
                    frac_of_demand_vectors_covered_list=frac_of_demand_vectors_covered_list,
                )
                E_frac_of_demand_vectors_covered_list.append(numpy.mean(frac_of_demand_vectors_covered_list))
                std_frac_of_demand_vectors_covered_list.append(numpy.std(frac_of_demand_vectors_covered_list))

                x_l.append(demand_for_popular)
                y_l.append(num_popular_obj)
                z_l.append(numpy.mean(frac_of_demand_vectors_covered_list))

        log(INFO, f"replica_design= {replica_design}",
            demand_for_popular_list=demand_for_popular_list,
            num_popular_obj_list=num_popular_obj_list,
            E_frac_of_demand_vectors_covered_list=E_frac_of_demand_vectors_covered_list,
            std_frac_of_demand_vectors_covered_list=std_frac_of_demand_vectors_covered_list,
        )

        ax.scatter(x_l, y_l, z_l, c=z_l, label=f"{replica_design.repr_for_plot()}", cmap="viridis", linewidth=0.5)

    # k = 45
    # k = 111
    k = 120
    n = k
    replica_design_list = [
        design.ClusteringDesign(k=k, n=n, d=d),
        # design.CyclicDesign(k=k, n=n, d=d, shift_size=1),
        # design.CyclicDesign(k=k, n=n, d=d, shift_size=2),
        # design.CyclicDesign(k=k, n=n, d=d, shift_size=3),
    ]

    for replica_design in replica_design_list:
        plot_(replica_design=replica_design)

    fontsize = 14
    plot.legend(fontsize=fontsize)
    # plot.yscale("log")
    ax.set_xlabel("Number of popular objects", fontsize=fontsize)
    ax.set_ylabel("Demand for popular object", fontsize=fontsize)
    ax.set_zlabel("Fraction of demands covered", fontsize=fontsize)

    plot.title(
        fr"$k= {k}$, "
        fr"$n= {n}$, "
        fr"$d= {d}$, "
        r"$N_{\textrm{sample}}= $" + fr"${num_sample}$, "
        r"$N_{\textrm{sim}}= $" + fr"${num_sim_run}$"
    )

    # Save the plot
    plot.gcf().set_size_inches(8, 6)
    file_name = (
        "plots/plot_frac_demand_vectors_covered_vs_num_popular_objs_and_demand_for_popular"
        + f"_k_{k}"
        + f"_d_{d}"
        + ".png"
    )
    plot.savefig(file_name, bbox_inches="tight")
    plot.gcf().clear()

    log(INFO, "Done")
