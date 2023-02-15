from src.model import popularity
from src.sim import random_variable


def test_PopModel_wZipf():
    pop_model = popularity.PopModel_wZipf(
        k=2,
        zipf_tail_index_rv=random_variable.TruncatedNormal(mu=1, sigma=2),
        arrival_rate_rv=random_variable.TruncatedNormal(mu=1.5, sigma=0.4),
    )
    # pop_model.plot_heatmap_2d()
    pop_model.plot_kde_2d()

    # rv = random_variable.TruncatedNormal(mu=2, sigma=0.01)
    # print("samples= \n{}".format([rv.sample() for _ in range(1000)]))
