import numpy

from src.storage_overlap import model

from src.utils.debug import *


def test_CyclicDesignModelForGivenDemandDistribution():
    k, n, d = 10, 10, 2
    cyclic_model = model.CyclicDesignModelForGivenDemandDistribution(k=k, n=n, d=d)

    log(DEBUG, "", cyclic_model=cyclic_model)

    for mean_obj_demand in numpy.linspace(0.1, 1, 10):
        demand_rv = random_variable.Exponential(mu=1 / mean_obj_demand)

        P_ub = cyclic_model.prob_serving_upper_bound(
            combination_size=2,
            demand_rv=,
        )

        log(DEBUG, "", mean_obj_demand=mean_obj_demand, P_ub=P_ub)
