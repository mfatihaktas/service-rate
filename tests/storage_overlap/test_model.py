import numpy

from src.storage_overlap import model

from src.utils.debug import *


def test_CyclicDesignModelForExpObjDemands():
    k, n, d = 10, 10, 2
    cyclic_model = model.CyclicDesignModelForExpObjDemands(k=k, n=n, d=d)

    log(DEBUG, "", cyclic_model=cyclic_model)

    for mean_obj_demand in numpy.linspace(0.1, 1, 10):
        P_ub = cyclic_model.prob_serving_upper_bound(
            combination_size=2,
            mean_obj_demand=mean_obj_demand,
        )

        log(DEBUG, "", mean_obj_demand=mean_obj_demand, P_ub=P_ub)
