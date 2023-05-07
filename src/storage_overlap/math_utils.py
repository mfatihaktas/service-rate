import math
import mpmath as mp

from typing import Callable

from src.utils.debug import *


# mp.dps = 100


def prob_cum_demand_leq_cum_supply(
    num_objs: int,
    obj_demand_pdf: Callable[[float], float],
    d: int,
    cum_supply: float,
) -> float:
    """
    Computes the following integral:
    Suppose `num_objs` = 3
    \int_{0}^{d} \int_{0}^{cum_supply - x1} \int_{0}^{cum_supply - x1 - x2}
    obj_demand_pdf(x1, x2, x3) dx3 dx2 dx1

    E.g.,
    def f3():
        def f2(x):
            def f1(x,y):
                def f(x,y,z):
                    return 1.0 + x*y + y**2.0 + 3.0*z
                return quadgl(f, [-1.0, 1], [1.2*x, 1.0], [y/4, x**2.0])
            return quadgl(f1, [-1, 1.0], [1.2*x, 1.0])
        return quadgl(f2, [-1.0, 1.0])

    Ref: https://stackoverflow.com/questions/63443828/is-there-a-multiple-integrator-in-python-providing-both-variable-integration-lim
    """

    log(DEBUG, "Started",
        num_objs=num_objs,
        obj_demand_pdf=obj_demand_pdf,
        d=d,
        cum_supply=cum_supply,
    )

    def helper(
        *args,
        integral_limits_list: list[float] = [],
    ):
        # check(len(args) + 1 == len(integral_limits_list))

        # log(DEBUG, "", args=args, integral_limits_list=integral_limits_list)

        if len(args) == num_objs:
            return math.prod(
                [
                    obj_demand_pdf(arg) for arg in args
                ]
            )

        integral_limits_list_ = [
            *integral_limits_list,
            [0, min(d, cum_supply - sum(args))]
        ]
        # log(DEBUG, "", integral_limits_list_=integral_limits_list_)

        return mp.quad(
            lambda x, *args: helper(x, *args, integral_limits_list=integral_limits_list_),
            *integral_limits_list_,
            verbose=True,
            error=True,
            maxdegree=4,
            # lambda x: x**2, *integral_limits_list
        )[0]

    return helper()
