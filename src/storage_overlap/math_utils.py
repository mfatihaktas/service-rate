import math
import mpmath as mp
import scipy

from typing import Callable

from src.utils.debug import *


# mp.dps = 100


def prob_cum_demand_leq_cum_supply_w_mpmath(
    num_demands: int,
    demand_pdf: Callable[[float], float],
    d: int,
    span_size: float,
    maximal_load: float = 1,
) -> float:
    """
    Computes the following integral:
    Suppose `num_demands` = 3
    \int_{0}^{d} \int_{0}^{span_size - x1} \int_{0}^{span_size - x1 - x2}
    demand_pdf(x1, x2, x3) dx3 dx2 dx1

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

    check(num_demands <= 3, "mpmath.quad supports at most 3 dimensions.")

    log(DEBUG, "Started",
        num_demands=num_demands,
        demand_pdf=demand_pdf,
        d=d,
        span_size=span_size,
        maximal_load=maximal_load,
    )

    d_ = d * maximal_load
    cum_supply_ = span_size * maximal_load

    def helper(
        *args,
        integral_limits_list: list[float] = [],
    ):
        # check(len(args) + 1 == len(integral_limits_list))

        # log(DEBUG, "", args=args, integral_limits_list=integral_limits_list)

        if len(args) == num_demands:
            return math.prod(
                [
                    demand_pdf(arg) for arg in args
                ]
            )

        integral_limits_list_ = [
            *integral_limits_list,
            [0, min(d_, cum_supply_ - sum(args))]
        ]
        # log(DEBUG, "", integral_limits_list_=integral_limits_list_)

        return mp.quad(
            lambda x, *args: helper(x, *args, integral_limits_list=integral_limits_list_),
            *integral_limits_list_,
            # verbose=True,
            error=True,
            # maxdegree=3,  # 4,
            # lambda x: x**2, *integral_limits_list
        )[0]

    return helper()


def prob_cum_demand_leq_cum_supply_w_scipy(
    num_demands: int,
    demand_pdf: Callable[[float], float],
    d: int,
    span_size: float,
    maximal_load: float = 1,
) -> float:
    """
    Computes the following integral:
    Suppose `num_demands` = 3
    \int_{0}^{d} \int_{0}^{span_size - x1} \int_{0}^{span_size - x1 - x2}
    demand_pdf(x1, x2, x3) dx3 dx2 dx1

    Ref: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.nquad.html#scipy.integrate.nquad
    """

    log(DEBUG, "Started",
        num_demands=num_demands,
        demand_pdf=demand_pdf,
        d=d,
        span_size=span_size,
        maximal_load=maximal_load,
    )

    d_ = d * maximal_load
    cum_supply_ = span_size * maximal_load

    def func(*args) -> float:
        return math.prod(
            [
                demand_pdf(arg) for arg in args
            ]
        )

    integral_result = scipy.integrate.nquad(
        func=func,
        ranges=[
            lambda *args: [0, min(d_, cum_supply_ - sum(args))]
            for i in range(num_demands)
        ],
        opts={
            # "limit": 10,
            "epsabs": 0.001
        }
    )
    log(DEBUG, "", integral_result=integral_result)

    return integral_result[0]
