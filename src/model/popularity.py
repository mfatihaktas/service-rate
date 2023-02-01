import numpy
import pandas
import random
import scipy
import seaborn

from typing import Callable

from src.sim import random_variable
from src.plot_utils import *
from src.debug_utils import *

seaborn.set(style='white', color_codes=True)


class PopModel_wZipf(object):
    def __init__(
        self,
        k: int,
        zipf_tail_index_rv: random_variable.RandomVariable,
        arrival_rate_rv: random_variable.RandomVariable,
    ):
        self.k = k
        self.zipf_tail_index_rv = zipf_tail_index_rv
        self.arrival_rate_rv = arrival_rate_rv

        self.cap_list_ = self.cap_list(5000)
        # self.values = numpy.array(self.cap_list_).reshape((self.k, len(self.cap_list_) )).T
        self.values = numpy.column_stack(tuple(self.cap_list_) )
        # log(INFO, "", values=self.values)

        self.kernel, self.max_l = self.gaussian_kde()

    def __repr__(self):
        return (
            "PopModel_wZipf( \n"
            f"\t k= {self.k} \n"
            f"\t zipf_tail_index_rv= {self.zipf_tail_index_rv} \n"
            f"\t arrival_rate_rv= {self.arrival_rate_rv} \n"
            ")"
        )

    def prob_list(self, tail_index: float):
        self.v_l = numpy.arange(1, self.k+1)
        w_l = [float(v)**(-tail_index) for v in self.v_l]
        return [w/sum(w_l) for w in w_l]

    def cap_list(self, num_points: int):
        cap_list = []
        tail_index_list = [self.zipf_tail_index_rv.sample() for _ in range(num_points)]
        for tail_index in tail_index_list:
            prob_list = self.prob_list(tail_index)
            if random.uniform(0, 1) < 0.5:  # each symbol is equally likely to be more popular
                prob_list.reverse()
                ar = self.arrival_rate_rv.sample()
                # if ar > 2:
                #   print("ar= {} > 2!".format(ar))
                cap_list.append(numpy.array(prob_list) * ar)

        return cap_list

    def integrate_over_pop_model(self, func: Callable):
        range_list = []
        for m in self.max_l:
            range_list.append((0, m))

        log(INFO, "", range_list=range_list)
        result, abserr = scipy.integrate.nquad(func, range_list, opts={"limit": 200, "epsabs": 1.49e-04})
        return round(result, 2)

    def gaussian_kde(self, num_points=10000):
        max_l = numpy.amax(self.values, axis=1).tolist()
        kernel = scipy.stats.gaussian_kde(self.values) # bw_method="silverman"
        return kernel, max_l

    def joint_pdf(self, *args):
        return self.kernel(numpy.array(args).reshape((self.k, 1)) )[0]

    def plot_heatmap_2d(self):
        plot.plot(self.values[0, :], self.values[1, :], "k.", markersize=2)
        fig = plot.gcf()
        fig.set_size_inches(5, 5)
        plot.savefig("plot_scatter_2d.png", bbox_inches="tight")
        fig.clear()

        data = pandas.DataFrame(self.cap_list_, columns=["a", "b"] )
        # print("data= {}".format(data) )
        # seaborn.jointplot(x="a", y="b", data=data)
        seaborn.jointplot(x="a", y="b", data=data, kind="kde", space=0) # color="red"

        plot.xlim(xmin=0)
        plot.ylim(ymin=0)
        st = plot.suptitle(r"$k= {}$, $a \sim {}$, $\lambda \sim {}$".format(self.k, self.zipf_tail_index_rv, self.arrival_rate_rv) )
        fig = plot.gcf()
        fig.set_size_inches(5, 5)
        plot.savefig("plot_heatmap_2d.png", bbox_extra_artists=[st], bbox_inches="tight")
        fig.clear()
        log(INFO, "Done")

    def plot_kde_2d(self):
        xmin, xmax = min(self.values[0, :]), max(self.values[0, :])
        ymin, ymax = min(self.values[1, :]), max(self.values[1, :])
        X, Y = numpy.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = numpy.vstack([X.ravel(), Y.ravel() ])
        # blog(positions=positions)
        Z = numpy.reshape(self.kernel(positions).T, X.shape)
        # blog(Z=Z)

        plot.imshow(numpy.rot90(Z), cmap=plot.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
        plot.plot(self.values[0, :], self.values[1, :], "k.", markersize=2)
        plot.xlim([xmin, xmax])
        plot.ylim([ymin, ymax])
        plot.xlabel(r"$\lambda_a$", fontsize=20)
        plot.ylabel(r"$\lambda_b$", fontsize=20)
        # plot.title(r"$\lambda \sim {}$, $\alpha \sim {}$".format(self.arrival_rate_rv, self.zipf_tail_index_rv), fontsize=18)
        prettify(plot.gca())
        fig = plot.gcf()
        fig.set_size_inches(4, 4)
        plot.savefig("plot_kde_2d.png", bbox_inches="tight")
        fig.clear()

        log(INFO, "Done")
