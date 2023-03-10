import dataclasses
import random

from src.service_rate import (
    service_rate_w_stripe,
    storage_scheme as storage_scheme_module,
)
from src.storage_overlap import design

from src.utils.debug import *


@dataclasses.dataclass
class DesignWithStripe(design.ReplicaDesign):
    # Each object is split into `s` stripes, stored with `d - s` parity chunks.
    s: int

    def __post_init__(self):
        log(WARNING, "", use_cvxpy=self.use_cvxpy)
        self.service_rate_inspector = self.get_service_rate_inspector()

    def get_service_rate_inspector(self) -> service_rate_w_stripe.ServiceRateInspectorForStorageWithStripeAndParity:
        # node_id_to_objs_list = [[] for _ in range(self.n)]
        # for obj_id, node_id_set in self.obj_id_to_node_id_set_map.items():
        #     node_id_list = list(node_id_set)
        #     random.shuffle(node_id_list)

        #     index = 0
        #     for stripe_id in range(self.s):
        #         node_id_to_objs_list[node_id_list[index]].append(
        #             storage_scheme_module.PlainObj(id_str=f"{obj_id}-{stripe_id}")
        #         )
        #         index += 1

        #     for i in range(len(node_id_list) - self.s):
        #         node_id_to_objs_list[node_id_list[index]].append(
        #             storage_scheme_module.CodedObj(
        #                 coeff_obj_list=[
        #                     ((i + 1)**stripe_id, storage_scheme_module.PlainObj(id_str=f"{obj_id}-{stripe_id}"))
        #                     for stripe_id in range(self.s)
        #                 ]
        #             )
        #         )
        #         index += 1

        # storage_scheme = storage_scheme_module.StorageScheme(node_id_to_objs_list=node_id_to_objs_list)
        # log(DEBUG, "", storage_scheme=storage_scheme)

        # return service_rate.ServiceRateInspector(
        #     m=len(node_id_to_objs_list),
        #     C=1,
        #     G=storage_scheme.obj_encoding_matrix,
        #     obj_id_to_node_id_map=storage_scheme.obj_id_to_node_id_map,
        #     max_repair_set_size=self.s,
        # )

        return service_rate_w_stripe.ServiceRateInspectorForStorageWithStripeAndParity(
            k=self.k,
            n=self.n,
            s=self.s,
            obj_id_to_node_id_set_map=self.obj_id_to_node_id_set_map,
        )


@dataclasses.dataclass(repr=False)
class ClusteringDesignWithStripe(DesignWithStripe):
    def __post_init__(self):
        check(self.n % self.d == 0, f"d= {self.d} must divide n= {self.n}")
        check(self.d >= self.s, "`d` must be greater than `s`", d=self.d, s=self.s)

        self.obj_id_to_node_id_set_map = {
            obj_id: set(
                i % self.n for i in range(obj_id * self.d , obj_id * self.d + self.d)
            )
            for obj_id in range(self.k)
        }

        super().__post_init__()

    def __repr__(self):
        return (
            "ClusteringDesignWithStripe( \n"
            f"\t k= {self.k} \n"
            f"\t n= {self.n} \n"
            f"\t d= {self.d} \n"
            f"\t s= {self.s} \n"
            ")"
        )

    def repr_for_plot(self):
        # return f"ClusteringDesign_wStripeAndParity(k= {self.k}, n= {self.n}, d= {self.d})"
        return r"$\textrm{ClusteringDesign_wStripeAndParity}, s= $" + fr"{self.s}"

    def is_demand_vector_covered(
        self,
        demand_vector: list[float],
    ) -> bool:
        return self.service_rate_inspector.is_in_cap_region(demand_vector)
