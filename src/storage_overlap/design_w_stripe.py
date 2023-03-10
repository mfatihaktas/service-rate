import collections
import dataclasses
import random

from src.service_rate import service_rate_w_stripe
from src.storage_overlap import design

from src.utils.debug import *


@dataclasses.dataclass
class DesignWithStripe(design.ReplicaDesign):
    # Each object is split into `s` stripes, stored with `d - s` parity chunks.
    s: int

    def __post_init__(self):
        check(self.d >= self.s, "`d` must be greater than `s`", d=self.d, s=self.s)

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

    def is_demand_vector_covered(
        self,
        demand_vector: list[float],
    ) -> bool:
        return self.service_rate_inspector.is_in_cap_region(demand_vector)


@dataclasses.dataclass(repr=False)
class ClusteringDesignWithStripe(DesignWithStripe):
    def __post_init__(self):
        check(self.n % self.d == 0, f"d= {self.d} must divide n= {self.n}")

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
        # return f"ClusteringWithStripe(k= {self.k}, n= {self.n}, d= {self.d})"
        return r"$\textrm{ClusteringWithStripe}, s= $" + fr"{self.s}"


@dataclasses.dataclass
class CyclicDesignWithStripe(DesignWithStripe):
    def __post_init__(self):
        check(self.n % self.d == 0, f"d= {self.d} must divide n= {self.n}")

        shift_size = 1
        self.obj_id_to_node_id_set_map = {
            obj_id: set(
                i % self.n
                for i in range(obj_id, obj_id + shift_size * self.d, shift_size)
            )
            for obj_id in range(self.k)
        }

        super().__post_init__()

    def __repr__(self):
        return (
            "CyclicDesignWithStripe( \n"
            f"\t k= {self.k} \n"
            f"\t n= {self.n} \n"
            f"\t d= {self.d} \n"
            f"\t s= {self.s} \n"
            ")"
        )

    def repr_for_plot(self):
        # return f"CyclicWithStripe(k= {self.k}, n= {self.n}, d= {self.d}, s= {self.s})"
        return r"$\textrm{CyclicWithStripe}$, " + fr"$s={self.s}$"


@dataclasses.dataclass
class RandomBlockDesignWithStripe(DesignWithStripe):
    def __post_init__(self):
        self.obj_id_to_node_id_set_map = collections.defaultdict(set)

        num_objs_per_node = (self.k * self.d) // self.n
        # log(DEBUG, f"num_objs_per_node= {num_objs_per_node}")
        node_id_to_obj_id_set_map = collections.defaultdict(set)
        # node_id_to_num_objs_map = {node_id: 0 for node_id in range(self.n)}

        obj_id_queue = collections.deque(
            [
                obj_id
                for _ in range(self.d)
                for obj_id in range(self.k)
            ]
        )

        for obj_id in obj_id_queue:
            # log(DEBUG, f"> obj_id= {obj_id}, rep_id= {rep_id}")

            node_id = random.randint(0, self.n - 1)
            # log(DEBUG, f"node_id= {node_id}")

            counter = 0
            while (
                len(node_id_to_obj_id_set_map[node_id]) == num_objs_per_node
                or node_id in self.obj_id_to_node_id_set_map[obj_id]
            ):
                node_id = (node_id + 1) % self.n
                # log(DEBUG, f"counter= {counter}",
                #     node_id=node_id,
                #     rep_id=rep_id,
                # )

                counter += 1
                if counter == self.n:
                    # check(False, f"Could not find node for obj_id= {obj_id}",
                    #       node_id_to_num_objs_map=node_id_to_num_objs_map,
                    #       obj_id_to_node_id_set_map=self.obj_id_to_node_id_set_map,
                    # )

                    for node_id in range(self.n):
                        if node_id in self.obj_id_to_node_id_set_map[obj_id]:
                            continue

                        obj_id_to_swap_set = node_id_to_obj_id_set_map[node_id]
                        obj_id_to_swap = next(iter(obj_id_to_swap_set))

                        self.obj_id_to_node_id_set_map[obj_id_to_swap].remove(node_id)
                        self.obj_id_to_node_id_set_map[obj_id].add(node_id)

                        obj_id_queue.append(obj_id_to_swap)

            self.obj_id_to_node_id_set_map[obj_id].add(node_id)
            node_id_to_obj_id_set_map[node_id].add(obj_id)

        super().__post_init__()

    def __repr__(self):
        return (
            "RandomBlockDesignWithStripe( \n"
            f"\t k= {self.k} \n"
            f"\t n= {self.n} \n"
            f"\t d= {self.d} \n"
            f"\t s= {self.s} \n"
            ")"
        )

    def repr_for_plot(self):
        # return f"RandomBlockDesignWithStripe(k= {self.k}, n= {self.n}, d= {self.d}, s= {self.s})"
        return r"$\textrm{RandomBlockDesignWithStripe}$, " + fr"$s={self.s}$"
