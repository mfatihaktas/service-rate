import abc
import collections
import cvxpy
import dataclasses
import itertools
import networkx

from src.utils import storage_object

from src.utils.debug import *
from src.utils.plot import *


def get_symbol_recovered_by_object_pair(obj_1: storage_object.Object, obj_2: storage_object.Object) -> int:
    if abs(obj_1.get_num_symbols() - obj_2.get_num_symbols()) != 1:
        return None

    if obj_1.get_num_symbols() < obj_2.get_num_symbols():
        obj_1, obj_2 = obj_2, obj_1

    recovered_symbol_set = set(obj_1.get_symbols()) - set(obj_2.get_symbols())
    if len(recovered_symbol_set) > 1:
        return None

    recovered_symbol = next(iter(recovered_symbol_set))
    return recovered_symbol


@dataclasses.dataclass
class AccessEdge:
    @abc.abstractmethod
    def get_touched_objects(self):
        pass


@dataclasses.dataclass
class AccessLoop:
    obj: storage_object.Object

    def get_touched_objects(self) -> list[storage_object.Object]:
        return [self.obj]

@dataclasses.dataclass
class RecoveryEdge:
    obj_1: storage_object.Object
    obj_2: storage_object.Object

    def get_touched_objects(self) -> list[storage_object.Object]:
        return [self.obj_1, self.obj_2]


@dataclasses.dataclass
class AccessGraph:
    k: int
    obj_to_num_copies_map: dict[storage_object.Object, int] = dataclasses.field(default=None)

    obj_to_num_copies_var_map: dict[storage_object.Object, cvxpy.Variable] = dataclasses.field(default=None)
    symbol_to_access_edges_map: dict[int, list[AccessEdge]] = dataclasses.field(default=None)

    def __post_init__(self):
        if self.obj_to_num_copies_map is None:
            # Construct `obj_to_num_copies_var_map`
            self.obj_to_num_copies_var_map = {}
            # Systematic copies
            for s in range(self.k):
                obj = storage_object.SysObject(symbol=s)
                self.obj_to_num_copies_var_map[obj] = cvxpy.Variable(integer=True)

            # XOR'ed copies
            for xor_size in range(2, self.k + 1):
                for symbol_combination in itertools.combinations(list(range(self.k)), xor_size):
                    obj = storage_object.XORedObject(symbols=symbol_combination)
                    self.obj_to_num_copies_var_map[obj] = cvxpy.Variable(integer=True)

        # Construct `symbol_to_access_edges_map`
        self.symbol_to_access_edges_map = collections.defaultdict(list)
        for symbol in range(self.k):
            self.symbol_to_access_edges_map[symbol].append(AccessLoop(obj=storage_object.SysObject(symbol=symbol)))

        obj_list = list(self.obj_to_num_copies_map.keys()) if self.obj_to_num_copies_map else list(self.obj_to_num_copies_var_map.keys())
        for (obj_1, obj_2) in itertools.combinations(obj_list, 2):
            recovered_symbol = get_symbol_recovered_by_object_pair(obj_1=obj_1, obj_2=obj_2)
            if recovered_symbol is None or not (0 <= recovered_symbol <= self.k):
                # log(DEBUG, "Not a recovery group", recovered_symbol=recovered_symbol, obj_1=obj_1, obj_2=obj_2)
                continue

            self.symbol_to_access_edges_map[recovered_symbol].append(RecoveryEdge(obj_1=obj_1, obj_2=obj_2))

        # log(DEBUG, "Constructed",
        #     obj_to_num_copies_var_map=self.obj_to_num_copies_var_map,
        #     symbol_to_access_edges_map=self.symbol_to_access_edges_map,
        # )

    def set_obj_to_num_copies_map_after_optimization(self):
        self.obj_to_num_copies_map = {
            obj: round(float(num_copies_var.value))
            for obj, num_copies_var in self.obj_to_num_copies_var_map.items()
        }

    def get_total_num_nodes(self):
        return sum(self.obj_to_num_copies_map.values())

    def get_networkx_graph(self):
        graph = networkx.Graph()

        # Add nodes
        for obj, num_copies in self.obj_to_num_copies_map.items():
            graph.add_node(
                obj,
                multiplicity=num_copies,
                level=obj.get_num_symbols(),
            )

        # Add edges
        # self.symbol_to_access_edges_map

        return graph

    def draw(self, file_name_suffix: str = None):
        """Refs:
        - https://networkx.org/documentation/stable/auto_examples/drawing/plot_labels_and_colors.html
        """

        log(INFO, "Started",
            file_name_suffix=file_name_suffix,
        )

        max_num_copies = max(self.obj_to_num_copies_map.values())

        networkx_graph = self.get_networkx_graph()
        networkx.draw(
            networkx_graph,
            pos=networkx.multipartite_layout(networkx_graph, subset_key="level"),
            with_labels=True,
            labels={
                obj: f"{obj.get_networkx_label()}\n ({num_copies})"
                for obj, num_copies in self.obj_to_num_copies_map.items()
            },
            node_color=[
                lighten_color(color="tab:blue", amount=(num_copies / max_num_copies))
                for _, num_copies in self.obj_to_num_copies_map.items()
            ],
            font_size=20,
            node_size=1300,
            alpha=0.6,
        )

        if file_name_suffix:
            plot.savefig(f"plots/storage_graph_{file_name_suffix}.png", bbox_inches="tight")
            plot.gcf().clear()

        log(INFO, "Done")
