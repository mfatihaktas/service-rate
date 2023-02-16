import abc
import collections
import cvxpy
import dataclasses
import functools
import itertools
import networkx
import operator

from src.utils.debug import *
from src.utils.plot import *


@dataclasses.dataclass
class Object:
    @abc.abstractmethod
    def get_num_symbols(self):
        pass

    @abc.abstractmethod
    def get_symbols(self):
        pass

    @abc.abstractmethod
    def get_networkx_label(self):
        pass


@dataclasses.dataclass
class SysObject(Object):
    symbol: int

    # def __eq__(self, other_obj: Object):
    #     return isinstance(other_obj, SysObject) and self.symbol == other_obj.symbol

    def __cmp__(self, other_obj: Object):
        if self.symbol == other_obj.symbol:
            return 0
        elif self.symbol < other_obj.symbol:
            return -1
        else:
            return 1

    def __hash__(self):
        return hash(self.symbol)

    def get_symbols(self) -> list[int]:
        return [self.symbol]

    def get_xor(self) -> int:
        return self.symbol

    def get_num_symbols(self) -> int:
        return 1

    def get_networkx_label(self) -> str:
        return str(self.symbol)


@dataclasses.dataclass
class XORedObject(Object):
    symbols: tuple[int]

    # def __eq__(self, other_obj: Object):
    #     return (
    #         isinstance(other_obj, XORedObject)
    #         and self.symbols == other_obj.symbols
    #     )

    def __cmp__(self, other_obj: Object):
        if self.symbols == other_obj.symbols:
            return 0
        elif self.symbols < other_obj.symbols:
            return -1
        else:
            return 1

    def __hash__(self):
        return hash(self.symbols)

    def get_symbols(self) -> list[int]:
        return self.symbols

    def get_xor(self) -> int:
        return functools.reduce(operator.ixor, self.symbols)

    def get_num_symbols(self) -> int:
        return len(self.symbols)

    def get_networkx_label(self) -> str:
        return "+".join(str(sym) for sym in self.symbols)


def get_symbol_recovered_by_object_pair(obj_1: Object, obj_2: Object) -> int:
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
    obj: Object

    def get_touched_objects(self) -> list[Object]:
        return [self.obj]

@dataclasses.dataclass
class RecoveryEdge:
    obj_1: Object
    obj_2: Object

    def get_touched_objects(self) -> list[Object]:
        return [self.obj_1, self.obj_2]


@dataclasses.dataclass
class AccessGraph:
    k: int
    obj_to_num_copies_map: dict[Object, int] = dataclasses.field(default=None)

    obj_to_num_copies_var_map: dict[Object, cvxpy.Variable] = dataclasses.field(default=None)
    symbol_to_access_edges_map: dict[int, list[AccessEdge]] = dataclasses.field(default=None)

    def __post_init__(self):
        if self.obj_to_num_copies_map is None:
            # Construct `obj_to_num_copies_var_map`
            self.obj_to_num_copies_var_map = {}
            # Systematic copies
            for s in range(self.k):
                obj = SysObject(symbol=s)
                self.obj_to_num_copies_var_map[obj] = cvxpy.Variable(integer=True)

            # XOR'ed copies
            for xor_size in range(2, self.k + 1):
                for symbol_combination in itertools.combinations(list(range(self.k)), xor_size):
                    obj = XORedObject(symbols=symbol_combination)
                    self.obj_to_num_copies_var_map[obj] = cvxpy.Variable(integer=True)

        # Construct `symbol_to_access_edges_map`
        self.symbol_to_access_edges_map = collections.defaultdict(list)
        for symbol in range(self.k):
            self.symbol_to_access_edges_map[symbol].append(AccessLoop(obj=SysObject(symbol=symbol)))

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

    def get_networkx_graph(self):
        graph = networkx.Graph()

        # Add nodes
        for obj, num_copies in self.obj_to_num_copies_map.items():
            graph.add_node(
                obj,
                multiplicity=num_copies,
                subset=f"{obj.get_num_symbols()}"
            )

        # Add edges
        # self.symbol_to_access_edges_map

        return graph

    def draw(self, file_name_suffix: str = None):
        log(INFO, "Started",
            file_name_suffix=file_name_suffix,
        )

        networkx_graph = self.get_networkx_graph()
        networkx.draw(
            networkx_graph,
            pos=networkx.multipartite_layout(networkx_graph),
            with_labels=True,
            labels={
                obj: f"{obj.get_networkx_label()}\n x {num_copies}"
                for obj, num_copies in self.obj_to_num_copies_map.items()
            },
            node_color=["w" if num_copies == 0 else "0.8" for obj, num_copies in self.obj_to_num_copies_map.items()],
            font_size=20,
        )

        if file_name_suffix:
            plot.savefig(f"plots/storage_graph_{file_name_suffix}.png", bbox_inches="tight")
            plot.gcf().clear()

        log(INFO, "Done")
