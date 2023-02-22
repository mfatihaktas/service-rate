import abc
import dataclasses
import functools
import operator


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
