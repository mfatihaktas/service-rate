import abc
import numpy

from src.debug_utils import *


class Obj(abc.ABC):
    """Base object class"""

    def __init__(self):
        self.id_ = None

    @property
    def id_(self):
        return self._id_

    @id_.setter
    def id_(self, id_):
        self._id_ = id_

    @abc.abstractmethod
    def __hash__(self):
        pass

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


class PlainObj(Obj):
    def __init__(self, id_str: str, orig_id: int = None):
        super().__init__()

        self.id_str = id_str
        self.orig_id = orig_id

    def __hash__(self):
        return hash(self.id_str)

    def __repr__(self):
        return (
            "PlainObj( \n" f"\t id_str= {self.id_str} \n" f"\t id_= {self.id_} \n" ")"
        )


class CodedObj(Obj):
    def __init__(self, coeff_obj_list: list[tuple[int, Obj]]):
        super().__init__()

        check(
            len(coeff_obj_list) > 0,
            f"Needs to be positive; len(coeff_obj_list)= {len(coeff_obj_list)}",
        )
        self._coeff_obj_list = coeff_obj_list

    @property
    def coeff_obj_list(self):
        return self._coeff_obj_list

    def __hash__(self):
        id_tuple = (obj.id_str for _, obj in self.coeff_obj_list)
        return hash(id_tuple)

    def __repr__(self):
        s = "CodedObj( \n"
        s += f"id_= {self.id_} \n"

        for coeff, obj in self.coeff_obj_list:
            s += f"{coeff} x \n{obj} \n"
        s += ")"

        return s


name_to_node_objs_list_map = {
    "a_b_a+b": [
        [PlainObj(id_str="a")],
        [PlainObj(id_str="b")],
        [
            CodedObj(
                coeff_obj_list=[
                    (1, PlainObj(id_str="a")),
                    (1, PlainObj(id_str="b")),
                ]
            )
        ],
    ],

    "a_a_b": [
        [PlainObj(id_str="a")],
        [PlainObj(id_str="a")],
        [PlainObj(id_str="b")],
    ],

    "a_a_b_b": [
        [PlainObj(id_str="a")],
        [PlainObj(id_str="a")],
        [PlainObj(id_str="b")],
        [PlainObj(id_str="b")],
    ],

    "a_b_a+b_a+2b": [
        [PlainObj(id_str="a")],
        [PlainObj(id_str="b")],
        [
            CodedObj(
                coeff_obj_list=[
                    (1, PlainObj(id_str="a")),
                    (1, PlainObj(id_str="b")),
                ]
            )
        ],
        [
            CodedObj(
                coeff_obj_list=[
                    (1, PlainObj(id_str="a")),
                    (2, PlainObj(id_str="b")),
                ]
            )
        ],
    ],

    "a_a_a_b_a+b_a+2b": [
        [PlainObj(id_str="a")],
        [PlainObj(id_str="a")],
        [PlainObj(id_str="a")],
        [PlainObj(id_str="b")],
        [
            CodedObj(
                coeff_obj_list=[
                    (1, PlainObj(id_str="a")),
                    (1, PlainObj(id_str="b")),
                ]
            )
        ],
        [
            CodedObj(
                coeff_obj_list=[
                    (1, PlainObj(id_str="a")),
                    (2, PlainObj(id_str="b")),
                ]
            )
        ],
    ],

    "a,b_a,b": [
        [PlainObj(id_str="a"), PlainObj(id_str="b")],
        [PlainObj(id_str="a"), PlainObj(id_str="b")],
    ],

    "a_b_c_a+b_a+c_b+c": [
        [PlainObj(id_str="a")],
        [PlainObj(id_str="b")],
        [PlainObj(id_str="c")],
        [
            CodedObj(
                coeff_obj_list=[
                    (1, PlainObj(id_str="a")),
                    (1, PlainObj(id_str="b")),
                ]
            )
        ],
        [
            CodedObj(
                coeff_obj_list=[
                    (1, PlainObj(id_str="a")),
                    (1, PlainObj(id_str="c")),
                ]
            )
        ],
        [
            CodedObj(
                coeff_obj_list=[
                    (1, PlainObj(id_str="b")),
                    (1, PlainObj(id_str="c")),
                ]
            )
        ],
    ],
}


class StorageScheme:
    def __init__(self, node_id_to_objs_list: list[list[Obj]]):
        self._node_id_to_objs_list = node_id_to_objs_list

        # Refers to `k`
        self._num_original_objs = self.get_num_original_objs()
        # Refers to `n`
        self._total_num_objs = sum(len(obj_list) for obj_list in node_id_to_objs_list)
        # Refers to `m`
        self._num_nodes = len(self._node_id_to_objs_list)

        self._plain_obj_to_orig_id_map = self.get_plain_obj_to_orig_id_map()

        self._obj_id_to_node_id_map = self.get_obj_id_to_node_id_map()
        # log(DEBUG, "", obj_id_to_node_id_map=self.obj_id_to_node_id_map)

        # This refers to G
        self._obj_encoding_matrix = self.get_obj_encoding_matrix()

    def __repr__(self):
        s = "StorageScheme( \n"
        # for node_id, obj_list in enumerate(self.node_id_to_objs_list):
        #     s += f"node-{node_id}: [\n"
        #     for obj in obj_list:
        #         s += f"{obj} \n"
        #     s += "] \n"
        for node_id, obj_list in enumerate(self.node_id_to_objs_list):
            # s += f"\t node-{node_id}: {len(obj_list)} objs \n"
            num_plain, num_coded = 0, 0
            for obj in obj_list:
                if isinstance(obj, PlainObj):
                    num_plain += 1
                elif isinstance(obj, CodedObj):
                    num_coded += 1
            s += f"\t node-{node_id}: {num_plain} plain, {num_coded} coded objs \n"
        s += ") \n"

        return s

    @property
    def node_id_to_objs_list(self):
        return self._node_id_to_objs_list

    @property
    def num_original_objs(self):
        return self._num_original_objs

    @property
    def total_num_objs(self):
        return self._total_num_objs

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def obj_id_to_node_id_map(self):
        return self._obj_id_to_node_id_map

    @property
    def obj_encoding_matrix(self):
        return self._obj_encoding_matrix

    def get_obj_encoding_matrix(self):
        G = numpy.zeros((self.num_original_objs, self.total_num_objs))

        # log(DEBUG, "", _plain_obj_to_orig_id_map=self._plain_obj_to_orig_id_map)

        for node, obj_list in enumerate(self.node_id_to_objs_list):
            for obj in obj_list:

                if isinstance(obj, PlainObj):
                    G[self._plain_obj_to_orig_id_map[obj], obj.id_] = 1
                elif isinstance(obj, CodedObj):
                    for coeff, plain_obj in obj.coeff_obj_list:
                        G[self._plain_obj_to_orig_id_map[plain_obj], obj.id_] = coeff
                else:
                    raise ValueError("Unexpected obj")

        return G

    def get_obj_id_to_node_id_map(self):
        obj_id_to_node_id_map = {}

        id_ = 0
        for node_id, obj_list in enumerate(self.node_id_to_objs_list):
            for obj in obj_list:
                obj.id_ = id_
                obj_id_to_node_id_map[id_] = node_id

                id_ += 1

        return obj_id_to_node_id_map

    def get_num_original_objs(self):
        original_obj_set = set()
        for obj_list in self.node_id_to_objs_list:
            for obj in obj_list:
                if isinstance(obj, PlainObj):
                    original_obj_set.add(obj)
        num_original_objs = len(original_obj_set)

        check(
            num_original_objs > 0,
            f"Must be larger than 0; num_original_objs= {num_original_objs}",
        )

        return num_original_objs

    def get_plain_obj_to_orig_id_map(self):
        plain_obj_to_orig_id_map = {}

        orig_id = 0
        for obj_list in self.node_id_to_objs_list:
            for obj in obj_list:
                if obj in plain_obj_to_orig_id_map:
                    continue
                if isinstance(obj, PlainObj) is False:
                    continue

                if obj.orig_id:
                    plain_obj_to_orig_id_map[obj] = obj.orig_id
                else:
                    plain_obj_to_orig_id_map[obj] = orig_id
                    orig_id += 1

        return plain_obj_to_orig_id_map
