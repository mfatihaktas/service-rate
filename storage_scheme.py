import numpy

from debug_utils import *


class Obj:
    """Base object class"""

    def __init__(self):
        self._index = None

    @property
    def index(self):
        return self._index

    @index.setter
    def set_index(self, index):
        self._index = index


class PlainObj(Obj):
    def __init__(self, identifier: str):
        super().__init__()

        self.identifier = identifier

    def __hash__(self):
        return hash(self.identifier)

    def __repr__(self):
        return (
            "PlainObj( \n"
            f"\t identifier= {self.identifier} \n"
            f"\t index= {self.index}"
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
        id_tuple = (obj.identifier for _, obj in self.coeff_obj_list)
        return hash(id_tuple)

    def __repr__(self):
        s = "CodedObj( \n"
        for coeff, obj in self.coeff_obj_list:
            s += f"{coeff} x \n{obj} \n"
        s += ")"

        return s


name_to_node_objs_list_map = {
    "a_b_a+b": [
        [PlainObj(identifier="a")],
        [PlainObj(identifier="b")],
        [
            CodedObj(
                coeff_obj_list=[
                    (1, PlainObj(identifier="a")),
                    (1, PlainObj(identifier="b")),
                ]
            )
        ],
    ]
}


class StorageScheme:
    def __init__(self, node_objs_list: list[list[Obj]]):
        self._node_objs_list = node_objs_list

        # This refers to `k`
        self._num_original_objs = self.get_num_original_objs()
        self._total_num_objs = sum(len(obj_list) for obj_list in node_objs_list)

        self._plain_obj_to_orig_index_map = self.get_plain_obj_to_orig_index_map()

        # This refers to G
        self._obj_encoding_matrix = self.get_obj_encoding_matrix()

        self._obj_to_node_id_map = self.get_obj_to_node_id_map()

    def __repr__(self):
        s = "StorageScheme( \n"
        for node_id, obj_list in self.node_objs_list:
            s += f"node-{node_id}: [\n"
            for obj in obj_list:
                s += f"{obj} \n"
            s += "] \n"
        s += ") \n"

        return s

    @property
    def node_objs_list(self):
        return self._node_objs_list

    @property
    def obj_to_node_id_map(self):
        return self.obj_to_node_id_map

    @property
    def num_original_objs(self):
        return self._num_original_objs

    @property
    def total_num_objs(self):
        return self._total_num_objs

    def get_obj_encoding_matrix(self):
        G = numpy.zeros((self.num_original_objs, self.total_num_objs))

        index = 0
        for node, obj_list in enumerate(self.node_objs_list):
            for obj in obj_list:
                obj.set_index(index)
                if isinstance(obj, PlainObj):
                    G[self._plain_obj_to_orig_index_map[obj], index] = 1
                elif isinstance(obj, CodedObj):
                    for coeff, obj in obj.coeff_obj_list:
                        G[self._plain_obj_to_orig_index_map[obj], index] = coeff
                else:
                    raise ValueError("Unexpected obj")

                index += 1

    def get_obj_to_node_id_map(self):
        obj_to_node_id_map = {}

        for node_id, obj_list in enumerate(node_objs_list):
            for obj in obj_list:
                obj_to_node_id_map[obj] = node_id

        return obj_to_node_id_map

    def get_num_original_objs(self):
        original_obj_set = set()
        for obj_list in self.node_objs_list:
            for obj in obj_list:
                if isinstance(obj, PlainObj):
                    original_obj_set.add(obj)
        num_original_objs = len(original_obj_set)

        check(
            num_original_objs > 0,
            f"Must be larger than 0; num_original_objs= {num_original_objs}",
        )

        return num_original_objs

    def get_plain_obj_to_orig_index_map(self):
        plain_obj_to_orig_index_map = {}

        index = 0
        for obj_list in self.node_objs_list:
            for obj in obj_list:
                if isinstance(obj, PlainObj) and obj not in plain_obj_to_orig_index_map:
                    plain_obj_to_orig_index_map[obj] = index
                    index += 1
