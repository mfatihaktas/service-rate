import pytest

from src import service_rate, storage_scheme
from src.debug_utils import *


node_id_objs_list = [
    [storage_scheme.PlainObj(id_str="a")],
    [storage_scheme.PlainObj(id_str="b")],
    [storage_scheme.PlainObj(id_str="c")],
    [storage_scheme.PlainObj(id_str="a")],
    [storage_scheme.PlainObj(id_str="b")],
    [storage_scheme.PlainObj(id_str="c")],

    [
        storage_scheme.CodedObj(
            coeff_obj_list=[
                (1, storage_scheme.PlainObj(id_str="a")),
                (1, storage_scheme.PlainObj(id_str="b")),
            ]
        )
    ],

    [
        storage_scheme.CodedObj(
            coeff_obj_list=[
                (1, storage_scheme.PlainObj(id_str="a")),
                (1, storage_scheme.PlainObj(id_str="c")),
            ]
        )
    ],

    [
        storage_scheme.CodedObj(
            coeff_obj_list=[
                (1, storage_scheme.PlainObj(id_str="b")),
                (1, storage_scheme.PlainObj(id_str="c")),
            ]
        )
    ],
]


@pytest.fixture(
    scope="function",
    params=[
        False,
        True,
    ],
)
def redundancy_w_two_xors(request):
    return request.param


def test_service_rate_inspector_on_redundancy_w_two_xors(redundancy_w_two_xors: bool):
    scheme = storage_scheme.StorageScheme(node_id_objs_list)
    log(DEBUG, "", storage_scheme=scheme)

    inspector = service_rate.ServiceRateInspector(
        m=len(node_id_objs_list),
        C=1,
        G=scheme.obj_encoding_matrix,
        obj_to_node_id_map=scheme.obj_id_to_node_id_map,
        redundancy_w_two_xors=redundancy_w_two_xors,
    )
