from src import service_rate, storage_scheme
from src.debug_utils import *


def test_w_a_b_c_ab_ac_bc():
    node_id_objs_list = storage_scheme.name_to_node_objs_list_map["a_b_c_a+b_a+c_b+c"]

    scheme = storage_scheme.StorageScheme(node_id_objs_list)
    log(DEBUG, "", storage_scheme=scheme)

    inspector = service_rate.ServiceRateInspector(
        m=len(node_id_objs_list),
        C=1,
        G=scheme.obj_encoding_matrix,
        obj_to_node_id_map=scheme.obj_id_to_node_id_map,
        redundancy_w_two_xors=True,
    )

    assert inspector.is_in_cap_region([2, 0, 1]) == True
    assert inspector.is_in_cap_region([1.5, 0.5, 1]) == True
    assert inspector.is_in_cap_region([1.5, 0.6, 0]) == True
    assert inspector.is_in_cap_region([1.3, 0.7, 1]) == True
    assert inspector.is_in_cap_region([2, 1, 1]) == False
    assert inspector.is_in_cap_region([1.5, 0.6, 1]) == False
