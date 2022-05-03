import pytest

from debug_utils import *
from service_rate import ServiceRateInspector
from storage_scheme import Obj, StorageScheme, name_to_node_objs_list_map

# @pytest.fixture(
#     scope="function",
#     params=[
#         "a_b_a+b",
#     ],
# )
# def scheme_names(request):
#     return request.param


# @pytest.fixture(scope="session")
# def dataset_bw_insurance_small():
#     dataset = Dataset(
#         model_name="bw_insurance",
#         model_type=MODEL_TYPE,
#         dataset_folder_path="./_data/labelbox_schemas/bw_insurance_small",
#         datarow_group_class=datarow_group.DataRowGroup_wSelectFirstDataRow,
#     )

#     return dataset


def test_is_in_cap_region():
    node_id_objs_list = name_to_node_objs_list_map["a,b_a,b"]

    scheme = StorageScheme(node_id_objs_list)
    log(DEBUG, "", storage_scheme=scheme)

    inspector = ServiceRateInspector(
        m=len(node_id_objs_list),
        C=1,
        G=scheme.obj_encoding_matrix,
        obj_to_node_id_map=scheme.obj_id_to_node_id_map,
    )

    assert inspector.is_in_cap_region([1, 1]) == True
    assert inspector.is_in_cap_region([1.5, 0.35]) == True
    assert inspector.is_in_cap_region([0.25, 2]) == False
    assert inspector.is_in_cap_region([1.8, 1.4]) == False
    assert inspector.is_in_cap_region([2.1, 2]) == False
    assert inspector.is_in_cap_region([1, 2.3]) == False
