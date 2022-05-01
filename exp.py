from debug_utils import *
from service_rate import ServiceRateInspector
from storage_scheme import Obj, StorageScheme, name_to_node_objs_list_map


def run(node_objs_list: list[list[Obj]]):
    scheme = StorageScheme(node_objs_list)
    inspector = ServiceRateInspector(
        m=len(node_objs_list),
        C=1,
        G=scheme.plain_obj_to_orig_index_map,
        obj_to_node_id_map=scheme.obj_to_node_id_map,
    )
    inspector.plot_cap_2d(d)


if __name__ == "__main__":
    run(name_to_node_objs_list_map["a_b_a+b"])
