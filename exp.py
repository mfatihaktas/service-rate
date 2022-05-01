from debug_utils import *
from service_rate import ServiceRateInspector
from storage_scheme import Obj, StorageScheme, name_to_node_objs_list_map


def run(node_objs_list: list[list[Obj]]):
    log(DEBUG, "Started;", node_objs_list=node_objs_list)

    scheme = StorageScheme(node_objs_list)
    log(DEBUG, "", storage_scheme=scheme)

    inspector = ServiceRateInspector(
        m=len(node_objs_list),
        C=1,
        G=scheme.obj_encoding_matrix,
        obj_to_node_id_map=scheme.obj_id_to_node_id_map,
    )
    inspector.plot_cap_2d()


if __name__ == "__main__":
    # run(name_to_node_objs_list_map["a_b_a+b"])
    # run(name_to_node_objs_list_map["a_a_b_b"])
    run(name_to_node_objs_list_map["a_a_a_b_a+b_a+2b"])
