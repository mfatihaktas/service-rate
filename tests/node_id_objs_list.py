from src import (
    storage_scheme as storage_scheme_module,
)


node_id_objs_list_1 = [
    [storage_scheme_module.PlainObj(id_str="a")],
    [storage_scheme_module.PlainObj(id_str="b")],
    [storage_scheme_module.PlainObj(id_str="c")],
    [storage_scheme_module.PlainObj(id_str="a")],
    [storage_scheme_module.PlainObj(id_str="b")],
    [storage_scheme_module.PlainObj(id_str="c")],

    [
        storage_scheme_module.CodedObj(
            coeff_obj_list=[
                (1, storage_scheme_module.PlainObj(id_str="a")),
                (1, storage_scheme_module.PlainObj(id_str="b")),
            ]
        )
    ],

    [
        storage_scheme_module.CodedObj(
            coeff_obj_list=[
                (1, storage_scheme_module.PlainObj(id_str="a")),
                (1, storage_scheme_module.PlainObj(id_str="c")),
            ]
        )
    ],

    [
        storage_scheme_module.CodedObj(
            coeff_obj_list=[
                (1, storage_scheme_module.PlainObj(id_str="b")),
                (1, storage_scheme_module.PlainObj(id_str="c")),
            ]
        )
    ],
]
