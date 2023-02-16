from src.storage_opt.access_graph import SysObject, XORedObject


NUM_OBJS_TO_MAX_DEMAND_TO_OBJ_TO_NUM_COPIES_MAP = {
    3: {
        1: {
            SysObject(symbol=0): 1,
            SysObject(symbol=1): 1,
            SysObject(symbol=2): 1,
            XORedObject(symbols=(0, 1)): 0,
            XORedObject(symbols=(0, 2)): 0,
            XORedObject(symbols=(1, 2)): 0,
            XORedObject(symbols=(0, 1, 2)): 0
        },

        2: {
            SysObject(symbol=1): 1,
            SysObject(symbol=2): 2,
            SysObject(symbol=0): 1,
            XORedObject(symbols=(0, 1)): 1,
            XORedObject(symbols=(0, 1, 2)): 0,
            XORedObject(symbols=(1, 2)): 0,
            XORedObject(symbols=(0, 2)): 0
        },

        3: {
            SysObject(symbol=0): 1,
            SysObject(symbol=2): 1,
            SysObject(symbol=1): 1,
            XORedObject(symbols=(0, 1)): 1,
            XORedObject(symbols=(0, 2)): 1,
            XORedObject(symbols=(1, 2)): 1,
            XORedObject(symbols=(0, 1, 2)): 0
        },

        4: {
            SysObject(symbol=0): 1,
            SysObject(symbol=1): 1,
            SysObject(symbol=2): 1,
            XORedObject(symbols=(0, 1)): 1,
            XORedObject(symbols=(1, 2)): 1,
            XORedObject(symbols=(0, 2)): 1,
            XORedObject(symbols=(0, 1, 2)): 1
        },

        5: {
            SysObject(symbol=0): 2,
            SysObject(symbol=1): 2,
            SysObject(symbol=2): 2,
            XORedObject(symbols=(0, 1, 2)): 1,
            XORedObject(symbols=(0, 2)): 1,
            XORedObject(symbols=(0, 1)): 1,
            XORedObject(symbols=(1, 2)): 1
        },

        6: {
            SysObject(symbol=2): 3,
            SysObject(symbol=1): 2,
            SysObject(symbol=0): 2,
            XORedObject(symbols=(0, 1)): 2,
            XORedObject(symbols=(1, 2)): 1,
            XORedObject(symbols=(0, 1, 2)): 1,
            XORedObject(symbols=(0, 2)): 1
        },

        7: {
            SysObject(symbol=1): 2,
            SysObject(symbol=0): 2,
            SysObject(symbol=2): 2,
            XORedObject(symbols=(0, 1)): 2,
            XORedObject(symbols=(0, 2)): 2,
            XORedObject(symbols=(0, 1, 2)): 1,
            XORedObject(symbols=(1, 2)): 2
        },

        8: {
            SysObject(symbol=2): 2,
            SysObject(symbol=0): 2,
            SysObject(symbol=1): 2,
            XORedObject(symbols=(1, 2)): 2,
            XORedObject(symbols=(0, 2)): 2,
            XORedObject(symbols=(0, 1, 2)): 2,
            XORedObject(symbols=(0, 1)): 2
        },

        9: {
            SysObject(symbol=0): 3,
            SysObject(symbol=1): 3,
            SysObject(symbol=2): 3,
            XORedObject(symbols=(0, 1)): 2,
            XORedObject(symbols=(0, 1, 2)): 2,
            XORedObject(symbols=(0, 2)): 2,
            XORedObject(symbols=(1, 2)): 2
        },

        10: {
            SysObject(symbol=1): 3,
            SysObject(symbol=2): 4,
            SysObject(symbol=0): 3,
            XORedObject(symbols=(0, 2)): 2,
            XORedObject(symbols=(1, 2)): 2,
            XORedObject(symbols=(0, 1, 2)): 2,
            XORedObject(symbols=(0, 1)): 3
        },

        11: {
            SysObject(symbol=2): 3,
            SysObject(symbol=0): 3,
            SysObject(symbol=1): 3,
            XORedObject(symbols=(1, 2)): 3,
            XORedObject(symbols=(0, 2)): 3,
            XORedObject(symbols=(0, 1, 2)): 2,
            XORedObject(symbols=(0, 1)): 3
        },

        12: {
            SysObject(symbol=0): 3,
            SysObject(symbol=2): 3,
            SysObject(symbol=1): 3,
            XORedObject(symbols=(1, 2)): 3,
            XORedObject(symbols=(0, 2)): 3,
            XORedObject(symbols=(0, 1, 2)): 3,
            XORedObject(symbols=(0, 1)): 3
        },

        13: {
            SysObject(symbol=1): 4,
            SysObject(symbol=2): 4,
            SysObject(symbol=0): 4,
            XORedObject(symbols=(1, 2)): 3,
            XORedObject(symbols=(0, 2)): 3,
            XORedObject(symbols=(0, 1, 2)): 3,
            XORedObject(symbols=(0, 1)): 3
        },

        14: {
            SysObject(symbol=0): 4,
            SysObject(symbol=2): 5,
            SysObject(symbol=1): 4,
            XORedObject(symbols=(0, 1)): 4,
            XORedObject(symbols=(0, 2)): 3,
            XORedObject(symbols=(0, 1, 2)): 3,
            XORedObject(symbols=(1, 2)): 3
        },

        15: {
            SysObject(symbol=2): 4,
            SysObject(symbol=1): 4,
            SysObject(symbol=0): 4,
            XORedObject(symbols=(1, 2)): 4,
            XORedObject(symbols=(0, 2)): 4,
            XORedObject(symbols=(0, 1, 2)): 3,
            XORedObject(symbols=(0, 1)): 4
        },

        16: {
            SysObject(symbol=0): 4,
            SysObject(symbol=1): 4,
            SysObject(symbol=2): 4,
            XORedObject(symbols=(0, 1, 2)): 4,
            XORedObject(symbols=(0, 2)): 4,
            XORedObject(symbols=(1, 2)): 4,
            XORedObject(symbols=(0, 1)): 4
        },

        17: {
            SysObject(symbol=2): 5,
            SysObject(symbol=0): 5,
            SysObject(symbol=1): 5,
            XORedObject(symbols=(1, 2)): 4,
            XORedObject(symbols=(0, 1, 2)): 4,
            XORedObject(symbols=(0, 2)): 4,
            XORedObject(symbols=(0, 1)): 4
        },

        18: {
            SysObject(symbol=2): 6,
            SysObject(symbol=0): 5,
            SysObject(symbol=1): 5,
            XORedObject(symbols=(1, 2)): 4,
            XORedObject(symbols=(0, 2)): 4,
            XORedObject(symbols=(0, 1)): 5,
            XORedObject(symbols=(0, 1, 2)): 4
        },

        19: {
            SysObject(symbol=2): 5,
            SysObject(symbol=0): 5,
            SysObject(symbol=1): 5,
            XORedObject(symbols=(0, 2)): 5,
            XORedObject(symbols=(1, 2)): 5,
            XORedObject(symbols=(0, 1, 2)): 4,
            XORedObject(symbols=(0, 1)): 5
        },

        20: {
            SysObject(symbol=1): 5,
            SysObject(symbol=2): 5,
            SysObject(symbol=0): 5,
            XORedObject(symbols=(0, 2)): 5,
            XORedObject(symbols=(1, 2)): 5,
            XORedObject(symbols=(0, 1, 2)): 5,
            XORedObject(symbols=(0, 1)): 5
        }
    },

    4: {
        1: {
            SysObject(symbol=0): 1,
            SysObject(symbol=1): 1,
            SysObject(symbol=2): 1,
            SysObject(symbol=3): 1,
            XORedObject(symbols=(0, 1)): 0,
            XORedObject(symbols=(0, 2)): 0,
            XORedObject(symbols=(0, 3)): 0,
            XORedObject(symbols=(1, 2)): 0,
            XORedObject(symbols=(1, 3)): 0,
            XORedObject(symbols=(2, 3)): 0,
            XORedObject(symbols=(0, 1, 2)): 0,
            XORedObject(symbols=(0, 1, 3)): 0,
            XORedObject(symbols=(0, 2, 3)): 0,
            XORedObject(symbols=(1, 2, 3)): 0,
            XORedObject(symbols=(0, 1, 2, 3)): 0
        },

        2: {
            SysObject(symbol=2): 1,
            SysObject(symbol=0): 1,
            SysObject(symbol=3): 1,
            SysObject(symbol=1): 1,
            XORedObject(symbols=(1, 2, 3)): 0,
            XORedObject(symbols=(0, 2, 3)): 0,
            XORedObject(symbols=(0, 1, 3)): 0,
            XORedObject(symbols=(0, 1, 2)): 0,
            XORedObject(symbols=(2, 3)): 0,
            XORedObject(symbols=(1, 3)): 1,
            XORedObject(symbols=(0, 3)): 0,
            XORedObject(symbols=(0, 1)): 0,
            XORedObject(symbols=(1, 2)): 0,
            XORedObject(symbols=(0, 2)): 1,
            XORedObject(symbols=(0, 1, 2, 3)): 0
        },

        3: {
            SysObject(symbol=0): 1,
            SysObject(symbol=1): 1,
            SysObject(symbol=3): 1,
            SysObject(symbol=2): 1,
            XORedObject(symbols=(0, 1)): 1,
            XORedObject(symbols=(0, 2, 3)): 0,
            XORedObject(symbols=(1, 3)): 0,
            XORedObject(symbols=(0, 2)): 0,
            XORedObject(symbols=(1, 2, 3)): 0,
            XORedObject(symbols=(2, 3)): 1,
            XORedObject(symbols=(0, 3)): 1,
            XORedObject(symbols=(0, 1, 2, 3)): 0,
            XORedObject(symbols=(0, 1, 3)): 0,
            XORedObject(symbols=(1, 2)): 1,
            XORedObject(symbols=(0, 1, 2)): 0
        },

        4: {
            SysObject(symbol=0): 1,
            SysObject(symbol=3): 1,
            SysObject(symbol=2): 1,
            SysObject(symbol=1): 1,
            XORedObject(symbols=(0, 1)): 1,
            XORedObject(symbols=(2, 3)): 1,
            XORedObject(symbols=(1, 2)): 1,
            XORedObject(symbols=(0, 3)): 1,
            XORedObject(symbols=(1, 3)): 1,
            XORedObject(symbols=(0, 2)): 1,
            XORedObject(symbols=(0, 1, 2, 3)): 0,
            XORedObject(symbols=(1, 2, 3)): 0,
            XORedObject(symbols=(0, 1, 3)): 0,
            XORedObject(symbols=(0, 1, 2)): 0,
            XORedObject(symbols=(0, 2, 3)): 0
        },

        5: {
            SysObject(symbol=0): 2,
            SysObject(symbol=1): 1,
            SysObject(symbol=3): 1,
            SysObject(symbol=2): 1,
            XORedObject(symbols=(0, 1)): 1,
            XORedObject(symbols=(0, 1, 3)): 0,
            XORedObject(symbols=(2, 3)): 1,
            XORedObject(symbols=(1, 3)): 1,
            XORedObject(symbols=(0, 1, 2)): 0,
            XORedObject(symbols=(0, 2)): 1,
            XORedObject(symbols=(0, 3)): 1,
            XORedObject(symbols=(1, 2)): 1,
            XORedObject(symbols=(0, 1, 2, 3)): 0,
            XORedObject(symbols=(0, 2, 3)): 0,
            XORedObject(symbols=(1, 2, 3)): 1
        },

        6: {
            SysObject(symbol=0): 1,
            SysObject(symbol=2): 1,
            SysObject(symbol=3): 1,
            SysObject(symbol=1): 1,
            XORedObject(symbols=(2, 3)): 1,
            XORedObject(symbols=(1, 2, 3)): 1,
            XORedObject(symbols=(1, 2)): 1,
            XORedObject(symbols=(0, 1)): 1,
            XORedObject(symbols=(1, 3)): 1,
            XORedObject(symbols=(0, 1, 2, 3)): 0,
            XORedObject(symbols=(0, 3)): 1,
            XORedObject(symbols=(0, 2, 3)): 0,
            XORedObject(symbols=(0, 1, 3)): 1,
            XORedObject(symbols=(0, 1, 2)): 1,
            XORedObject(symbols=(0, 2)): 1
        },

        7: {
            SysObject(symbol=0): 1,
            SysObject(symbol=1): 1,
            SysObject(symbol=2): 1,
            SysObject(symbol=3): 1,
            XORedObject(symbols=(0, 1)): 1,
            XORedObject(symbols=(1, 3)): 1,
            XORedObject(symbols=(2, 3)): 1,
            XORedObject(symbols=(0, 1, 2)): 1,
            XORedObject(symbols=(0, 2, 3)): 1,
            XORedObject(symbols=(0, 1, 3)): 1,
            XORedObject(symbols=(1, 2)): 1,
            XORedObject(symbols=(1, 2, 3)): 1,
            XORedObject(symbols=(0, 1, 2, 3)): 0,
            XORedObject(symbols=(0, 2)): 1,
            XORedObject(symbols=(0, 3)): 1
        },

        8: {
            SysObject(symbol=0): 1,
            SysObject(symbol=1): 1,
            SysObject(symbol=3): 1,
            SysObject(symbol=2): 1,
            XORedObject(symbols=(0, 1)): 1,
            XORedObject(symbols=(0, 2)): 1,
            XORedObject(symbols=(0, 1, 2)): 1,
            XORedObject(symbols=(0, 1, 3)): 1,
            XORedObject(symbols=(2, 3)): 1,
            XORedObject(symbols=(0, 1, 2, 3)): 1,
            XORedObject(symbols=(1, 2, 3)): 1,
            XORedObject(symbols=(0, 2, 3)): 1,
            XORedObject(symbols=(1, 3)): 1,
            XORedObject(symbols=(0, 3)): 1,
            XORedObject(symbols=(1, 2)): 1
        },

        9: {
            SysObject(symbol=0): 2,
            SysObject(symbol=1): 2,
            SysObject(symbol=3): 2,
            SysObject(symbol=2): 2,
            XORedObject(symbols=(0, 1)): 1,
            XORedObject(symbols=(0, 1, 3)): 1,
            XORedObject(symbols=(0, 2, 3)): 1,
            XORedObject(symbols=(1, 2, 3)): 1,
            XORedObject(symbols=(0, 1, 2, 3)): 1,
            XORedObject(symbols=(2, 3)): 1,
            XORedObject(symbols=(0, 1, 2)): 1,
            XORedObject(symbols=(0, 2)): 1,
            XORedObject(symbols=(1, 2)): 1,
            XORedObject(symbols=(1, 3)): 1,
            XORedObject(symbols=(0, 3)): 1
        },

        10: {
            SysObject(symbol=0): 2,
            SysObject(symbol=3): 2,
            SysObject(symbol=1): 2,
            SysObject(symbol=2): 2,
            XORedObject(symbols=(1, 3)): 1,
            XORedObject(symbols=(0, 3)): 1,
            XORedObject(symbols=(1, 2)): 1,
            XORedObject(symbols=(0, 1, 3)): 1,
            XORedObject(symbols=(2, 3)): 2,
            XORedObject(symbols=(0, 1)): 2,
            XORedObject(symbols=(1, 2, 3)): 1,
            XORedObject(symbols=(0, 2)): 1,
            XORedObject(symbols=(0, 1, 2)): 1,
            XORedObject(symbols=(0, 1, 2, 3)): 1,
            XORedObject(symbols=(0, 2, 3)): 1
        },

        11: {
            SysObject(symbol=0): 2,
             SysObject(symbol=2): 2,
             SysObject(symbol=1): 2,
             SysObject(symbol=3): 2,
             XORedObject(symbols=(0, 2)): 2,
             XORedObject(symbols=(0, 1, 2)): 1,
             XORedObject(symbols=(1, 2)): 2,
             XORedObject(symbols=(0, 2, 3)): 1,
             XORedObject(symbols=(1, 3)): 2,
             XORedObject(symbols=(2, 3)): 1,
             XORedObject(symbols=(1, 2, 3)): 1,
             XORedObject(symbols=(0, 1, 3)): 1,
             XORedObject(symbols=(0, 3)): 2,
             XORedObject(symbols=(0, 1, 2, 3)): 1,
             XORedObject(symbols=(0, 1)): 1
        },

        12: {
            SysObject(symbol=1): 2,
            SysObject(symbol=2): 2,
            SysObject(symbol=0): 2,
            SysObject(symbol=3): 2,
            XORedObject(symbols=(0, 2)): 2,
            XORedObject(symbols=(0, 1)): 2,
            XORedObject(symbols=(2, 3)): 2,
            XORedObject(symbols=(0, 1, 2, 3)): 1,
            XORedObject(symbols=(1, 3)): 2,
            XORedObject(symbols=(0, 1, 3)): 1,
            XORedObject(symbols=(0, 2, 3)): 1,
            XORedObject(symbols=(0, 1, 2)): 1,
            XORedObject(symbols=(1, 2, 3)): 1,
            XORedObject(symbols=(0, 3)): 2,
            XORedObject(symbols=(1, 2)): 2
        },

        13: {
            SysObject(symbol=2): 2,
            SysObject(symbol=0): 3,
            SysObject(symbol=3): 2,
            SysObject(symbol=1): 2,
            XORedObject(symbols=(0, 1)): 2,
            XORedObject(symbols=(1, 2)): 2,
            XORedObject(symbols=(0, 3)): 2,
            XORedObject(symbols=(0, 1, 2, 3)): 1,
            XORedObject(symbols=(1, 2, 3)): 2,
            XORedObject(symbols=(0, 1, 3)): 1,
            XORedObject(symbols=(0, 1, 2)): 1,
            XORedObject(symbols=(2, 3)): 2,
            XORedObject(symbols=(0, 2)): 2,
            XORedObject(symbols=(0, 2, 3)): 1,
            XORedObject(symbols=(1, 3)): 2
        },

        14: {
            SysObject(symbol=2): 2,
            SysObject(symbol=0): 2,
            SysObject(symbol=3): 2,
            SysObject(symbol=1): 2,
            XORedObject(symbols=(0, 1)): 2,
            XORedObject(symbols=(0, 2)): 2,
            XORedObject(symbols=(0, 3)): 2,
            XORedObject(symbols=(1, 3)): 2,
            XORedObject(symbols=(0, 1, 2, 3)): 0,
            XORedObject(symbols=(1, 2)): 2,
            XORedObject(symbols=(1, 2, 3)): 2,
            XORedObject(symbols=(0, 1, 3)): 2,
            XORedObject(symbols=(0, 2, 3)): 2,
            XORedObject(symbols=(0, 1, 2)): 2,
            XORedObject(symbols=(2, 3)): 2
        },

        15: {
            SysObject(symbol=0): 2,
            SysObject(symbol=3): 2,
            SysObject(symbol=2): 2,
            SysObject(symbol=1): 2,
            XORedObject(symbols=(2, 3)): 2,
            XORedObject(symbols=(0, 3)): 2,
            XORedObject(symbols=(1, 2)): 2,
            XORedObject(symbols=(0, 1)): 2,
            XORedObject(symbols=(0, 2)): 2,
            XORedObject(symbols=(0, 2, 3)): 2,
            XORedObject(symbols=(0, 1, 2)): 2,
            XORedObject(symbols=(0, 1, 3)): 2,
            XORedObject(symbols=(1, 2, 3)): 2,
            XORedObject(symbols=(0, 1, 2, 3)): 1,
            XORedObject(symbols=(1, 3)): 2
        },

        16: {
            SysObject(symbol=1): 2,
            SysObject(symbol=0): 2,
            SysObject(symbol=3): 2,
            SysObject(symbol=2): 2,
            XORedObject(symbols=(0, 2, 3)): 2,
            XORedObject(symbols=(1, 2, 3)): 2,
            XORedObject(symbols=(0, 1, 2, 3)): 2,
            XORedObject(symbols=(1, 2)): 2,
            XORedObject(symbols=(0, 1, 2)): 2,
            XORedObject(symbols=(2, 3)): 2,
            XORedObject(symbols=(0, 3)): 2,
            XORedObject(symbols=(0, 1)): 2,
            XORedObject(symbols=(1, 3)): 2,
            XORedObject(symbols=(0, 2)): 2,
            XORedObject(symbols=(0, 1, 3)): 2
        },

        17: {
            SysObject(symbol=2): 3,
            SysObject(symbol=1): 3,
            SysObject(symbol=0): 3,
            SysObject(symbol=3): 3,
            XORedObject(symbols=(0, 1)): 2,
            XORedObject(symbols=(0, 2, 3)): 2,
            XORedObject(symbols=(2, 3)): 2,
            XORedObject(symbols=(1, 2, 3)): 2,
            XORedObject(symbols=(0, 1, 3)): 2,
            XORedObject(symbols=(0, 1, 2)): 2,
            XORedObject(symbols=(0, 1, 2, 3)): 2,
            XORedObject(symbols=(1, 2)): 2,
            XORedObject(symbols=(0, 2)): 2,
            XORedObject(symbols=(0, 3)): 2,
            XORedObject(symbols=(1, 3)): 2
        },

        18: {
            SysObject(symbol=2): 3,
            SysObject(symbol=0): 3,
            SysObject(symbol=1): 3,
            SysObject(symbol=3): 3,
            XORedObject(symbols=(0, 2, 3)): 2,
            XORedObject(symbols=(0, 3)): 2,
            XORedObject(symbols=(2, 3)): 3,
            XORedObject(symbols=(0, 1, 2)): 2,
            XORedObject(symbols=(0, 1, 3)): 2,
            XORedObject(symbols=(0, 2)): 2,
            XORedObject(symbols=(0, 1)): 3,
            XORedObject(symbols=(1, 2)): 2,
            XORedObject(symbols=(1, 2, 3)): 2,
            XORedObject(symbols=(0, 1, 2, 3)): 2,
            XORedObject(symbols=(1, 3)): 2
        },

        19: {
            SysObject(symbol=1): 3,
            SysObject(symbol=0): 3,
            SysObject(symbol=3): 3,
            SysObject(symbol=2): 3,
            XORedObject(symbols=(0, 1)): 3,
            XORedObject(symbols=(0, 1, 3)): 2,
            XORedObject(symbols=(0, 1, 2, 3)): 2,
            XORedObject(symbols=(1, 2, 3)): 2,
            XORedObject(symbols=(0, 2, 3)): 2,
            XORedObject(symbols=(0, 1, 2)): 2,
            XORedObject(symbols=(1, 3)): 2,
            XORedObject(symbols=(1, 2)): 3,
            XORedObject(symbols=(2, 3)): 3,
            XORedObject(symbols=(0, 3)): 3,
            XORedObject(symbols=(0, 2)): 2
        },

        20: {
            SysObject(symbol=0): 3,
            SysObject(symbol=1): 3,
            SysObject(symbol=3): 3,
            SysObject(symbol=2): 3,
            XORedObject(symbols=(1, 3)): 3,
            XORedObject(symbols=(0, 2, 3)): 2,
            XORedObject(symbols=(2, 3)): 3,
            XORedObject(symbols=(1, 2)): 3,
            XORedObject(symbols=(0, 1, 3)): 2,
            XORedObject(symbols=(0, 1, 2, 3)): 2,
            XORedObject(symbols=(1, 2, 3)): 2,
            XORedObject(symbols=(0, 1, 2)): 2,
            XORedObject(symbols=(0, 1)): 3,
            XORedObject(symbols=(0, 2)): 3,
            XORedObject(symbols=(0, 3)): 3
        }
    },

    5: {

    },

    6: {

    },
}
