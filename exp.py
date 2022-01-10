from bucket_conf import BucketConfInspector_wCode
from bucket_utils import get_m_G__obj_bucket_m

from debug_utils import *

def example(k):
  if k == 2:
    # bucket__objdesc_l_l = \
    # [[((0, 1),), ((1, 1),) ],
    #   [((1, 1),), ((0, 1),) ] ]
    bucket__objdesc_l_l = \
     [[((0, 1),), ((1, 1),) ],
      [((0, 1), (1, 1)), ((0, 1), (1, 2)) ] ]
    
    ## Example with Caroline
    d = 2
    # bucket__objdesc_l_l = \
    #   [[((0, 1),) ],
    #    [((0, 1),) ],
    #    [((0, 1),) ],
    #    [((0, 1),) ],
    #    [((0, 1),) ],
    #    [((1, 1),) ],
    #    [((1, 1),) ],
    #    [((1, 1),) ],
    #    [((1, 1),) ],
    #    [((0, 1), (1, 1)) ],
    #    [((0, 1), (1, 2)) ] ]

  elif k == 3:
    d = 2
    # if d == 1:
    #   bucket__objdesc_l_l = \
    #   [[((0, 1),)],
    #     [((1, 1),)],
    #     [((2, 1),)] ]
    # elif d == 2:
    #   bucket__objdesc_l_l = \
    #   [[((0, 1),), ((2, 1),) ],
    #     [((1, 1),), ((0, 1),) ],
    #     [((2, 1),), ((1, 1),) ] ]
    # elif d == 3:
    #   bucket__objdesc_l_l = \
    #   [[((0, 1),), ((2, 1),), ((1, 1),) ],
    #     [((1, 1),), ((0, 1),), ((2, 1),) ],
    #     [((2, 1),), ((1, 1),), ((0, 1),) ] ]
    ## Balanced coding
    # bucket__objdesc_l_l = \
    #   [[((0, 1),), ((1, 1), (2, 1)) ],
    #     [((1, 1),), ((0, 1), (2, 1)) ],
    #     [((2, 1),), ((0, 1), (1, 1)) ] ]
    ## Unbalanced coding
    # bucket__objdesc_l_l = \
    # [[((0, 1),), ((1, 1),) ],
    #   [((2, 1),), ((0, 1), (1, 1)) ],
    #   [((1, 1), (2, 1)), ((0, 1), (2, 1)) ] ]
    ## Simplex over 4 nodes
    # bucket__objdesc_l_l = \
    # [[((0, 1),), ((1, 1),) ],
    #   [((2, 1),), ((0, 1), (1, 1)) ],
    #   [((1, 1), (2, 1)), ((0, 1), (2, 1)) ],
    #   [((0, 1), (1, 1), (2, 1)) ] ]
    # bucket__objdesc_l_l = \
    # [[((0, 1),), ((1, 1), (2, 1)) ],
    #   [((1, 1),), ((0, 1), (2, 1)) ],
    #   [((2, 1),), ((0, 1), (1, 1)) ],
    #   [((0, 1), (1, 1), (2, 1)) ] ]
  
  m, G, obj_bucket_m = get_m_G__obj_bucket_m(k, bucket__objdesc_l_l)
  log(INFO, "", k=k, m=m, G=G, obj_bucket_m=obj_bucket_m)
  C = 1
  cf = BucketConfInspector_wCode(m, C, G, obj_bucket_m)
  log(DEBUG, "", cf=cf, to_sysrepr=cf.to_sysrepr())
  
  cf.plot_cap(d)

def plot_capregion_reed_muller():
  k = 4
  bucket__objdesc_l_l = \
    [[((3, 1),) ],
     [((2, 1), (3, 1)) ],
     [((1, 1), (3, 1)) ],
     [((1, 1), (2, 1), (3, 1)) ],
     [((0, 1), (3, 1)) ],
     [((0, 1), (2, 1), (3, 1)) ],
     [((0, 1), (1, 1), (3, 1)) ],
     [((0, 1), (1, 1), (2, 1), (3, 1)) ] ]
  n, G, obj_bucket_m = get_m_G__obj_bucket_m(k, bucket__objdesc_l_l)
  log(INFO, "G=\n{}".format(pprint.pformat(list(G) ) ), n=n, obj_bucket_m=obj_bucket_m)
  C = 1
  bci = BucketConfInspector_wCode(n, C, G, obj_bucket_m)
  bci.plot_cap_2d_when_k_g_2()
  log(INFO, "done.")

def checking_plausible_regular_balanced_dchoice_wxors():
  k = 8
  bucket__objdesc_l_l = \
    [[((0, 1),), ((6, 1), (7, 1)) ],
     [((1, 1),), ((3, 1), (7, 1)) ],
     [((2, 1),), ((0, 1), (1, 1)) ],
     [((3, 1),), ((0, 1), (5, 1)) ],
     [((4, 1),), ((2, 1), (3, 1)) ],
     [((5, 1),), ((2, 1), (6, 1)) ],
     [((6, 1),), ((4, 1), (5, 1)) ],
     [((7, 1),), ((1, 1), (4, 1)) ] ]
  m, G, obj_bucket_m = get_m_G__obj_bucket_m(k, bucket__objdesc_l_l)
  log(INFO, "G=\n{}".format(pprint.pformat(list(G) ) ), m=m, obj_bucket_m=obj_bucket_m)
  C = 1
  cf = BucketConfInspector_wCode(m, C, G, obj_bucket_m)
  # blog(cf=cf, to_sysrepr=cf.to_sysrepr() )
  
if __name__ == "__main__":
  example(k = 2)
  
  # checking_plausible_regular_balanced_dchoice_wxors()
  # plot_capregion_reed_muller()
