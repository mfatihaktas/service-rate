import numpy as np


def get_m_G__obj_bucket_m(k, bucket_objdesc_list):
  num_objs = sum([len(objdesc_list) for objdesc_list in bucket_objdesc_list] )
  G = np.zeros((k, num_objs))
  obj_to_bucket_map = {}
  
  obj = 0
  for bucket, objdesc_l in enumerate(bucket_objdesc_list):
    for objdesc in objdesc_l:
      for part in objdesc:
        G[part[0], obj] = part[1]
      obj_to_bucket_map[obj] = bucket
      obj += 1
  
  return bucket + 1, G, obj_to_bucket_map
