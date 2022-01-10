import numpy as np

def get_m_G__obj_bucket_m(k, bucket__objdesc_l_l):
  nobj = sum([len(objdesc_l) for objdesc_l in bucket__objdesc_l_l] )
  G = np.zeros((k, nobj))
  obj_bucket_m = {}
  
  obj = 0
  for bucket, objdesc_l in enumerate(bucket__objdesc_l_l):
    for objdesc in objdesc_l:
      for part in objdesc:
        G[part[0], obj] = part[1]
      obj_bucket_m[obj] = bucket
      obj += 1
  
  return bucket + 1, G, obj_bucket_m
