# get chamfer distance map from image segmentation

import numpy as np
import scipy

def get_chamfer(mask, scale=0.25): 
  
  h, w = mask.shape
  mask_tmp = np.zeros((h,w,3))
  mask_tmp[:,:,:] = np.reshape(mask, [h, w, 1])
  mask = mask_tmp
  mask_small = scipy.misc.imresize(mask, scale)
  
  h_ = int(scale * h)
  w_ = int(scale * w)

  non_occupied_pixels = []
  occupied_pixels = []
  for i in range(h_):
    for j in range(w_):
      pixel = mask_small[i,j,0]
      if pixel > 0.5:
        occupied_pixels.append([i, j])
      else:
        non_occupied_pixels.append([i,j]) 
  if len(occupied_pixels) == 0:
    return h_ * np.ones((h_, w_)), h_, w_ 
  if len(non_occupied_pixels) == 0:
    return np.zeros((h_, w_)), h_, w_ 
 
  subsample = np.random.permutation(len(occupied_pixels))[: len(occupied_pixels)/4]  
  non_occupied_pixels = np.array(non_occupied_pixels) 
  occupied_pixels = np.array(occupied_pixels)
  occupied_pixels = occupied_pixels[np.array(subsample)]  
  
  x2 = np.reshape(np.sum(np.square(non_occupied_pixels), axis=1), [-1, 1])
  y2 = np.reshape(np.sum(np.square(occupied_pixels), axis=1), [1, -1])  
  xy = np.matmul(non_occupied_pixels, occupied_pixels.T)  
  dist = np.min(x2 - 2* xy + y2, axis = 1) 
  chamfer = np.zeros((h_,w_),dtype=np.float32)
  for pos_id in range(non_occupied_pixels.shape[0]):
    pos = non_occupied_pixels[pos_id,:] 
    chamfer[pos[0], pos[1]] = np.sqrt(dist[pos_id])

  return chamfer, h_, w_
