#from opendr.renderer import ColoredRenderer
#from opendr.util_tests import get_earthmesh
#from opendr.camera import ProjectPoints
#from opendr.lighting import LambertianPointLight
#from opendr.simple import *
#from smpl_webuser.serialization import load_model
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

import tensorflow as tf
kEpsilon = 1e-8


def tf_get_visibility(tf_v, tf_project, h, w):
  # project_i: batch_size x mesh_num x 2
  batch_size, mesh_num, _  = tf_v.get_shape().as_list()
 
  project_i = tf.cast(tf.round(tf_project), 'int32')
  x, y = tf.split(project_i, 2, 2)
  x = tf.clip_by_value(x, 0, w - 1)
  y = tf.clip_by_value(y, 0, h - 1)


  # batch_size x mesh_num
  distance = tf.reshape(tf.slice(tf_v, [0,0,2], [-1, -1, 1]), [-1])
  
  batch_id = tf.reshape(tf.tile(tf.reshape(tf.range(batch_size), [-1, 1]), [1, mesh_num]), [-1])
  mesh_id = tf.reshape(tf.tile(tf.reshape(tf.range(mesh_num), [1, -1]), [batch_size, 1]), [-1]) 
 
  # assign value into a (bxm) x (wxh) matrix
  first_id  = batch_id * mesh_num + mesh_id
  second_id = tf.reshape(y * w + x, [-1])

  dis_map = tf.sparse_to_dense(tf.stack([first_id, second_id], 1), [batch_size*mesh_num, h*w], distance, 100000)
  
  # concat a dummay dimesion in case for no preojection on a pixel
  dis_map = tf.concat([tf.reshape(dis_map, [batch_size, mesh_num, -1]), 99999 * tf.ones([batch_size, 1, w*h])], 1)
  # batch * (128*128)
  arg_min_dis = tf.argmin(dis_map, 1)
  arg_min_dis_tmp = arg_min_dis 
  arg_min_dis = tf.unstack(arg_min_dis, batch_size, 0)  
  onehot = []
  for id in range(batch_size):
    arg_min_list, _ = tf.unique(tf.reshape(arg_min_dis[id], [-1]))
    #arg_min_list, _ = tf.nn.top_k(arg_min_list, arg_min_list.get_shape().as_list()[0])
    oneh = tf.sparse_to_dense(arg_min_list, [mesh_num+ 1], 1.0, 0, validate_indices=False)
    onehot.append(oneh)
 
  visibility = tf.slice(tf.stack(onehot), [0, 0], [-1, mesh_num])
  return visibility, arg_min_dis_tmp
def norm(n):
  dim = len(n.get_shape().as_list())
  return tf.sqrt(tf.reduce_sum(tf.square(n), dim-1))
def normalize(n):
  return n/tf.expand_dims(norm(n), [-1])
def tf_get_visibility_raycast(tf_v, f, reduce_step=4):
  # tf_v: batch_size x mesh_num x 3
  # f: num_faces x 3
  batch_size, mesh_num, _  = tf_v.get_shape().as_list()
  num_faces = f.shape[0] 
  #reduced_f = tf.constant(f[np.arange(0, num_faces, 3)], tf.int32)
  f = tf.constant(f, tf.int32)
  idx0, idx1, idx2 = tf.unstack(f, 3, 1)
  idx0 = tf.tile(tf.reshape(tf.range(0, batch_size), [-1, 1]), [1, num_faces]) * mesh_num + tf.reshape(idx0, [1, -1])
  idx1 = tf.tile(tf.reshape(tf.range(0, batch_size), [-1, 1]), [1, num_faces]) * mesh_num + tf.reshape(idx1, [1, -1])
  idx2 = tf.tile(tf.reshape(tf.range(0, batch_size), [-1, 1]), [1, num_faces]) * mesh_num + tf.reshape(idx2, [1, -1])
  # batch x #face(13776) x 3 
  v0 = tf.gather(tf.reshape(tf_v, [-1, 3]), idx0)
  v1 = tf.gather(tf.reshape(tf_v, [-1, 3]), idx1)
  v2 = tf.gather(tf.reshape(tf_v, [-1, 3]), idx2)
  v0v1 = v1 - v0
  v0v2 = v2 - v0

  reduce_idx = tf.range(0, num_faces, reduce_step)
  num_faces_r = reduce_idx.get_shape().as_list()[0]
  # 5 x 4592 x 3
  v0_r = tf.gather(tf.reshape(v0, [-1, 3]), tf.tile(tf.reshape(tf.range(0, batch_size), [-1 ,1]), [1, num_faces_r]) * mesh_num + tf.reshape(reduce_idx, [1, -1])) 
  v1_r = tf.gather(tf.reshape(v1, [-1, 3]), tf.tile(tf.reshape(tf.range(0, batch_size), [-1 ,1]), [1, num_faces_r]) * mesh_num + tf.reshape(reduce_idx, [1, -1]))
  v2_r = tf.gather(tf.reshape(v2, [-1, 3]), tf.tile(tf.reshape(tf.range(0, batch_size), [-1 ,1]), [1, num_faces_r]) * mesh_num + tf.reshape(reduce_idx, [1, -1])) 
  face_center = (v0_r + v1_r + v2_r)/3.0
  # tf_project: 5 x 4592 x 2
  tf_project = tf.divide(tf.slice(face_center, [0,0,0], [-1,-1, 2]), tf.slice(face_center, [0, 0, 2], [-1, -1, 1]))
  # dir_ 5 x 4592 x 3
  dir_ = normalize(tf.concat([tf_project, tf.ones([batch_size, num_faces_r, 1])], 2))
  
  # N: 5 x 13776 x 3 
  N = tf.cross(v0v1, v0v2)
  NdotRayDirection = tf.matmul(N, tf.transpose(dir_, [0, 2, 1]))
  isNotParallel = tf.where(tf.less(tf.abs(NdotRayDirection), kEpsilon), tf.zeros_like(NdotRayDirection), tf.ones_like(NdotRayDirection))
  

  # find P
  d = tf.reduce_sum(tf.multiply(N, v0), 2)
  # t: batch_size x 13776 x 4592
  t = tf.expand_dims(d, 2)/NdotRayDirection
  isNotBehind = tf.where(tf.less(t, 0), tf.zeros_like(NdotRayDirection), tf.ones_like(NdotRayDirection))
  # p: batch_size x 13776 x 4592 x 3
  P = tf.expand_dims(t, 3) * tf.expand_dims(dir_, 1)
  
  # batch x #face(13776) x 1 x 3 
  edge0 = tf.tile(tf.expand_dims(v1 - v0, 2), [1,1, num_faces_r, 1])
  # vp0: batch_size x 13776 x 4592 x 3
  vp0 = P - tf.expand_dims(v0, 2)
  C = tf.cross(edge0, vp0)
  inner = tf.reduce_sum(tf.multiply(tf.expand_dims(N, 2), C), 3)
  isInTri0= tf.where(tf.less(inner, 0), tf.zeros_like(NdotRayDirection), tf.ones_like(NdotRayDirection))
  edge1 = tf.tile(tf.expand_dims(v2 - v1, 2), [1,1, num_faces_r, 1])
  vp1 = P - tf.expand_dims(v1, 2)
  C = tf.cross(edge1, vp1)  
  inner = tf.reduce_sum(tf.multiply(tf.expand_dims(N, 2), C), 3)
  isInTri1= tf.where(tf.less(inner, 0), tf.zeros_like(NdotRayDirection), tf.ones_like(NdotRayDirection))
  edge2 = tf.tile(tf.expand_dims(v0 - v2, 2), [1,1, num_faces_r, 1])
  # vp0: batch_size x 13776 x 4592 x 3
  vp2 = P - tf.expand_dims(v2, 2)
  C = tf.cross(edge2, vp2)  
  inner = tf.reduce_sum(tf.multiply(tf.expand_dims(N, 2), C), 3)
  isInTri2= tf.where(tf.less(inner, 0), tf.zeros_like(NdotRayDirection), tf.ones_like(NdotRayDirection))
  # vp0: batch_size x 13776 x 4592
  final_decision = isNotParallel * isNotBehind *  isInTri0 * isInTri1 * isInTri2
  dist = tf.where(tf.greater(final_decision, 0.5), t, 1000 * tf.ones_like(final_decision)) 
  select_faces = tf.argmin(dist, 1)
  out = tf.tile(tf.expand_dims(tf.equal(reduce_idx, tf.cast(select_faces, tf.int32)), 2), [1,1,3])
  f_reduce_idx = tf.tile(tf.expand_dims(tf.gather(f, reduce_idx), 0), [batch_size, 1, 1])
  collected_face_idx = tf.where(out, f_reduce_idx, (mesh_num) * tf.ones_like(f_reduce_idx))
 
  face_idx = tf.unstack(collected_face_idx, batch_size, 0) 
  onehot = []
  for id in range(batch_size):
    arg_min_list, _ = tf.unique(tf.reshape(face_idx[id], [-1]))
    oneh = tf.sparse_to_dense(arg_min_list, [mesh_num+ 1], 1.0, 0, validate_indices=False)
    onehot.append(oneh)
 
  visibility = tf.slice(tf.stack(onehot), [0, 0], [-1, mesh_num])

  return visibility, face_center


"""
# get visibility from igl
import pyigl as igl
from iglhelpers import *

def get_visibility(v, f, focal, depth, sr=0.5):
  
  F = p2e(np.array(f, dtype='int32'))
  V = p2e(v.astype('float64'))

  model = p2e(np.eye(4))
  view = p2e(np.eye(4))
  focal = focal.astype('float64')
  focal_x = focal[0].astype('float64')
  focal_y = focal[1].astype('float64')

  num_not_hit = 0
  num_hit = 0
  correct_hit = 0
  num_vertex = v.shape[0]
  cast_or_not = np.random.binomial(1, sr, f.shape[0])
  visibility = np.zeros((num_vertex), dtype=np.float32)
  for f_id in range(f.shape[0]):
    if cast_or_not[f_id] == 0:
      continue 
    face = f[f_id]
    face_center = (v[face[0], :] + v[face[1], :] + v[face[2], :])/3.0
    pos2 = np.reshape(face_center[:2]*focal/depth, [2, 1]) + [[focal_x], [focal_y]]
    pos2 = p2e(pos2)
    bc = igl.eigen.MatrixXd()
    viewport = p2e(np.array([0, 0, focal_x*2, focal_y*2], dtype='float64'))
    proj = p2e(np.array([[1.0/depth, 0, 0, 0], 
                         [0, 1.0/depth, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0,0 , 1]], dtype='float64'))
    #proj = p2e(np.array([[1, 0, 0, w/2], [0, 1, 0, h/2], [0, 0, 1, 0],
    #                   [0, 0,0 ,1]], dtype='float64'))
  
    pos = proj * p2e(np.concatenate([face_center, np.array([1])]))
    pos= e2p(pos)
    pos = p2e(pos[:2])
    fid = igl.eigen.MatrixXi([-1])
    hit = igl.unproject_onto_mesh(pos2, view*model, proj, viewport, V, F, fid, bc)
 
    #print 
    if e2p(fid)[0,0] == f_id:
      for v_ in face:
        visibility[v_] = 1
      #print "hit or not:", hit, ",face hit:", fid, "face id:", f_id
      correct_hit += 1
    if not hit:
      num_not_hit += 1
    else:
      num_hit += 1

  #print("num_correct_hit", float(correct_hit)/float(num_hit + num_not_hit))
  #print("num_hit", float(num_hit)/float(num_hit + num_not_hit))
  #print("num_not_hit", float(num_not_hit)/float(num_hit + num_not_hit))
  return visibility
"""
