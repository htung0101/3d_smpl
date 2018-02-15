from __future__ import division
import os
import time
from glob import glob #??
import tensorflow as tf
import numpy as np
from six.moves import xrange
import colorsys
#from misc import load_bin_with_shape
from data import Data_Helper_h36_syn

from visibility import tf_get_visibility, tf_get_visibility_raycast
from ops import *
import scipy.io as sio
import math
import pickle
#import cv2 
import struct

from pack_data.tfrecord_utils import inputs_surreal, inputs_surreal_with_idx  
from tqdm import tqdm

from utils import warper 

class _3DINN(object):
    def __init__(self, sess, checkpoint_dir, logs_dir, sample_dir, config=None):
        self.sess = sess
        self.config = config
        self.is_unsup_train = self.config.key_loss or self.config.silh_loss \
                              or self.config.pixel_loss 
        # dump path
        self.checkpoint_dir = checkpoint_dir
        self.logs_dir = logs_dir
        self.sample_dir = sample_dir
         
        # shrink chamfer image size in order to save memory
        self.small_h = int(self.config.image_size_h * self.config.chamfer_scale)
        self.small_w = int(self.config.image_size_w * self.config.chamfer_scale)

        # batch normalization
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn1_1 = batch_norm(name='g_bn1_1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn2_1 = batch_norm(name='g_bn2_1')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.g_bn3_1 = batch_norm(name='g_bn3_1')
        self.g_bn4 = batch_norm(name='g_bn4')
        self.g_bn4_1 = batch_norm(name='g_bn4_1')
        self.g_bn5 = batch_norm(name='g_bn5')
        self.g_bn5_1 = batch_norm(name='g_bn5_1')
        self.f_bn1 = batch_norm(name='f_bn1')
        self.f_bn1_1 = batch_norm(name='f_bn1_1')
        self.f_bn2 = batch_norm(name='f_bn2')
        self.f_bn2_1 = batch_norm(name='f_bn2_1')
        self.f_bn3 = batch_norm(name='f_bn3')
        self.f_bn3_1 = batch_norm(name='f_bn3_1')
        self.f_bn4 = batch_norm(name='f_bn4')
        self.f_bn4_1 = batch_norm(name='f_bn4_1')
        self.Build_Model()
 
    def Build_Model(self):
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.image_center  = tf.constant(np.array([(self.config.image_size_h - 1)/2.0,
                                       (self.config.image_size_w - 1)/2.0], dtype=np.float32))
         
        # initial variables and constants
        self.xIdxMap = np.zeros((self.config.image_size_h, self.config.image_size_w), dtype=np.float32) 
        self.xIdxMap[:,:] = np.reshape(range(self.config.image_size_w), [1,-1])
        self.yIdxMap = np.zeros((self.config.image_size_h, self.config.image_size_w), dtype=np.float32) 
        self.yIdxMap[:,:] = np.reshape(range(self.config.image_size_h), [-1,1])
        
        # constant variables 
        # rest post for smpl models: male/female 
        f_model_pkl_filename = '../smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
        dd_f = pickle.load(open(f_model_pkl_filename, 'rb'))
        m_model_pkl_filename = '../smpl/models/basicModel_m_lbs_10_207_0_v1.0.0.pkl'
        dd_m = pickle.load(open(m_model_pkl_filename, 'rb'))
        # facet
        self.f = dd_f['f']
        self.kintree_table = dd_f['kintree_table']
        dd_v_template = np.concatenate((np.expand_dims(dd_f['v_template'], 0),
                                        np.expand_dims(dd_m['v_template'], 0)), 0) 
        self.mesh_mu = tf.constant(dd_v_template, dtype=tf.float32, name="mesh_mu")
        
        dd_shapedirs = np.concatenate((np.expand_dims(dd_f['shapedirs'], 0),
                                        np.expand_dims(dd_m['shapedirs'], 0)), 0)  
        self.mesh_pca = tf.constant(np.array(dd_shapedirs), dtype=tf.float32, name="mesh_pca")
        
        dd_posedirs = np.concatenate((np.expand_dims(dd_f['posedirs'], 0),
                                        np.expand_dims(dd_m['posedirs'], 0)), 0)  
        self.posedirs = tf.constant(np.array(dd_posedirs), dtype=tf.float32, name="posedirs")

        
        dd_J_regressor = np.concatenate((np.expand_dims(dd_f['J_regressor'].todense(), 0),
                                        np.expand_dims(dd_m['J_regressor'].todense(), 0)), 0)  
        self.J_regressor = tf.constant(dd_J_regressor, dtype=tf.float32, name = "J_regressor")
        
        dd_weights = np.concatenate((np.expand_dims(dd_f['weights'], 0),
                                        np.expand_dims(dd_m['weights'], 0)), 0)  
        self.weights = tf.constant(dd_weights, dtype=tf.float32, name="weights")
        
        #load data from tfrecords 
        # synthetic data from surreal
        surreal_filenames = ["../tf_code/tfrecords2/surreal2_quo" + str(id) + ".tfrecords" for id in range(1)] 
        self.pose_sr, self.beta_sr, self.T_sr, self.R_sr, self.J_sr, self.J_2d_sr, self.image_sr, self.seg_sr, \
            self.chamfer_sr, self.c_sr, self.f_sr, self.resize_scale_sr, self.gender_sr, self.J_c_sr, self.pmesh_sr, self.v_gt_sr = \
            self.centered_3d(*inputs_surreal(surreal_filenames, self.config.batch_size))
        
        # validation data from surreal
        test_list = [5]
        surreal_test_filenames = ["../tf_code/tfrecords2/surreal2_100_test_quo1.tfrecords"]
        self.pose_sr_v, self.beta_sr_v, self.T_sr_v, self.R_sr_v, self.J_sr_v, self.J_2d_sr_v, self.image_sr_v, self.seg_sr_v, \
            self.chamfer_sr_v, self.c_sr_v, self.f_sr_v, self.resize_scale_sr_v, self.gender_sr_v, self.J_c_sr_v, self.idx_sr_v, self.pmesh_sr_v, self.v_gt_v = \
            self.centered_3d_with_idx(*inputs_surreal_with_idx(surreal_test_filenames, self.config.batch_size))
        bat_nframes = [self.config.batch_size, self.config.num_frames] 
        image_size = [self.config.image_size_h, self.config.image_size_w]
        # input/gt
        self.pose_gt = tf.placeholder(tf.float32, bat_nframes + \
                                   [self.config.keypoints_num, 3], name="pose_gt")
        
        self.gender_gt = tf.placeholder(tf.int32, self.config.batch_size, name="gender_gt")
        self.T_gt = tf.placeholder(tf.float32, bat_nframes + [3], name="T_gt")
        self.R_gt = tf.placeholder(tf.float32, bat_nframes + [6], name="R_gt")
        self.f_gt = tf.placeholder(tf.float32, bat_nframes + [2], name="f_gt")
        self.c_gt = tf.placeholder(tf.float32, bat_nframes + [2], name="c_gt")
        self.resize_scale_gt = tf.placeholder(tf.float32, bat_nframes, name="resize_scale_gt")
        self.beta_gt = tf.placeholder(tf.float32, bat_nframes + [self.config.bases_num], 
                                      name="beta_gt") 
        self.J_gt = tf.placeholder(tf.float32, bat_nframes + [self.config.keypoints_num, 3], 
                                   name="J_gt")
        self.pmesh_gt = tf.placeholder(tf.float32, bat_nframes + [self.config.mesh_num, 2], 
                                   name="pmesh_gt")
        self.v_gt = tf.placeholder(tf.float32, bat_nframes + [self.config.mesh_num, 3], 
                                   name="v_gt")
        self.J_c_gt = tf.placeholder(tf.float32, bat_nframes + [self.config.keypoints_num, 3], 
                                   name="J_c_gt")
        self.J_2d_gt = tf.placeholder(tf.float32, bat_nframes + [self.config.keypoints_num, 2],
                                       name="J_2d_gt")
        self.images = tf.placeholder(tf.float32, bat_nframes + image_size + [3]\
                                     , name="images")       
        #self.visibility = tf.placeholder(tf.float32, bat_nframes + \
        #                                 [self.config.mesh_num], name="visibility")       
        self.seg_gt = tf.placeholder(tf.float32, bat_nframes + image_size,\
                                   name="seg_gt")  
        self.chamfer_gt = tf.placeholder(tf.float32, bat_nframes + \
                                         [self.small_h, self.small_w], name="chamfer_gt")       
        
        # split into frames
        pose_gt_split = split(self.pose_gt, 1) 
        beta_gt_split = split(self.beta_gt, 1)
        T_gt_split = split(self.T_gt, 1)
        R_gt_split = split(self.R_gt, 1)
        J_gt_split = split(self.J_gt, 1)
        pmesh_gt_split = split(self.pmesh_gt, 1)
        v_gt_split = split(self.v_gt, 1)
        J_c_gt_split = split(self.J_c_gt, 1)
        f_gt_split = split(self.f_gt, 1)
        c_gt_split = split(self.c_gt, 1)
        resize_scale_gt_split = split(self.resize_scale_gt, 1)
        J_2d_gt_split = split(self.J_2d_gt, 1)
        image_split = split(self.images, 1)
        #visibility_split = split(self.visibility, 1)
        seg_gt_split = split(self.seg_gt, 1)
        chamfer_gt_split = split(self.chamfer_gt, 1)
        
        # main network that takes one rgb and output pose, beta, T, R
         
        self.heatmaps = {}
        self.pose = {}
        self.beta = {}
        self.R = {}
        self.T = {}

        self.pose_loss = 0
        self.beta_loss = 0
        self.R_loss = 0
        self.T_loss = 0
        
        # predict smpl parameters: beta, pose from heatmaps and rgb 
        for frame_id in range(self.config.num_frames):
          self.heatmaps[frame_id] = self.ToHeatmaps(self.config.gStddev, 
                            self.config.gWidth,
                            self.config.image_size_h, self.config.image_size_w,
                            J_2d_gt_split[frame_id])
          self.beta[frame_id], self.pose[frame_id], self.R[frame_id], self.T[frame_id] = \
              self._3D_mesh_Interpretor(self.heatmaps[frame_id], image_split[frame_id],
              self.gender_gt, f_gt_split[frame_id], c_gt_split[frame_id], \
                  resize_scale_gt_split[frame_id] ,is_train=True, reuse=frame_id)
          self.pose_loss += eud_loss(self.pose[frame_id], pose_gt_split[frame_id]) 
          self.beta_loss += eud_loss(self.beta[frame_id], beta_gt_split[frame_id]) 
          self.R_loss += eud_loss(self.R[frame_id], R_gt_split[frame_id])
          self.T_loss += eud_loss(self.T[frame_id], T_gt_split[frame_id])

        # supervised loss
        self.sup_loss = self.pose_loss + 0.05 * self.beta_loss + self.R_loss + 0.1*self.T_loss
        seg = self.seg_gt
        chamfer = self.chamfer_gt
        # supervised summary
        sup_loss_summary = get_scalar_summary("supervised loss", self.sup_loss) 
        pose_loss_summary = get_scalar_summary("pose loss", self.pose_loss)
        beta_loss_summary = get_scalar_summary("beta loss", self.beta_loss) 
        R_loss_summary = get_scalar_summary("R loss", self.R_loss) 
        T_loss_summary = get_scalar_summary("T loss", self.T_loss) 
         
        seg_split = split(seg, 1)
        chamfer_split = split(chamfer, 1)
        
        # pass pose/beta to smpl model and add rotation and translation 
        self.v = {}
        self.J = {}
        self.J_c = {}
        self.J_ori = {}
        #self.mesh_loss = 0  
        self.d3_loss = 0
        self.d3_joint_loss = 0
        self.centered_d3_joint_loss = 0
        self.centered_mesh_loss = 0
        for frame_id in range(self.config.num_frames): 
          self.v[frame_id], self.J[frame_id], self.J_ori[frame_id] = self.pose_beta_to_mesh(self.beta[frame_id], 
              self.pose[frame_id], self.gender_gt) 
          R = tf.transpose(self.angle2R(self.R[frame_id]), [0, 2, 1])
          # 2x 6890x3
          v_centered = self.v[frame_id]
          self.v[frame_id] = tf.matmul(self.v[frame_id], R) + tf.expand_dims(self.T[frame_id], 1)
          self.J_c[frame_id] = tf.matmul(self.J[frame_id], R)
          self.J[frame_id] = self.J_c[frame_id] + tf.expand_dims(self.T[frame_id], 1)

          self.d3_loss += eud_loss(self.J[frame_id], J_gt_split[frame_id])
          self.d3_joint_loss += per_joint_loss(self.J[frame_id], J_gt_split[frame_id])
          self.centered_d3_joint_loss += per_joint_loss(self.J_c[frame_id], J_c_gt_split[frame_id])
          self.centered_mesh_loss += per_joint_loss(v_centered, v_gt_split[frame_id])
        self.d3_joint_loss /= self.config.num_frames
        self.centered_d3_joint_loss /= self.config.num_frames
        self.centered_mesh_loss /= self.config.num_frames

        d3_loss_summary = get_scalar_summary("d3 joint loss", self.d3_joint_loss)
        centered_d3_loss_summary = get_scalar_summary("centered d3 joint loss", self.centered_d3_joint_loss)
        centered_mesh_loss_summary = get_scalar_summary("centered mesh loss", self.centered_mesh_loss)
        # projections
        project = {}
        direct_project = {}
        project_J = {}
        self.depth_J = {}
        self.d2_loss = 0
        self.d2_joint_loss = 0
        for frame_id in range(self.config.num_frames):
          focal_length = tf.expand_dims(f_gt_split[frame_id], 1)
          depth_mesh = tf.slice(self.v[frame_id], [0, 0, 2], [-1, -1, 1])
          depth_J = tf.slice(self.J[frame_id], [0, 0, 2], [-1, -1, 1])
          direct_project[frame_id] = tf.divide(tf.slice(self.v[frame_id], [0, 0, 0], [-1, -1, 2])\
                                , depth_mesh) 
          project_J[frame_id] = tf.divide(tf.slice(self.J[frame_id], [0, 0, 0], [-1, -1, 2])\
                                ,depth_J)
           
          project[frame_id] = tf.reshape(resize_scale_gt_split[frame_id], [-1, 1, 1]) \
                              * direct_project[frame_id] * focal_length + tf.expand_dims(c_gt_split[frame_id], 1)
          project_J[frame_id] = tf.reshape(resize_scale_gt_split[frame_id], [-1, 1, 1])\
                                * project_J[frame_id] * focal_length + tf.expand_dims(c_gt_split[frame_id], 1)
          
          self.depth_J[frame_id] = project_J[frame_id]
          self.d2_loss = eud_loss(project_J[frame_id], J_2d_gt_split[frame_id])
          self.d2_joint_loss += per_joint_loss(project_J[frame_id], J_2d_gt_split[frame_id])
          

        self.d2_joint_loss /= self.config.num_frames
        d2_loss_summary = get_scalar_summary("d2 joint loss", self.d2_joint_loss)
        heatmaps = self.ToHeatmaps(self.config.gStddev, 
                            self.config.gWidth,
                            self.config.image_size_h, self.config.image_size_w,
                            project_J[0])
        d2_heatmap = self.visualize_joint_heatmap(image_split[0], self.config.keypoints_num, heatmaps)
        d2_image_summary = get_image_summary("input_2d_heatmap", d2_heatmap, 4);

        flow = project[1] - project[0]
        self.flow = flow
        self.project1 = project_J[0]
        self.project_mesh0 = project[0]
        self.project_mesh1 = project[1]
        # 13776x3 
        
        # only calculate visibility in the first frame
        self.tf_visibility, self.tf_arg_min = tf_get_visibility_raycast(self.v[0], self.f, 6)  
        self.tf_visibility = tf.tile(tf.expand_dims(self.tf_visibility, 1), [1, self.config.num_frames ,1])

        # segmentation loss 
        self.silh_loss = 0 
        small_height = self.small_h
        small_width = self.small_w
        self.C_M = {}
        self.S_M = {}
        for frame_id in range(self.config.num_frames): 
          self.C_M[frame_id], self.S_M[frame_id] = self.get_chamfer_and_seg(project[frame_id], scale=self.config.chamfer_scale)
          S_I = tf.squeeze(tf.image.resize_images(tf.expand_dims(seg_split[frame_id], 3), [small_height, small_width]), 3) 
          C_I = chamfer_split[frame_id]
          
          self.silh_loss += tf.reduce_mean(tf.square(\
              self.config.alpha * self.S_M[frame_id] * C_I + \
              (1-self.config.alpha) * S_I * self.C_M[frame_id]))
          if frame_id == 0:
            self.output = self.S_M[frame_id]

        silh_loss_summary = get_scalar_summary("silh loss", self.silh_loss) 
        # use projected points to get pixels
        self.pixels = {}
        for frame_id in range(self.config.num_frames):
          self.pixels[frame_id] = self.pixel_interpolate(image_split[0], project[frame_id], self.config.image_size_w, self.config.image_size_h)
        # pixel loss 
        visibility_split = split(self.tf_visibility, 1)
        
        self.flow_pixels = pmesh_gt_split[1] - pmesh_gt_split[0] 
        self.pixel_loss = l1_loss(tf.expand_dims(visibility_split[0], 2) * self.flow_pixels, 
                                  tf.expand_dims(visibility_split[0], 2) * self.flow)  

        pixel_hist_summary = get_hist_summary("flow pixels", self.flow_pixels)        
        pixel_loss_summary = get_scalar_summary("pixel loss", self.pixel_loss) 
        self.pixel0 = self.pixels[0]
        self.pixel1 = self.pixels[1]
        # need to add smooth loss here
        self.recon_loss = tf.Variable(0, dtype=tf.float32)
        if self.config.key_loss:
          self.recon_loss += 0.01*self.d2_loss #self.pixel_loss + 10 * self.silh_loss #+ self.d3_loss  
        if self.config.silh_loss:
          self.recon_loss += 0.0000000000000000001*self.silh_loss

        if self.config.pixel_loss:
          self.recon_loss += 0.01 * self.pixel_loss
   
        # summary
        base_summ = [pose_loss_summary, beta_loss_summary, d3_loss_summary, centered_d3_loss_summary, d2_loss_summary, 
                     d2_image_summary, R_loss_summary, T_loss_summary, #flow_summary, 
                     centered_mesh_loss_summary]
        if self.config.silh_loss:
          base_summ.append(silh_loss_summary) 
        if self.config.pretrained_flownet:
          base_summ.append(i1e_image_summary) 
          base_summ.append(flow_image_summary) 
          
        syn_summ = base_summ + [sup_loss_summary]
        self.syn_summary = tf.summary.merge([dict_["syn_train"] for dict_ in syn_summ])
        if self.config.pixel_loss:
          syn_summ.append(pixel_loss_summary)
          syn_summ.append(pixel_hist_summary)

        self.syn_v_summary = tf.summary.merge([dict_["syn_test"] for dict_ in syn_summ])
        self.writer = tf.summary.FileWriter(self.logs_dir, self.sess.graph)
        
        self.saver = tf.train.Saver()  

    

    
    def centered_3d(self, pose, beta, T, R, J, J_2d, image, seg, chamfer, c, f, resize_scale, gender):
      centered_J_2d = (J_2d - tf.reshape(self.image_center, [1,1,1,2]))/(tf.reshape(resize_scale, [self.config.batch_size, self.config.num_frames, 1, 1])* tf.expand_dims(f, 2))
      centered_J_2d = merge_bf(centered_J_2d)
    
      v_gt, J_gt, _ = self.pose_beta_to_mesh(merge_bf(beta), merge_bf(pose), tf.reshape(tf.tile(tf.expand_dims(gender, 1), [1,2]), [-1]))
      ang_R = tf.transpose(self.angle2R(merge_bf(R)), [0, 2, 1])
      J_gt = tf.matmul(J_gt, ang_R)
      v = tf.matmul(v_gt, ang_R)
      # [X, Y]
      T3 = tf.slice(T, [0,0,2], [-1, -1, 1])
      T12 = centered_J_2d * (tf.slice(J_gt, [0, 0, 2], [-1, -1, 1]) + tf.expand_dims(merge_bf(T3), 1)) - tf.slice(J_gt, [0, 0, 0], [-1, -1, 2])
      T12 = split_bf(tf.reduce_mean(T12, 1), self.config.batch_size, self.config.num_frames)
      T = tf.concat([T12, T3], 2)
      J_gt = split_bf(J_gt, self.config.batch_size, self.config.num_frames)
        
      v = v + tf.expand_dims(merge_bf(T),1)
      J = J_gt + tf.expand_dims(T, 2) 
 
      c = tf.tile(tf.reshape(self.image_center, [1,1,2]), [self.config.batch_size, self.config.num_frames, 1])
      pmesh = tf.divide(tf.slice(v, [0,0,0], [-1, -1, 2]) * tf.expand_dims(merge_bf(f), 1),\
                        tf.slice(v, [0, 0, 2], [-1, -1, 1]))
      pmesh = tf.reshape(merge_bf(resize_scale), [-1, 1, 1])\
                        * pmesh + tf.expand_dims(merge_bf(c), 1) 
      pmesh = split_bf(pmesh, self.config.batch_size, self.config.num_frames)
      v_gt = split_bf(v_gt, self.config.batch_size, self.config.num_frames)
      return pose, beta, T, R, J, J_2d, image, seg, chamfer, c, f, resize_scale, gender, J_gt, pmesh, v_gt
    
    def centered_3d_with_idx(self, pose, beta, T, R, J, J_2d, image, seg, chamfer, c, f, resize_scale, gender, idx):
      pose, beta, T, R, J, J_2d, image, seg, chamfer, c, f, resize_scale, gender, J_gt, pmesh, v_gt \
          = self.centered_3d(pose, beta, T, R, J, J_2d, image,seg, chamfer, c, f, resize_scale,\
                             gender)
      return pose, beta, T, R, J, J_2d, image, seg, chamfer, c, f, resize_scale, gender, J_gt, idx, pmesh, v_gt
      
    def angle2R(self, angle):
      batch_size = angle.get_shape().as_list()[0]
      [sinx, siny, sinz, cosx, cosy, cosz] = tf.unstack(angle, 6, 1) 
      one = tf.ones_like(sinx, name="one")
      zero = tf.zeros_like(sinx, name="zero")
      Rz = tf.reshape(tf.stack([cosz, -sinz, zero, 
                               sinz, cosz, zero, 
                               zero, zero, one], axis=1), 
                      [batch_size, 3, 3])
      Ry = tf.reshape(tf.stack([cosy, zero, siny,
                               zero, one, zero,
                               -siny, zero, cosy], axis=1),
                       [batch_size, 3, 3])  
      Rx = tf.reshape(tf.stack([one, zero, zero,
                               zero, cosx, -sinx,
                               zero, sinx, cosx], axis=1),
                      [batch_size, 3, 3]) 
      Rcam=tf.matmul(tf.matmul(Rz,Ry), Rx, name="Rcam")
      return Rcam 
 
    def get_chamfer_and_seg(self, project, scale = 1.0):
        small_height = self.small_h #int(self.config.image_size_h/scale)
        small_width = self.small_w #int(self.config.image_size_w/scale)
        
        xIdxMap_ = getIdxMap(self.config.batch_size, small_height, small_width)
        xIdxMap = tf.reshape(xIdxMap_, [self.config.batch_size, -1, 2])
        project0 = tf.transpose(project, [0, 2, 1]) * scale
        dist = tf.expand_dims(tf.reduce_sum(tf.square(xIdxMap), 2), 2)\
              - 2 * tf.matmul(xIdxMap, project0)\
              +tf.expand_dims(tf.reduce_sum(tf.square(project0),1), 1) 
        dist = tf.sqrt(tf.maximum(tf.reshape(tf.reduce_min(dist, 2), [-1, small_height, small_width]), 1e-6*tf.ones([self.config.batch_size, small_height, small_width])))
        C_M = tf.where(dist > 0.51, dist, tf.zeros_like(dist))
        S_M = tf.where(dist < 0.5, dist + 0.5 * tf.ones_like(dist), tf.zeros_like(dist))
        return C_M, S_M
    def pixel_interpolate(self, image, project, width, height):
        b, h, w, c = image.get_shape().as_list()
        # get pixel accoding to project1
        x = tf.reshape(tf.slice(project, [0, 0, 0], [-1, -1, 1]), [-1])
        y = tf.reshape(tf.slice(project, [0, 0, 1], [-1, -1, 1]), [-1])
        x = tf.clip_by_value(x, 0, width - 1)
        y = tf.clip_by_value(y, 0, height - 1)
          
        x0_f = tf.floor(x)
        y0_f = tf.floor(y)
        x1_f = x0_f + 1 
        y1_f = y0_f + 1
        x0 = tf.cast(x0_f, 'int32')
        y0 = tf.cast(y0_f, 'int32')
        x1 = tf.cast(tf.minimum(x1_f, width-1), 'int32')
        y1 = tf.cast(tf.minimum(y1_f, height-1), 'int32')
        
        dim2 = width
        dim1 = width*height
        base = repeat(tf.range(self.config.batch_size)*dim1, self.config.mesh_num)
        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1


        im_flat = tf.reshape(image, [-1, c]) 
       
        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)
        wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
        wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
        wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
        wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
        pixels = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])         
        return tf.reshape(pixels, [self.config.batch_size, -1, c])
    def get_poseweights(self, poses):
        # pose: batch x 24 x 3 
        pose_matrix, _ = self.rodrigues(tf.reshape(tf.slice(poses, [0, 1, 0], [-1, self.config.keypoints_num-1, -1]), [-1, 3]))
        pose_matrix = pose_matrix - np.expand_dims(np.eye(3, dtype=np.float32), 0)

        pose_matrix = tf.reshape(pose_matrix, [self.config.batch_size, -1])
        return pose_matrix 
    def pose_beta_to_mesh(self, betas, poses, gender):
        batch_size = betas.get_shape().as_list()[0]
        kintree_table = self.kintree_table
        id_to_col = {kintree_table[1,i] : i for i in range(kintree_table.shape[1])}
        parent = {i : id_to_col[kintree_table[0,i]] for i in range(1, kintree_table.shape[1])}
        mesh_mu = tf.gather(self.mesh_mu, gender)
        mesh_pca = tf.gather(self.mesh_pca, gender)
        posedirs = tf.gather(self.posedirs, gender)

        v_shaped =tf.matmul(tf.expand_dims(betas, 1), tf.reshape(tf.transpose(mesh_pca, [0, 3, 1, 2]),
                 [batch_size, self.config.bases_num, -1]))
        v_shaped = tf.reshape( 
             tf.squeeze(tf.matmul(tf.expand_dims(betas, 1), 
                        tf.reshape(tf.transpose(mesh_pca, [0, 3, 1, 2]),
                        [batch_size, self.config.bases_num, -1])), axis=1) 
             + tf.reshape(mesh_mu, [batch_size, -1]), 
             [batch_size, self.config.mesh_num, 3]) #6890x3

        print("posedirs", posedirs.get_shape())
        # posedirs: batch x 6890 x 3 x 207
        pose_weights = self.get_poseweights(poses)
        print("pose_weights", pose_weights.get_shape())
        v_posed = v_shaped + tf.squeeze(tf.matmul(posedirs, 
            tf.tile(tf.reshape(pose_weights, 
            [batch_size, 1, 
            (self.config.keypoints_num - 1)*9,1]), 
            [1, self.config.mesh_num, 1, 1])), 3)
        # v_shaped: batch x 6890 x3
        J_regressor = tf.gather(self.J_regressor, gender)
        J_posed = tf.matmul(tf.transpose(v_shaped, [0, 2, 1]), 
                            tf.transpose(J_regressor, [0, 2, 1]))         
        # J_posed: b x 24 x 3 
        J_posed = tf.transpose(J_posed, [0, 2, 1])
        # 24 x [b x 3]
        J_posed_split = [tf.reshape(sp, [batch_size, 3]) for sp in tf.split(tf.transpose(J_posed, [1, 0, 2]), self.config.keypoints_num, 0)]
       
        # 24 x [b x3]
        pose = tf.transpose(poses, [1, 0, 2])
        self.result = pose
        pose_split = tf.split(pose, self.config.keypoints_num, 0)
        #angle = self.rodrigues(pose[0, :, :])
        angle_matrix =[]
        for i in range(self.config.keypoints_num):
          out, tmp = self.rodrigues(tf.reshape(pose_split[i], [-1, 3]))
          angle_matrix.append(out)
        #angle_matrix = [self.rodrigues(tf.reshape(pose_split[i], [-1, 3]))[0] for i in range(self.config.keypoints_num)] 
        with_zeros = lambda x: tf.concat((x, 
            tf.tile(tf.constant([[[0.0, 0.0, 0.0, 1.0]]], dtype=np.float32), 
                    [batch_size, 1, 1])), 1)
        pack = lambda x: tf.concat((tf.zeros((batch_size, 4, 3), dtype=tf.float32), x), 2)
        results = {}
        
        results[0] = with_zeros(tf.concat((angle_matrix[0], 
            tf.reshape(J_posed_split[0], [batch_size, 3, 1])), 2))        
        
        for i in range(1, kintree_table.shape[1]):
            tmp = with_zeros(tf.concat((angle_matrix[i],
                tf.reshape(J_posed_split[i] - J_posed_split[parent[i]], 
                [batch_size, 3, 1])), 2)) 
            results[i] = tf.matmul(results[parent[i]], tmp)
        # 24, 2x4x4
        results_global = results
        Jtr = []
        for j_id in range(len(results_global)):
          Jtr.append(tf.slice(results_global[j_id], [0,0,3], [-1, 3, 1]))
        # batchsize x 24 x 3
        Jtr = tf.transpose(tf.concat(Jtr, 2), [0, 2, 1])    
         
        #pack = lambda x : tf.concat((np.zeros((4, 3)), x.reshape((4,1))), 1) 
        results2 = []
        for i in range(len(results)):
            vec = tf.reshape(tf.concat((J_posed_split[i], \
                       tf.zeros([batch_size, 1])), 1), [batch_size, 4, 1])
            results2.append(tf.expand_dims(results[i] - pack(tf.matmul(results[i], vec)), axis=0))
        # 24xbx4x4

        weights = tf.gather(self.weights, gender)
        # 24 x 2 x 4 x 4 
        results = tf.concat(results2, 0)

        #print "results", results.shape
        # 2 x 4 x 4 x 6890
        # batch x 4 x 4 x 6890, batchx1x6890x24
        T = tf.matmul(tf.transpose(results, [1,2,3,0]), tf.tile(tf.expand_dims(tf.transpose(weights, [0, 2, 1]), 1), [1,4,1,1]))
        Ts = tf.split(T, 4, 2) 
        # 2 x 6890 x4
        rest_shape_h = tf.concat((v_posed, np.ones((batch_size, 
                                  self.config.mesh_num, 1))), 2) 
        rest_shape_hs = tf.split(rest_shape_h, 4, 2)
        # 2 x 4 x6890
        v = tf.reshape(Ts[0], [batch_size, 4, self.config.mesh_num]) \
                * tf.reshape(rest_shape_hs[0], [-1, 1, self.config.mesh_num]) \
            + tf.reshape(Ts[1], [batch_size, 4, self.config.mesh_num]) \
                * tf.reshape(rest_shape_hs[1], [-1, 1, self.config.mesh_num]) \
            + tf.reshape(Ts[2], [batch_size, 4, self.config.mesh_num]) \
                * tf.reshape(rest_shape_hs[2], [-1, 1, self.config.mesh_num]) \
            + tf.reshape(Ts[3], [batch_size, 4, self.config.mesh_num]) \
                * tf.reshape(rest_shape_hs[3], [-1, 1, self.config.mesh_num]) 
        v = tf.slice(tf.transpose(v, [0, 2, 1]), [0,0,0], [-1, -1, 3])
        #J = tf.matmul(tf.transpose(v, [0, 2, 1]), tf.transpose(J_regressor, [0, 2, 1]))         
        #J = tf.transpose(J, [0, 2, 1])
        J = Jtr
        return v, Jtr, Jtr


    def rodrigues(self, r):     
      theta = tf.sqrt(tf.reduce_sum(tf.square(r), 1))
      #theta = tf.stop_gradient(theta)
      def S(n_):
        ns = tf.split(n_, 3, 1)
        Sn_ = tf.stack([tf.zeros_like(ns[0]),-ns[2],ns[1],ns[2],tf.zeros_like(ns[0]),-ns[0],-ns[1],ns[0],tf.zeros_like(ns[0])], 1)
        Sn_ = tf.reshape(Sn_, [-1, 3, 3])
        return Sn_ 
      #if theta > 1e-30:
      n = r/tf.reshape(theta, [-1, 1])
      Sn = S(n)
     
      R = tf.expand_dims(tf.eye(3), 0) + tf.reshape(tf.sin(theta), [-1, 1, 1])*Sn\
          + (1-tf.reshape(tf.cos(theta), [-1, 1, 1]))* tf.matmul(Sn,Sn)
      # else:
      Sr = S(r)
      theta2 = theta**2
      R2 = tf.expand_dims(tf.eye(3), 0) + (1- tf.reshape(theta2, [-1, 1, 1])/6.)*Sr \
           + (.5-tf.reshape(theta2, [-1, 1, 1])/24.)*tf.matmul(Sr,Sr)
      
      R = tf.where(tf.greater(theta, 1e-30), R, R2)
      return R, Sn


    def ToHeatmaps(self, gStddev, gWidth, image_size_h, image_size_w, coords_2d):
        # coords_2d: [batch_size, keypoints_num, channel=2]
        print("==========ToHeatmaps==========")
        # original codes
        #jointPos = tf.reshape(jointPos, [self.config.batch_size, 1, 1, self.config.keypoints_num*2])
        #[posX, posY] = tf.split(3, 2, jointPos)

        # adjusted codes
        [x, y] = tf.split(coords_2d, 2, 2)
        print_shape(x)
        print_shape(y)
        posX = tf.reshape(x, [self.config.batch_size, 1, 1, self.config.keypoints_num], name="posX") # [batch_size, 1, 1, keypoints_num]
        posY = tf.reshape(y, [self.config.batch_size, 1, 1, self.config.keypoints_num], name="posY")
        return self.batchPointToGaussianMap(posX, posY, self.xIdxMap, self.yIdxMap, 
                                                image_size_h, image_size_w, gWidth, gStddev)

    
    def batchPointToGaussianMap(self, batchX0, batchY0, xIdxMap, yIdxMap, imgH, imgW, gWH, sigma):
        # batchX0: [batch_size, 1, 1, keypoints_num]
        # batchY0: [batch_size, 1, 1, keypoints_num]
        # xIdxMap: [batch_size, h, w, keypoints_num]
        # yIdxMap: [batch_size, h, w, keypoints_num]
        #(1/(2*3.1315926*0.1*0.1))*math.exp(-(1*1+1*1)/(2*0.1*0.1)
        var = sigma*sigma;

        x0Maps = tf.tile(batchX0, [1, imgH, imgW, 1])
        y0Maps = tf.tile(batchY0, [1, imgH, imgW, 1])
        (x0Maps - tf.reshape(xIdxMap, [-1, imgH, imgW, 1]))
        x2 = tf.square((x0Maps - tf.reshape(xIdxMap, [-1, imgH, imgW, 1]))/gWH, name="x2")
        print_shape(x2)
        y2 = tf.square((y0Maps - tf.reshape(yIdxMap, [-1, imgH, imgW, 1]))/gWH, name="y2")
        print_shape(y2)

        batch_gmap = (1./(2.*math.pi*var))*tf.exp(-(x2+y2)/(2.*var), name="batch_gmap") #2d gaussian doen't have sqrt()
        print_shape(batch_gmap)

        #normalize
        batch_sum = tf.reduce_sum(batch_gmap, [1, 2], True, name="batch_sum")
        print_shape(batch_sum)
        batch_sum = tf.clip_by_value(batch_sum, 0.000000001, float('inf')) # prevent divide by zero
        batch_norm = tf.tile(batch_sum, [1, imgH, imgW, 1], name="batch_norm")
        print_shape(batch_norm)
        batch_gmap = tf.div(batch_gmap, batch_norm, name="batch_gmap")
        print_shape(batch_gmap)
        return batch_gmap

    def _3D_mesh_Interpretor(self, heatmaps, image, gender, f_gt, c_gt, resize_scale, is_train=False, reuse=False):
        # heatmaps: [batch, h, w, keypoints_num]
        # image: [batch, h, w, 3]
        # gender: [batch, 1]
        # 4 fc layers as in 3DINN paper. We can also use other net structures e.g. autoencoder
        print("==========_3D_mesh_Interpretor==========")
        with tf.variable_scope("3D_mesh_Interpretor") as scope:
            if reuse:
               scope.reuse_variables()
            scale_factor = f_gt * tf.expand_dims(resize_scale, 1)/400.0
            shift_factor = c_gt * tf.expand_dims(resize_scale, 1)/100.0
            #heatmaps_r = tf.reshape(heatmaps, [self.config.batch_size, -1], name="heatmaps_r")
            f_dim = self.config.gf_dim
            input_ = tf.concat([heatmaps*20, image], 3) - 0.5
            fc1 = lrelu(self.g_bn1(conv2d(input_, f_dim, name='g_h1_conv'), train=is_train))
            fc1_1 = lrelu(self.g_bn1_1(conv2d(fc1, f_dim, 3, 3, 1, 1, name='g_h1_1_conv'), train=is_train))
            fc2 = lrelu(self.g_bn2(conv2d(fc1_1, f_dim * 2, name='g_h2_conv'), train=is_train))
            fc2_1 = lrelu(self.g_bn2_1(conv2d(fc2, f_dim * 2, 3, 3, 1, 1, name='g_h2_1_conv'), train=is_train))
            fc3 = lrelu(self.g_bn3(conv2d(fc2_1, f_dim * 4, name='g_h3_conv'), train=is_train))
            fc3_1 = lrelu(self.g_bn3_1(conv2d(fc3, f_dim * 4, 3, 3, 1, 1, name='g_h3_1_conv'), train=is_train))
            fc4 = lrelu(self.g_bn4(conv2d(fc3_1, f_dim * 8, name='g_h4_conv'), train=is_train))
            fc4_1 = lrelu(self.g_bn4_1(conv2d(fc4, f_dim * 8, 3, 3, 1, 1, name='g_h4_1_conv'), train=is_train))
            fc5 = lrelu(self.g_bn5(conv2d(fc4_1, f_dim * 16, name='g_h5_conv'), train=is_train))
            fc5_1 = lrelu(conv2d(fc5, f_dim * 16, 3, 3, 1, 1, name='g_h5_1_conv'))
            #out = tf.reduce_mean(fc5_1, [1,2])
            #fc4 = lrelu(conv2d(fc3, f_dim * 4, name='g_h3_1_conv'))
            out = tf.reshape(fc5_1, [self.config.batch_size, -1])
            out = tf.concat([out, tf.expand_dims(tf.cast(gender, tf.float32), 1)], 1)
            fc6 = linear(out, 1024, scope="g_fc6")
            num_hidden = 512
            fc6_1 = tf.slice(fc6, [0, 0], [-1, 512])
            fc6_2 = tf.concat([tf.slice(fc6, [0, 512], [-1, 512]), scale_factor, shift_factor], 1)

            params = linear(lrelu(fc6_1), num_hidden, scope="g_fc6_1")
            #
            beta_params = tf.slice(params, [0, 0], [-1, 256], name="beta_params")
            beta_params = linear(lrelu(beta_params), 128, scope="g_fc6_beta")   
            beta_params = linear(lrelu(beta_params), self.config.bases_num, scope="g_fc6_2_beta")    
            pose_params = tf.slice(params, [0, 256], [-1, 256], name="pose_params")
            pose_params = linear(lrelu(pose_params), (self.config.keypoints_num-1) * 4, scope="g_fc6_pose")   
            pose_params = tf.reshape(pose_params, [self.config.batch_size, (self.config.keypoints_num - 1), 4])
            pose_vector = tf.slice(pose_params, [0, 0, 0], [-1, -1, 3])
            #pose_params = pose_vector
            pose_vector = pose_vector/norm(pose_vector)
            pose_scale = tf.slice(pose_params, [0, 0, 3], [-1, -1, 1])
            pose_params = pose_vector * pose_scale
            pose_first3 = np.concatenate((math.pi * np.ones((self.config.batch_size, 1, 1)),
                                          np.zeros((self.config.batch_size, 1, 2))), axis=2).astype(np.float32)
            pose_first3 = tf.constant(pose_first3)
            pose_params = tf.concat([pose_first3, pose_params], 1)

            params_RT = linear(lrelu(fc6_2), num_hidden, scope="g_fc6_2")           
            R_params = tf.slice(params_RT, [0, 0], [-1, 256], name="R_params")
            R_params = linear(lrelu(R_params), 6, scope="R_params_2")           
            R_params = tf.reshape(R_params, [-1, 3, 2])
            R_params = tf.reshape(tf.transpose(R_params/norm(R_params), [0, 2, 1]), [-1, 6])
            
            T3_params = tf.slice(params_RT, [0, 256], [-1, 128], name="T3_params")
            T3_params = tf.nn.relu(linear(lrelu(T3_params), 128, scope="T3_params_2")) 
            T3_params = tf.nn.relu(linear(lrelu(T3_params), 1, scope="T3_params_3"))*6.0   
         
           
            T12_params = tf.slice(params_RT, [0, 384], [-1, 128], name="T12_params")
            T12_params = linear(lrelu(T12_params), 2, scope="T12_params_2")   
            T12_params = T12_params #+ (c_gt - self.image_center) * T3_params/f_gt 
            
            T_params = tf.concat([T12_params, T3_params], 1)
        return beta_params, pose_params, R_params, T_params

    def flow_net(self, frame1, frame2, is_train=True, reuse=False):
        input_ = tf.concat([frame1, frame2], 3)
        print("==========Flow_net==========")
        with tf.variable_scope("flownet") as scope:
            if reuse:
               scope.reuse_variables()
            
            f_dim = self.config.flow_dim
            fc1 = lrelu(self.f_bn1(conv2d(input_, f_dim, name='f_h1_conv'), train=is_train))
            fc1_1 = lrelu(self.f_bn1_1(conv2d(fc1, f_dim, 3, 3, 1, 1, name='f_h1_1_conv'), train=is_train))
            fc2 = lrelu(self.f_bn2(conv2d(fc1_1, f_dim * 2, name='f_h2_conv'), train=is_train))
            fc2_1 = lrelu(self.f_bn2_1(conv2d(fc2, f_dim * 2, 3, 3, 1, 1, name='f_h2_1_conv'), train=is_train))
            fc3 = lrelu(self.f_bn3(conv2d(fc2_1, f_dim * 4, name='f_h3_conv'), train=is_train))
            fc3_1 = lrelu(self.f_bn3_1(conv2d(fc3, f_dim * 4, 3, 3, 1, 1, name='f_h3_1_conv'), train=is_train))
            fc4 = lrelu(self.f_bn4(conv2d(fc3_1, f_dim * 8, name='f_h4_conv'), train=is_train))
            fc4_1 = lrelu(self.f_bn4_1(conv2d(fc4, f_dim * 8, 3, 3, 1, 1, name='f_h4_1_conv'), train=is_train))
            s = self.config.image_size_h 
            sw = self.config.image_size_w 
            s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)
            s2_w, s4_w, s8_w, s16_w = int(sw/2), int(sw/4), int(sw/8), int(sw/16)
            
            fc5 = deconv2d(fc4_1,  [self.config.batch_size, s8, s8_w, f_dim*4], name="f_5")
            fc5 = tf.concat([fc5, fc3_1], 3)
            fc6 = deconv2d(fc5,  [self.config.batch_size, s4, s4_w, f_dim*2], name="f_6")
            fc6 = tf.concat([fc6, fc2_1], 3)
            fc7 = deconv2d(fc6,  [self.config.batch_size, s2, s2_w, f_dim], name="f_7")
            fc7 = tf.concat([fc7, fc1_1], 3)
            flow = deconv2d(fc7,  [self.config.batch_size, s, sw, 2], name="f_8")

            i1_e, i2_nocc = warper(frame2,flow)              
            return flow, i1_e
   
    def _3D_Reconstructor(self, intern_params, bases, mu):
        ################################################## 
        # coords_3d = \sum(alpha_i*bases_i)
        ##################################################
        print("===========_3D_Reconstructor===========")
        with tf.variable_scope("3D_Reconstructor"):
            # intern_params: [batch_size, bases_num]
            # bases: [batch_size, bases_num, keypoints_num, channel=3]
            
            coords_3d_ = tf.matmul(intern_params, bases) + mu 
            coords_3d = tf.reshape(coords_3d_, 
                                   [self.config.batch_size, self.config.keypoints_num, 3], 
                                   name="coords_3d")   
        return coords_3d


    def _3D_3D_Transformer(self, extern_params, coords_3d, eval=False):
        ################################################## 
        # coords_3d_new = R*coords_3d+t
        ##################################################
        print("==========_3D_3D_Transformer==========")
        with tf.variable_scope("3D_3D_Transformer"):
            # this part is from Adam's code
            [sinx, siny, sinz, cosx, cosy, cosz, tx, ty, tz] = tf.unpack(extern_params, axis=1)

            #  get translation params 
            print_shape(tx)
            print_shape(ty)
            print_shape(tz)
            tcam=tf.stack([tx, ty, tz], axis=1, name="tcam") # batch_size * 3
            print_shape(tcam)
            tcam_ex = tf.expand_dims(tcam, dim=1, name="tcam_ex")
            print_shape(tcam_ex)
            tcam_tiled = tf.tile(tcam_ex, [1,self.config.keypoints_num,1], name="tcam_tiled")
            print_shape(tcam_tiled)
            print_shape(coords_3d)

            one = tf.ones_like(tx, name="one")
            zero = tf.zeros_like(tx, name="zero")
            Rz = tf.reshape(tf.stack([cosz, -sinz, zero,
                                     sinz, cosz, zero,
                                     zero, zero, one], axis=1),
                            [self.config.batch_size, 3, 3]) #[batch_size, 3, 3]
            Ry = tf.reshape(tf.stack([cosy, zero, siny,
                                     zero, one, zero,
                                     -siny, zero, cosy], axis=1),
                            [self.config.batch_size, 3, 3])
            Rx = tf.reshape(tf.stack([one, zero, zero,
                                     zero, cosx, -sinx,
                                     zero, sinx, cosx], axis=1),
                            [self.config.batch_size, 3, 3])
            print_shape(Rz)
            print_shape(Ry)
            print_shape(Rx)
            Rcam=tf.matmul(tf.matmul(Rx,Ry), Rz, name="Rcam") # [batch_size, 3, 3]
            print_shape(Rcam)

            # do transformation, coords_3d: [batch_size, keypoints_num, channel=3]
            # rotation
            coords_3d_t = tf.transpose(coords_3d, perm=[0,2,1], name="coords_3d_t")
            coords_3d_mm=tf.batch_matmul(Rcam, coords_3d_t, name="coords_3d_mm")
            coords_3d_rot=tf.transpose(coords_3d_mm, perm=[0,2,1], name="coords_3d_rot")
            # translation [batch_size, keypoints_num, channel=3]
            coords_3d_trans=tf.add(coords_3d_rot, tcam_tiled, name="coords_3d_trans") 
           
        if not eval:
          return coords_3d_trans
        else:
          avg_limb = 329.0 
          print("coord_3d_rot", coords_3d_rot.get_shape())
           
          diff_left = tf.slice(coords_3d_rot, [0, 2, 0], [-1, 1, 3]) - tf.slice(coords_3d_rot, [0, 12, 0], [-1, 1, 3]) 
          print("diff_left", diff_left.get_shape())
          diff_right = tf.slice(coords_3d_rot, [0, 3, 0], [-1, 1, 3]) - tf.slice(coords_3d_rot, [0, 13, 0], [-1, 1, 3]) 
          avg = (tf.sqrt(tf.reduce_sum(tf.square(diff_left), [1, 2])) + 
                 tf.sqrt(tf.reduce_sum(tf.square(diff_right), [1, 2])))/2
          
          scale = avg_limb/avg

          coords_3d_rot = coords_3d_rot * tf.reshape(scale, [-1, 1, 1])          
          return coords_3d_trans, coords_3d_rot, scale


    def _3D_2D_Projector(self, extern_params, coords_3d_trans):
        print("==========_3D_2D_Projector==========")
        with tf.variable_scope("3D_2D_Projector"):
            [sinx, siny, sinz, cosx, cosy, cosz, tx, ty, tz] = tf.unstack(extern_params, axis=1)
            tx = tf.reshape(tx, [self.config.batch_size, 1, 1])

            y0 = (self.config.image_size_w-1)/2
            z0 = (self.config.image_size_h-1)/2
            [X, Y, Z] = tf.split(coords_3d_trans, 3, axis=2, name="splitXYZ")
            print_shape(X)
            print_shape(Y)
            print_shape(Z)

            y_tmp = Y/tx
            y = Y/tx * y0 + y0
            z = Z/tx * z0 + z0
            print_shape(y)
            print_shape(z)

            coords_2d=tf.concat([y,z], 2,name="camera_projection") #[batch_size, keypoints_num, channel=2]
            print_shape(coords_2d)  
        return coords_2d, y_tmp


    def Loss_2D(self, predicts, labels):
        # labels: [batch_size, keypoints_num, channel=2]
        # predicts: [batch_size, keypoints_num, channel=2]
        labels_r = tf.reshape(labels, [-1,2]) # [batch_size*keypoints_num, channel=2]
        predicts_r = tf.reshape(predicts, [-1,2]) # [batch_size*keypoints_num, channel=2]
        x_gt, y_gt = tf.split(labels_r, 2, 1) 
        print_shape(x_gt) 
        print_shape(y_gt)
        x,y = tf.split(predicts_r, 2, 1) 
        print_shape(x)
        print_shape(y)

        loss = tf.reduce_mean(tf.sqrt(tf.square(x_gt-x) + tf.square(y_gt-y)))
        return loss


    def Loss_3D(self, predicts, labels):
        # labels: [batch_size, keypoints_num ,channel=3]
        # predicts: [batch_size, keypoints_num, channel=3]
        labels_r = tf.reshape(labels, [-1, 3]) # [batch_size*keypoints_num, channel=2]
        predicts_r = tf.reshape(predicts, [-1, 3]) # [batch_size*keypoints_num, channel=2]
        X_gt, Y_gt, Z_gt = tf.split(labels_r, 3, 1) 
        print_shape(X_gt) 
        print_shape(Y_gt)
        print_shape(Z_gt)
        X, Y, Z = tf.split(predicts_r, 3, 1) 
        print_shape(X)
        print_shape(Y)
        print_shape(Z)

        loss = tf.reduce_mean(tf.square(X_gt-X) + tf.square(Y_gt-Y) + tf.square(Z_gt-Z))
        #loss = tf.reduce_mean(tf.sqrt(tf.square(X_gt-X) + tf.square(Y_gt-Y) + tf.square(Z_gt-Z)))
        return loss


    def Loss_Heatmap(self, predicts, labels):
        # labels:  [batch_size, h, w, keypoints_num]
        # predicts:  [batch_size, h, w, keypoints_num]
        predicts_r = tf.reshape(predicts, [-1])
        labels_r = tf.reshape(labels, [-1])
        loss = tf.reduce_mean(tf.square(predicts_r - labels_r))
        return loss

    def train(self, config):
        print("-----------------")
        print("started the train")
        print("-----------------")
        """Training"""
        """
        recon_optim = tf.train.AdamOptimizer(config.learning_rate, beta1 = config.beta1) \
                                .minimize(self.recon_loss, global_step=self.global_step)
        """
        t_vars = tf.trainable_variables() 
        
        i_vars = [var for var in t_vars if "f_" not in var.name]
        # flownet variables
        #f_vars = [var for var in t_vars if "f_" in var.name]
        #self.saver_f = tf.train.Saver(f_vars)  
        
        self.saver_i = tf.train.Saver(i_vars + [self.global_step])  
        sup_optim = tf.train.AdamOptimizer(config.learning_rate,\
                                           beta1 = config.beta1) \
            .minimize(self.sup_loss, global_step=self.global_step, var_list=i_vars)
        #flow_optim = tf.train.GradientDescentOptimizer(config.learning_rate) \
        #                        .minimize(self.flow_loss, global_step=self.global_step,
        #                                  var_list=f_vars)
        if self.is_unsup_train: 
            recon_optim = tf.train.AdamOptimizer(config.learning_rate,\
                                                 beta1 = config.beta1)\
                .minimize(self.recon_loss, global_step=self.global_step, var_list=i_vars)
        #recon_optim = tf.train.GradientDescentOptimizer(config.learning_rate) \
        #                        .minimize(self.recon_loss, global_step=self.global_step,
        #                                  var_list=i_vars)

        init_op = tf.group(tf.global_variables_initializer(), 
                           tf.local_variables_initializer())
        self.sess.run(init_op)
       
        # Start input enqueue threads.
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)

        #TrainSummary = tf.summary.merge_all()
        #self.writer = tf.summary.FileWriter(os.path.join(config.logs_dir, config.name), self.sess.graph)

        #self.writer_heatmap = tf.train.SummaryWriter(self.config.log_dir+"/01_input_heatmap", self.sess.graph)

        """load data"""
        
        # facet 
        f_ = self.f 
        
        if self.config.is_dryrun: 
            batch_pose, batch_beta, batch_T, batch_R, batch_J, batch_J_2d, batch_image,\
                batch_seg, batch_chamfer, batch_c, batch_f, batch_resize_scale,\
                batch_gender, batch_pmesh, batch_v_gt = \
            self.sess.run([ self.pose_sr, self.beta_sr, self.T_sr, self.R_sr, self.J_sr, 
                            self.J_2d_sr, self.image_sr, self.seg_sr, self.chamfer_sr, 
                            self.c_sr, self.f_sr, self.resize_scale_sr, self.gender_sr, 
                            self.pmesh_sr, self.v_gt_sr])
            # 480 x 640 x 3 
            # get v for visibility
            start = time.time()
            v = self.sess.run(self.v, feed_dict={self.beta_gt:batch_beta, 
                                                 self.pose_gt:batch_pose, 
                                                 self.T_gt: batch_T, 
                                                 self.R_gt:batch_R,
                                                 self.gender_gt: batch_gender,
                                                 self.J_2d_gt: batch_J_2d,
                                                 self.c_gt: batch_c,
                                                 self.f_gt: batch_f,
                                                 self.resize_scale_gt: batch_resize_scale,
                                                 self.images:batch_image})
            print "construct 3d model takes %.2f secs" %(time.time() - start)          
            tf_vis, tf_am = self.sess.run([self.tf_visibility, self.tf_arg_min], 
                                feed_dict={self.beta_gt:batch_beta, 
                                           self.pose_gt:batch_pose, 
                                           self.T_gt: batch_T, 
                                           self.R_gt:batch_R,
                                           self.gender_gt: batch_gender,
                                           self.J_2d_gt: batch_J_2d,
                                           self.c_gt: batch_c,
                                           self.f_gt: batch_f,
                                           self.resize_scale_gt: batch_resize_scale,
                                           self.images:batch_image})
            print "infer visibility takes %.2f secs" %(time.time() - start)
            h, J_ori, J, pixel_loss, d3_loss, d3_joint_loss, d2_loss, d2_joint_loss, project1, project_mesh0, project_mesh1, pixel0, pixel1, flow, silh_loss, S_M1, C_M1\
             = self.sess.run([self.heatmaps[0], self.depth_J[0], self.J[0], self.pixel_loss,\
            self.d3_loss, self.d3_joint_loss, self.d2_loss, self.d2_joint_loss, self.project1,self.project_mesh0, self.project_mesh1, self.pixel0, self.pixel1, self.flow, self.silh_loss, self.S_M[0], self.C_M[0]], 
            feed_dict={self.beta_gt:batch_beta, self.pose_gt:batch_pose, 
                       self.T_gt: batch_T, self.R_gt:batch_R,
                       self.gender_gt:batch_gender,
                       self.J_gt: batch_J, self.J_2d_gt: batch_J_2d,
                       self.seg_gt:batch_seg, self.f_gt: batch_f,
                       self.c_gt: batch_c,
                       self.pmesh_gt: batch_pmesh,
                       self.chamfer_gt: batch_chamfer,
                       self.images:batch_image,
                       self.v_gt: batch_v_gt,
                       self.resize_scale_gt: batch_resize_scale})
            print "infer projections and losses take %.2f secs" %(time.time() - start)
            sio.savemat(os.path.join(self.sample_dir, "output.mat"), \
                     mdict={'flow': flow, 'J_2d': batch_J_2d, \
                    'project1': project1, 'v': v[0], 'visibility': tf_vis, \
                    'J':J, 'batch_J': batch_J, "image": batch_image, \
                    'S_M0': S_M1, 'seg': batch_seg, 'C_M0': C_M1, \
                    'chamfer': batch_chamfer, "project_mesh0":project_mesh0, \
                    "project_mesh1":project_mesh1, 'pixel0': pixel0, 'pixel1': pixel1})
       
            print("d3_loss: %.4f (%.4f), d2_loss: %.4f (%.4f), pixel_loss: %.4f,"
                  " silh_loss: %.4f" %(d3_joint_loss, d3_loss, d2_joint_loss, \
                                       d2_loss, pixel_loss, silh_loss))
            return 

        start_time = time.time()
        if self.load(self.checkpoint_dir, self.saver_i):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
            if self.config.model_dir:
              if self.load(self.config.model_dir, self.saver_i):
                print(" [*] Load pretrained SUCCESS: ", self.config.model_dir)
              else:
                print(" [!] Load pretrained failed...", self.config.model_dir)
                return

        try:
          while not coord.should_stop():
            tf_vis = 0
            pixel_loss = 0
            for idx in xrange(0, config.max_iter):
              #start = time.time()
              # load training data
              batch_pose, batch_beta, batch_T, batch_R, batch_J, batch_J_2d, batch_image,\
                batch_seg, batch_chamfer, batch_c, batch_f, batch_resize_scale, \
                batch_gender, batch_J_c, batch_v_gt = \
                self.sess.run([ self.pose_sr, self.beta_sr, self.T_sr, self.R_sr, self.J_sr,
                                self.J_2d_sr, self.image_sr, self.seg_sr, self.chamfer_sr,
                                self.c_sr, self.f_sr, self.resize_scale_sr, self.gender_sr,
                                self.J_c_sr, self.v_gt_sr])
              # load validation/testing data
              batch_pose_v, batch_beta_v, batch_T_v, batch_R_v, batch_J_v, batch_J_2d_v, \
                  batch_image_v, batch_seg_v, batch_chamfer_v, batch_c_v, batch_f_v, \
                  batch_resize_scale_v, batch_gender_v, batch_J_c_v, idx_v, batch_pmesh_v,\
                  batch_v_gt_v = \
                  self.sess.run([ self.pose_sr_v, self.beta_sr_v, self.T_sr_v, self.R_sr_v, 
                                  self.J_sr_v, self.J_2d_sr_v, self.image_sr_v, self.seg_sr_v,
                                  self.chamfer_sr_v, self.c_sr_v, self.f_sr_v, 
                                  self.resize_scale_sr_v, self.gender_sr_v, self.J_c_sr_v, 
                                  self.idx_sr_v, self.pmesh_sr_v, self.v_gt_v])
              if self.config.is_sup_train:
                _, step, sup_loss, d3_loss, d2_loss, beta_ \
                   = self.sess.run([sup_optim, self.global_step, self.sup_loss, self.d3_loss, self.d2_loss, self.beta[0]], 
                   feed_dict={self.beta_gt:batch_beta, self.pose_gt:batch_pose, 
                         self.T_gt: batch_T, self.R_gt:batch_R,
                         self.gender_gt:batch_gender,
                         self.J_gt: batch_J, self.J_2d_gt: batch_J_2d,
                         self.seg_gt:batch_seg, self.f_gt: batch_f,
                         self.c_gt: batch_c,
                         #self.chamfer_gt: batch_chamfer,
                         self.images:batch_image, 
                         self.resize_scale_gt: batch_resize_scale})
              if self.is_unsup_train:

                _, step, sup_loss, d3_loss, d2_loss, beta_, tf_vis \
                   = self.sess.run([recon_optim, self.global_step, self.sup_loss, self.d3_loss, self.d2_loss, self.beta[0], self.tf_visibility], 
                   feed_dict={self.beta_gt:batch_beta_v, self.pose_gt:batch_pose_v, 
                         self.T_gt: batch_T_v, self.R_gt:batch_R_v,
                         self.gender_gt:batch_gender_v,
                         self.J_gt: batch_J_v, self.J_2d_gt: batch_J_2d_v,
                         self.seg_gt:batch_seg_v, self.f_gt: batch_f_v,
                         self.c_gt: batch_c_v,
                         self.pmesh_gt:batch_pmesh_v,
                         self.chamfer_gt: batch_chamfer_v,
                         self.images:batch_image_v, 
                         self.resize_scale_gt: batch_resize_scale_v})
                #print "time:", time.time() - start
                # print out everything 
              if idx %10 == 0: 
                # get v for visibility
                #start = time.time() 
                # if there is only supervised training, do not predict chamfer and
                # visibility to save time
                if self.is_unsup_train:
                  step, summ_str, sup_loss, v, J, d3_loss, d3_joint_loss, d3_c_loss, d2_loss,\
                  d2_joint_loss, project1, flow, silh_loss, S_M1, C_M1, beta_loss, pose_loss,\
                  R_loss, T_loss\
                     = self.sess.run([self.global_step, self.syn_summary, self.sup_loss, self.v[0],
                     self.J[0], self.d3_loss, self.d3_joint_loss, self.centered_d3_joint_loss, self.d2_loss, self.d2_joint_loss,\
                     self.project1, self.flow, self.silh_loss, self.S_M[0], self.C_M[0],\
                     self.beta_loss, self.pose_loss, self.R_loss, self.T_loss], 
                     feed_dict={self.beta_gt:batch_beta, self.pose_gt:batch_pose, 
                           self.T_gt: batch_T, self.R_gt:batch_R,
                           self.gender_gt:batch_gender,
                           self.J_gt: batch_J, self.J_c_gt: batch_J_c, 
                           self.J_2d_gt: batch_J_2d,
                           self.seg_gt:batch_seg, self.f_gt: batch_f,
                           self.c_gt: batch_c,
                           self.v_gt:batch_v_gt,
                           self.chamfer_gt: batch_chamfer,
                           self.images:batch_image, 
                           self.resize_scale_gt: batch_resize_scale})
                  self.writer.add_summary(summ_str, step)
                  print("[%s, iter: %d] sup_loss: %.4f, d3_loss: %.4f (%.6f) (%.4f), d2_loss: %.4f (%.6f), "
                      "pixel_loss: %.4f, silh_loss: %.4f, beta_loss: %.4f, pose_loss: %.4f, R_loss:%.4f, T_loss: %.4f" \
                      %(self.config.name, idx, sup_loss, d3_joint_loss, d3_loss, d3_c_loss, d2_joint_loss, d2_loss, pixel_loss, silh_loss, beta_loss, pose_loss, R_loss, T_loss))
                  #print "time2:", time.time() - start
                
       
                  step, summ_str, sup_loss, v, J, d3_loss, d3_joint_loss, d3_c_loss, d2_loss,\
                  d2_joint_loss, project1, flow, silh_loss, S_M1, C_M1, beta_loss, pose_loss,\
                  R_loss, T_loss, pixel_loss, project_mesh0, project_mesh1, pixel0, pixel1\
                     = self.sess.run([self.global_step, self.syn_v_summary, self.sup_loss, self.v[0],
                     self.J[0], self.d3_loss, self.d3_joint_loss, self.centered_d3_joint_loss, self.d2_loss, self.d2_joint_loss,\
                     self.project1, self.flow, self.silh_loss, self.S_M[0], self.C_M[0],\
                     self.beta_loss, self.pose_loss, self.R_loss, self.T_loss, self.pixel_loss,
                     self.project_mesh0, self.project_mesh1, self.pixel0, self.pixel1], 
                     feed_dict={self.beta_gt:batch_beta_v, self.pose_gt:batch_pose_v, 
                         self.T_gt: batch_T_v, self.R_gt:batch_R_v,
                         self.gender_gt:batch_gender_v,
                         self.J_gt: batch_J_v, self.J_c_gt: batch_J_c_v, 
                         self.J_2d_gt: batch_J_2d_v,
                         self.seg_gt:batch_seg_v, self.f_gt: batch_f_v,
                         self.c_gt: batch_c_v,
                         self.v_gt:batch_v_gt_v,
                         self.pmesh_gt:batch_pmesh_v,
                         self.chamfer_gt: batch_chamfer_v,
                         self.images:batch_image_v, 
                         self.resize_scale_gt: batch_resize_scale_v})
                  self.writer.add_summary(summ_str, step)
                  print("[test, iter: %d] sup_loss: %.4f, d3_loss: %.4f (%.6f)(%.4f), d2_loss: %.4f (%.6f), "
                    "pixel_loss: %.4f, silh_loss: %.4f, beta_loss: %.4f, pose_loss: %.4f, R_loss:%.4f, T_loss: %.4f" \
                    %(idx, sup_loss, d3_joint_loss, d3_loss, d3_c_loss, d2_joint_loss, d2_loss, pixel_loss, silh_loss, beta_loss, pose_loss, R_loss, T_loss))
                 
                else: # training with only supervision
                    # dump results from training and validation data
                    step, summ_str, sup_loss, v, J, d3_loss, d3_joint_loss, d3_c_loss, \
                    d2_loss, d2_joint_loss, beta_loss, pose_loss, R_loss, T_loss\
                        = self.sess.run([self.global_step, self.syn_summary, self.sup_loss,\
                        self.v[0], self.J[0], self.d3_loss, self.d3_joint_loss, \
                        self.centered_d3_joint_loss, self.d2_loss, self.d2_joint_loss,\
                        self.beta_loss, self.pose_loss, self.R_loss, self.T_loss], 
                        feed_dict={self.beta_gt:batch_beta, self.pose_gt:batch_pose, 
                            self.T_gt: batch_T, self.R_gt:batch_R,
                            self.gender_gt:batch_gender,
                            self.J_gt: batch_J, self.J_c_gt: batch_J_c, 
                            self.J_2d_gt: batch_J_2d,
                            self.seg_gt:batch_seg, self.f_gt: batch_f,
                            self.c_gt: batch_c,
                            self.v_gt:batch_v_gt,
                            self.chamfer_gt: batch_chamfer,
                            self.images:batch_image, 
                            self.resize_scale_gt: batch_resize_scale})
                    self.writer.add_summary(summ_str, step)
                    print("[%s, step: %d] sup_loss: %.4f, d3_loss: %.4f (%.6f) (%.4f),"
                        " d2_loss: %.4f (%.6f), beta_loss: %.4f, pose_loss: %.4f, "
                        "R_loss:%.4f, T_loss: %.4f" \
                        %(self.config.name, step, sup_loss, d3_joint_loss, d3_loss, \
                          d3_c_loss, d2_joint_loss, d2_loss, beta_loss, pose_loss, \
                          R_loss, T_loss))
                    step, summ_str, sup_loss, v, J, d3_loss, d3_joint_loss, d3_c_loss, \
                    d2_loss, project1, flow, d2_joint_loss, beta_loss, pose_loss, R_loss,\
                    T_loss, project_mesh0, project_mesh1, pixel0, pixel1\
                        = self.sess.run([self.global_step, self.syn_v_summary, \
                        self.sup_loss, self.v[0], self.J[0], self.d3_loss, \
                        self.d3_joint_loss, self.centered_d3_joint_loss, self.d2_loss,\
                        self.project1, self.flow, self.d2_joint_loss, self.beta_loss, \
                        self.pose_loss, self.R_loss, self.T_loss, self.project_mesh0, \
                        self.project_mesh1, self.pixel0, self.pixel1], 
                        feed_dict={self.beta_gt:batch_beta_v, self.pose_gt:batch_pose_v, 
                            self.T_gt: batch_T_v, self.R_gt:batch_R_v,
                            self.gender_gt:batch_gender_v,
                            self.J_gt: batch_J_v, self.J_c_gt: batch_J_c_v, 
                            self.J_2d_gt: batch_J_2d_v,
                            self.seg_gt:batch_seg_v, self.f_gt: batch_f_v,
                            self.c_gt: batch_c_v,
                            self.v_gt:batch_v_gt_v,
                            self.pmesh_gt:batch_pmesh_v,
                            self.chamfer_gt: batch_chamfer_v,
                            self.images:batch_image_v, 
                            self.resize_scale_gt: batch_resize_scale_v})
                    self.writer.add_summary(summ_str, step)
                    print "model name: %s" %(self.config.name)
                    print("[test, iter: %d] sup_loss: %.4f, d3_loss: %.4f (%.6f)(%.4f),"
                          "d2_loss: %.4f (%.6f), beta_loss: %.4f, pose_loss: %.4f, "
                          "R_loss:%.4f, T_loss: %.4f" \
                          %(idx, sup_loss, d3_joint_loss, d3_loss, d3_c_loss, \
                            d2_joint_loss, d2_loss, beta_loss, pose_loss, R_loss, T_loss))
                  

              if step %1000 == 0: 
                # save results in mat
                if self.is_unsup_train:
                  sio.savemat(os.path.join(self.sample_dir, "output" + str(int(idx)) + ".mat"), \
                    mdict={'flow': flow, 'J_2d': batch_J_2d_v, \
                    'project1': project1, 'v': v, 'visibility': tf_vis, \
                    'J':J, 'batch_J': batch_J_v, "image": batch_image_v, \
                    'S_M0': S_M1, 'seg': batch_seg_v, 'C_M0': C_M1, 'chamfer': batch_chamfer_v,
                    'project_mesh0':project_mesh0, "project_mesh1":project_mesh1, "pixel0": pixel0, 'pixel1':pixel1})
                else:
                  sio.savemat(os.path.join(self.sample_dir, "output" + str(int(idx)) + ".mat"), \
                    mdict={'flow': flow, 'J_2d': batch_J_2d_v, \
                    'project1': project1, 'v': v, \
                    'J':J, 'batch_J': batch_J_v, "image": batch_image_v, \
                    'seg': batch_seg_v, 'chamfer': batch_chamfer_v,
                    'project_mesh0':project_mesh0, "project_mesh1":project_mesh1, "pixel0": pixel0, 'pixel1':pixel1})

              if (step)%5000 == 0:
                self.save(self.checkpoint_dir, step)
            break
        except tf.errors.OutOfRangeError:
            print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs, step))
        finally:
            # When done, ask the threads to stop.
            coord.request_stop()

        # Wait for threads to finish.
        coord.join(threads)
        self.sess.close()
    def save_obj(self, filename, v, f):
        outmesh_path = os.path.join(self.sample_dir, filename)     
        with open( outmesh_path, 'w') as fp:    
            for v_ in v:
                fp.write( 'v %f %f %f\n' % ( v_[0], v_[1], v_[2]))
            for f_ in f:
                fp.write( 'f %d %d %d\n' %  (f_[0], f_[1], f_[2]) )

    def save(self, checkpoint_dir, step):
        model_name = "3D.model"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)


    def load(self, checkpoint_dir, saver):
        print(" [*] Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False        

    def numberHash(self, numberFloat):
        return colorsys.hsv_to_rgb(numberFloat, 1.0, 1.0)


    def visualize_joint_heatmap(self, bgimg, njoints, heatmaps, needNormalize=True):

        #normalize: make max to one (not sum to one)
        if(needNormalize):
            heatMapSize = heatmaps.get_shape().as_list()
            heatmapsMax = tf.reduce_max(heatmaps, [1, 2], True)
            heatmapsMax = tf.clip_by_value(heatmapsMax, 0.0000001, float('inf')) # prevent divide by zero
            heatmapsMax = tf.tile(heatmapsMax, [1, heatMapSize[1], heatMapSize[2], 1])
            heatmaps = tf.div(heatmaps, heatmapsMax)

        result = 0.7 * bgimg
        channels = tf.split(heatmaps, njoints, axis=3)

        for ch in range(0, njoints):
            rgb = self.numberHash(ch/float(njoints)) 
            rgbImg = tf.concat([channels[ch]*rgb[0], channels[ch]*rgb[1], channels[ch]*rgb[2]], 3)
            result = result + rgbImg

        # clamp over max value
        filled = tf.fill(tf.shape(result), 1.)
        result = tf.where(tf.greater(result, filled), filled, result)

        return result

    def visualize_joint_heatmap2(self, bgimg, njoints, heatmaps, needNormalize=True):

        #normalize: make max to one (not sum to one)
        if(needNormalize):
            heatMapSize = heatmaps.get_shape().as_list()
            heatmapsMax = tf.reduce_max(heatmaps, [1, 2], True)
            heatmapsMax = tf.clip_by_value(heatmapsMax, 0.0000001, float('inf')) # prevent divide by zero
            heatmapsMax = tf.tile(heatmapsMax, [1, heatMapSize[1], heatMapSize[2], 1])
            heatmaps = tf.div(heatmaps, heatmapsMax)

        result = 0.7 * bgimg
        channels = tf.split(heatmaps, njoints, 3)

        for ch in range(0, njoints):
            rgb = self.numberHash(ch/float(njoints)) 
            rgbImg = tf.concat([channels[ch]*rgb[0], channels[ch]*rgb[1], channels[ch]*rgb[2]], 3)
            result = result + rgbImg

        # clamp over max value
        filled = tf.fill(tf.shape(result), 1.)
        result = tf.where(tf.greater(result, filled), filled, result)

        return result
