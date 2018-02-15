import scipy.io as sio
from matrix_utils import avg_joint_error, rotationMatrixToEulerAngles, eulerAnglesToRotationMatrix
import imageio
#import matplotlib.pyplot as plt
import numpy as np
import os
#from visualize import visualize_smpl_2d, visualize_smpl_3d, visualize_smpl_mesh, visualize_smpl_3d_mesh
from smpl_webuser.serialization import load_model


#data_dir = './SURREAL/data/h36m/train/run0/'
smpl_dir = '/home/htung/Documents/2017/Spring/3d_smpl/smpl'

MALE = 1
def get_training_params(filename='', data_dir='../SURREAL/data/h36m/train/run0'):

  folder_name = filename[:-6] 
  data = sio.loadmat(os.path.join(os.path.join(data_dir, folder_name), filename) + "_info.mat") 
  segs = sio.loadmat(os.path.join(os.path.join(data_dir, folder_name), filename) + "_segm.mat") 

  #'h36m_S1_Directions/h36m_S1_Directions_c0028_info.mat' 
  # see whether it is male or female
  # gender: male(1), female(0)
  gender = data['gender'][0][0]
  if gender == MALE:
    smpl_model = "models/basicModel_m_lbs_10_207_0_v1.0.0.pkl"  
  else:
    smpl_model = "models/basicModel_f_lbs_10_207_0_v1.0.0.pkl"  
  m = load_model(os.path.join(smpl_dir, smpl_model)) 

  #import cv2
  cap = imageio.get_reader(os.path.join(os.path.join(data_dir, folder_name), filename) + ".mp4")

  #['zrot', 'camDist', 'bg', 'joints3D', 'pose', 'clipNo', 'joints2D', 'gender', 'sequence', '__header__,
  #'__globals__', 'source', 'shape', 'stride', 'light', '__version__', 'cloth', 'camLoc']  
  # one number data['camDist']
  # joints2D: 2x24x100 

  num_joints = data['joints2D'].shape[1]
  num_frames = data['joints2D'].shape[2]
  #plt.ion()
  import time
  import math
  w = 320
  h = 240
  all_pose = np.zeros((num_frames, num_joints * 3))
  all_beta = np.zeros((num_frames, 10))
  all_f = np.zeros((num_frames, 2))
  # center
  all_R = np.zeros((num_frames, 3))
  all_T = np.zeros((num_frames, 3))
  all_J = np.zeros((num_frames, num_joints, 3))
  all_J_2d = np.zeros((num_frames, num_joints, 2))
  all_seg = np.zeros((num_frames, h, w), dtype=np.bool_)
  all_image = np.zeros((num_frames, h, w, 3), dtype=np.uint8)

  for frame_id in range(num_frames):
 
    img = cap.get_data(frame_id)
    #print img.dtype
    seg = segs['segm_' + str(frame_id + 1)]
    idx = seg[:,:] > 0.5
    seg[idx] = 1
    seg = seg.astype(np.bool_)
    #print np.max(seg), np.min(seg)
    #seg_tmp = np.zeros((240, 320, 3))
    #seg_tmp[:,:,:] = np.reshape(seg.astype(np.float32), [240, 320, 1])
    #import scipy
    #scipy.misc.imsave("seg.png", seg_tmp)
    
    #if frame_id != 20:
    #  continue
    pose = data['pose'][:, frame_id]
    pose[:3] = [math.pi, 0, 0]
    m.pose[:] = pose
    m.betas[:] = data['shape'][:, frame_id]
  
    #get 3d mesh and body joints before R,T
    mesh = np.reshape(m.r, [-1, 3]).T
    J = np.array(m.J_transformed).T 
    #print "beta", np.array(m.shapedirs)
    #fig = plt.figure(5)
    #visualize_smpl_3d_mesh(J, mesh, title="init_mesh", fig=fig)
    # flip image and 2d gt
    img = np.fliplr(img)
    seg = np.fliplr(seg)
    d2 = data['joints2D'][:,:,frame_id]
    d2[0, :] =  (320 - d2[0,:])
    #visualize_smpl_2d(d2, bg=img, figure_id=10, title="2d gt") 
  
    from mpl_toolkits.mplot3d import Axes3D
    d3 = data['joints3D'][:,:,frame_id]
    matrix_world = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    d3 = np.matmul(matrix_world, d3) + np.reshape(np.array([0, 1, -(data['camDist']-3)]), [3,1])
    r_camera = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
    d3 = np.matmul(r_camera,d3)
    d3_tmp = d3
   
    new_center = np.zeros((3,1))
    new_center[:2] = np.expand_dims(d3[:2, 0], 1)
    d3 = d3 - new_center
    
    angle_x = -0.00102
    R_x = [[1,0,0], 
           [0, math.cos(angle_x), -math.sin(angle_x)], 
           [0, math.sin(angle_x), math.cos(angle_x)]] 
    d3 = np.matmul(R_x,d3)

    #visualize_smpl_3d(d3, autoscale=True)
    d3_project = d3[:2, :]/(d3[2,:])
    #center 2d ground truth
    centered_2d = d2 - np.reshape(d2[:, 0], [2, 1])
    #visualize_smpl_2d(centered_2d, xlim=(-160, 160), ylim=(-120, 120), figure_id=11, title="centered 2d") 
    sol = np.linalg.lstsq(np.reshape(d3_project[0, :], [-1, 1]), np.reshape(centered_2d[0,:], [-1, 1]))[0] 
    fx = sol[0][0]
    sol = np.linalg.lstsq(np.reshape(d3_project[1, :], [-1, 1]), np.reshape(centered_2d[1,:], [-1, 1]))[0] 
    fy = sol[0][0] #np.sum(centered_2d[1,:])/np.sum(d3_project[1,:])
    d3_project[0,:] *= fx
    d3_project[1,:] *= fy
    #visualize_smpl_2d(d3_project, xlim=(-160, 160), ylim=(-120, 120), title="projected 3d", figure_id=3)
    #print "piexl error", avg_joint_error(d3_project - centered_2d)
    T2 = np.zeros((3,1))
    T2[0] = (d2[0, 0] - 160) /fx * d3[2, 0]
    T2[1] = (d2[1, 0] - 120) /fy * d3[2, 1]


     
    #print fx, fy

    # smpl to 3d center
    J_ones = np.concatenate((J, np.ones((1, num_joints))), 0)
    R_T = np.linalg.lstsq(J_ones.T, d3.T)[0].T
    R = R_T[:3, :3]
    T = np.reshape(R_T[:, 3], [3,1]) 
    #reconstruct_3d = np.matmul(R, J) + np.reshape(T, [3,1])
    #print "3d recon error", np.mean(np.sqrt(np.sum(np.square(d3 - reconstruct_3d), 0)))
    T += T2
 
    angles = rotationMatrixToEulerAngles(R)
    #print angles 
    R_rec = eulerAnglesToRotationMatrix(angles)
  
    reconstruct_3d = np.matmul(R_rec, J) + np.reshape(T, [3,1]) 
    #print "recovered R error", np.mean(np.sqrt(np.sum(np.square(d3 - reconstruct_3d), 0)))
  
    #visualize_smpl_3d(reconstruct_3d, xlim = (T[0] -1, T[0] + 1), ylim = (T[1] -1, T[1] +1), 
    #                  zlim=(T[2] - 1, T[2] + 1),  title="recovered_3d")
    # use reconstruct_3d to do projection and get2d
    reconstruct_2d = reconstruct_3d[:2, :]/reconstruct_3d[2,:]
    
    reconstruct_2d *= np.array([[fx],[fy]])
    center = np.array([[160], [120]])
    reconstruct_2d += center

    #print "2d error", np.mean(np.sqrt(np.sum(np.square(reconstruct_2d-d2), 0)))
    #visualize_smpl_2d(reconstruct_2d, bg=img, figure_id=12, title="2d reconstructed") 
    #plt.pause(0.1)
    # things to store
    all_pose[frame_id, :] = pose
    all_beta[frame_id, :] = data['shape'][:, frame_id]
    all_f[frame_id] = [fx, fy]
    # center
    all_R[frame_id, :] = angles
    all_T[frame_id, :] = T[:,0]
    all_J[frame_id, :, :] = reconstruct_3d.T 
    all_J_2d[frame_id, :, :] = d2.T #reconstruct_2d.T 
    all_seg[frame_id, :, :] = seg
    all_image[frame_id, :, :, :] = img
  output = dict()
  output['pose'] = all_pose
  output['beta'] = all_beta
  output['f'] = all_f
  output['R'] = all_R
  output['T'] = all_T
  output['J'] = all_J
  output['J_2d'] = all_J_2d
  output['seg'] = all_seg
  output['image'] = all_image
  output['gender'] = gender
  return output

#get_training_params('h36m_S1_Directions_c0002')
