import numpy as np
from scipy import misc
import math
from chamfer_utils import get_chamfer
import scipy.misc
class Data_Helper_h36_syn:

    def __init__(self, data, batch_size, num_frames, h, w, chamfer_scale=1.0, keypoints_num=17, bases_num=97 ,is_perm=True):
        self.h = h
        self.w = w 
        self.small_h = int(self.h * chamfer_scale) 
        self.small_w = int(self.w * chamfer_scale) 
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.keypoints_num = keypoints_num
        self.bases_num = bases_num
        self.chamfer_scale = chamfer_scale
        self.is_perm = is_perm        
        self.data_pose = data['pose'] #[batch, keypoints_num*2] (x1,x2,...xn, y1,y2,..yn, z1,z2,...zn)
        self.data_T = data['T'] #[batch, keypoints_num*3] (x1,y1,z1, ...)
        self.data_R = data['R'] #[batch, keypoints_num*3] (x1,y1,z1, ...)
        self.data_beta = data['beta'] #[batch, keypoints_num*3] (x1,y1,z1, ...)
        self.data_J = data['J'] #[batch, keypoints_num*3] (x1,y1,z1, ...)
        self.data_J_2d = data['J_2d'] #[batch, keypoints_num*3] (x1,y1,z1, ...)
        self.data_image = data['image'] #[batch, keypoints_num*3] (x1,y1,z1, ...)
        self.data_seg = data['seg'] #[batch, keypoints_num*3] (x1,y1,z1, ...)
        self.data_f = data['f'] #[batch, keypoints_num*3] (x1,y1,z1, ...)
        self.data_gender = data['gender'] 
        #self.xyz = data['xyzs'] 
        #self.weights = data['weights'] #[96,1]
        self.num_samples = self.data_T.shape[0]
        self.num_batches = self.num_samples//batch_size
        print("num_samples =", self.num_samples)
        print("batch_size =", batch_size)	
        print("num_batches =", self.num_batches)
        self.reset()

    def get_bases(self):
        #bases = np.zeros((self.bases_num + 1, self.keypoints_num, 3), dtype=np.float32)
        #for i in range(self.bases_num):
        #    bases[i,:,:] = np.reshape(self.baseShape[i, :], [self.keypoints_num, 3])
        
        #bases[self.bases_num + 1,:,:] = np.reshape(self.meanShape, [self.keypoints_num, 3])
        # waste memory here
        #batch_bases= np.expand_dims(bases, axis=0)
        #batch_bases = np.tile(batch_bases, (self.batch_size, 1, 1, 1))
        return self.baseShape, self.meanShape

    def reset(self):
        self.batch_id = 0
        if self.is_perm:
            self.perm = np.random.permutation(self.num_samples)
        else:
            self.perm = range(self.num_samples)

    def next(self):
        if self.batch_id >= self.num_batches:
            self.reset()
        
        images_perm = self.perm[int(self.batch_id * self.batch_size):int((self.batch_id + 1)*self.batch_size)]
        #batch_images = self.images[images_perm,:,:,:]
        batch_pose = np.zeros((self.batch_size, self.num_frames, self.keypoints_num, 3), dtype=np.float32)
        batch_T = np.zeros((self.batch_size, self.num_frames, 3), dtype=np.float32)
        batch_R = np.zeros((self.batch_size, self.num_frames, 6), dtype=np.float32)
        batch_gender = np.zeros((self.batch_size), dtype=np.int32)
        batch_beta = np.zeros((self.batch_size, self.num_frames, self.bases_num), dtype=np.float32)
        batch_J = np.zeros((self.batch_size, self.num_frames, self.keypoints_num, 3), dtype=np.float32)
        batch_J_2d = np.zeros((self.batch_size, self.num_frames, self.keypoints_num, 2), dtype=np.float32)
        batch_image = np.zeros((self.batch_size, self.num_frames, self.h, self.w, 3), dtype=np.float32)
        batch_seg = np.zeros((self.batch_size, self.num_frames, self.h, self.w), dtype=np.float32)
        batch_chamfer = np.zeros((self.batch_size, self.num_frames, self.small_h, self.small_w), dtype=np.float32)
        batch_f = np.zeros((self.batch_size, self.num_frames, 2), dtype=np.float32)
        batch_c = np.zeros((self.batch_size, self.num_frames, 2), dtype=np.float32)
        batch_resize_scale = np.zeros((self.batch_size, self.num_frames), dtype=np.float32)

        old_2d_center = np.array([(320 - 1)/2.0, (240-1)/2.0])
 
        for i in range(self.batch_size):
            id_ = images_perm[i] 
            #batch_labels_2d[i,:,0] = self.labels_2d[id_, 0:self.keypoints_num]
            #batch_labels_2d[i,:,1] = self.labels_2d[id_, self.keypoints_num:self.keypoints_num*2]
            batch_pose[i, :, :, :] = self.data_pose[id_, :, :, :]
            batch_gender[i] = self.data_gender[id_]
            batch_R[i, :, :3] = np.sin(self.data_R[id_, :, :])
            batch_R[i, :, 3:6] = np.cos(self.data_R[id_, :, :])
            batch_beta[i, :, :] = self.data_beta[id_, :, :]
            batch_f[i, :, :] = self.data_f[id_, :, :] 
            batch_T[i, :, :] = self.data_T[id_, :, :]
            batch_J[i, :, :, :] = self.data_J[id_, :, :, :]

            J_2d = self.data_J_2d[id_, :, :, :]
            new_2d_center = np.round(J_2d[0, 0, :] + 10 * (np.random.uniform((2)) - 1)) + 0.5*np.ones((2))
            crop_size = np.round(1.2 * np.max(np.abs(J_2d - np.reshape(new_2d_center, [1, 1, -1]))))
            new_image_size = int(2*crop_size)
            x_min = int(math.ceil(new_2d_center[0] - crop_size))
            x_max = int(math.floor(new_2d_center[0] + crop_size))
            y_min = int(math.ceil(new_2d_center[1] - crop_size))
            y_max = int(math.floor(new_2d_center[1] + crop_size))
            resize_scale = self.h/(crop_size * 2.0)
            batch_resize_scale[i, :] = resize_scale
            # frame_id * 2
            new_origin = np.array([x_min, y_min])
            batch_c[i, :, :] = np.reshape(old_2d_center - new_origin, [-1, 2]) 
            batch_J_2d[i, :, :, :] = resize_scale * (self.data_J_2d[id_, :, :, :] - np.reshape(new_origin, [1, 1, -1]))
            image = self.data_image[id_, :, :, :, :].astype(np.float32)
            seg_float = self.data_seg[id_, :, :, :].astype(np.float32)
            
            img_x_min = max(x_min, 0)
            img_x_max = min(x_max, 359)
            img_y_min = max(y_min, 0)
            img_y_max = min(y_max, 239)
            #print new_2d_center
            #print crop_size
            print(img_x_min, img_x_max, img_y_min, img_y_max)
            print(x_min, x_max, y_min, y_max)
       
            for frame_id in range(self.num_frames):
              
              crop_image = np.zeros((new_image_size, new_image_size, 3), dtype=np.float32) 
              crop_image[max(0, -y_min):max(0, -y_min) + img_y_max - img_y_min + 1, \
                         max(0, -x_min):max(0, -x_min) + img_x_max - img_x_min + 1, :] \
                         = image[frame_id, img_y_min:img_y_max + 1, img_x_min:img_x_max +1, :]
              batch_image[i, frame_id, :, :, :] = scipy.misc.imresize(crop_image, [self.h, self.w])
              crop_seg = np.zeros((new_image_size, new_image_size, 3), dtype=np.float32) 
              crop_seg[max(0, -y_min):max(0, -y_min) + img_y_max - img_y_min + 1, \
                       max(0, -x_min):max(0, -x_min) + img_x_max - img_x_min + 1] \
                       = np.expand_dims(seg_float[frame_id, img_y_min:img_y_max + 1, \
                                        img_x_min:img_x_max +1], 2)
              seg = scipy.misc.imresize(crop_seg, [self.h, self.w]) 
              seg[seg < 0.5] = 0
              seg[seg >= 0.5] = 1

              batch_seg[i, frame_id, :, :] = seg[:,:,0]
               
              # calculate chamfer dist
              #for img_id in range(self.num_frames):
              batch_chamfer[i, frame_id, :, :], _, _ = get_chamfer(seg[:,:,0], 
                                                                   self.chamfer_scale) 
     
        self.batch_id += 1
        return batch_pose, batch_T, batch_R, batch_beta, batch_J, batch_J_2d, batch_image/255.0,\
               batch_seg, batch_f, batch_chamfer, batch_c, batch_gender, batch_resize_scale

class Data_Helper_video_2d:

    def __init__(self, data, h, w, batch_size, keypoints_num=17 ,is_perm=True):
        self.h = h
        self.w = w
        self.batch_size = batch_size
        self.keypoints_num = keypoints_num
        self.is_perm = is_perm
        self.labels_2d = data['pos2d'] #[batch, keypoints_num*2] (x1,x2,...xn, y1,y2,..yn, z1,z2,...zn)
        self.labels_3d = data['pos3d']
        self.labels_videos = data['videos'] #[batch, keypoints_num*3] (x1,y1,z1, ...)
        self.baseShape = data['baseShape'] #[96, keypoints_num*3] (x1,y1,z1, ...)
        self.meanShape = data['meanShape'] #[1, keypoints_num*3] (x1,y1,z1, ...)
        
        self.num_samples = self.labels_2d.shape[0]
        self.num_batches = self.num_samples//batch_size
        print("num_samples =", self.num_samples)
        print("batch_size =", batch_size)	
        print("num_batches =", self.num_batches)
        self.reset()

    def reset(self):
        self.batch_id = 0
        if self.is_perm:
            self.perm = np.random.permutation(self.num_samples)
        else:
            self.perm = range(self.num_samples)
    def get_bases(self):
        return self.baseShape, self.meanShape

    def next(self):
        if self.batch_id >= self.num_batches:
            self.reset()
        images_perm = self.perm[int(self.batch_id * self.batch_size):int((self.batch_id + 1)*self.batch_size)]
        batch_labels_videos = np.zeros((self.batch_size, self.h, self.w, 3), dtype=np.float32)
        batch_labels_2d = np.zeros((self.batch_size, self.keypoints_num, 2), dtype=np.float32)
        batch_labels_3d = np.zeros((self.batch_size, self.keypoints_num, 3), dtype=np.float32)
        for i in range(self.batch_size):
            id_ = images_perm[i] 
            batch_labels_2d[i,:,:] = np.reshape(self.labels_2d[id_, :], [self.keypoints_num, 2]) 
            batch_labels_3d[i,:,:] = np.reshape(self.labels_3d[id_, :], [self.keypoints_num, 3]) 
            batch_labels_videos[i,:,:] = self.labels_videos[id_, :, :, :]
        self.batch_id += 1
        return batch_labels_videos/255.0, batch_labels_2d, batch_labels_3d
