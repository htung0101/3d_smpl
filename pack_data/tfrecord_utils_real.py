import numpy as np
import pickle
import os 
import random
#from write_utils import read_syn_to_bin
from write_utils import read_syn_to_bin2
import struct 
from tqdm import tqdm
import tensorflow as tf
import math 
import scipy.misc
import sys
import scipy.io
sys.path.insert(0, '../')
from chamfer_utils import get_chamfer

def get_file_list(data_path, quo =0, test=False):
  files = []
  num = 0
  #for folder in tqdm(os.listdir(data_path)):
    #condition = "_S9_" not in folder
    #if test:
    #  condition = "_S9_" in folder and "1" not in folder
    #if condition:
      #num += 1
      #p = os.path.join(data_path, folder)
  for filename in os.listdir(data_path):#p):
     #if filename.startswith("output_S1_c54138969_Directions.bin"):  
     if filename.startswith("output_S9_"):#h36m_S1_Directions_c0002"):
      if " " not in filename:
        with open(os.path.join(data_path, filename), 'rb') as f_:
          #line = f_.read(4) # gender
          line = f_.read(4) # num_frames
          num_frames = struct.unpack('f', line)[0]
          #if num_frames != 100:
          #print filename, "nframes", num_frames
        """
        if test:
          for frame_id in range(num_frames - 1):
            if frame_id %20 == quo:
              #    files.append(os.path.join(p, filename) + "#" + str(frame_id))
              #fid = np.random.randint(num_frames-1)
              files.append(os.path.join(p, filename) + "#" + str(frame_id))
        else:
        """
        fid = np.random.randint(num_frames-1)
        files.append(os.path.join(data_path,filename) + "#" + str(fid))
        fid = np.random.randint(num_frames-1)
        files.append(os.path.join(data_path,filename) + "#" + str(fid))

        #files.append(os.path.join(p, filename) + "#" + str(fid))
  #print("files-----",files)   
  #print "number of folder", num
  return files


def convert_to_npz_from_folder(folder_name, npz_filename, get_samples=None, quo=0):
  files = get_file_list(folder_name, quo)
  random.shuffle(files)

  num_files = len(files)
  num_frames = 2
  image_size_h = 240
  image_size_w = 320

  keypoints_num = 24
  bases_num = 10
  print "num_files", num_files
  if not get_samples:
    get_samples = num_files
  data_pose = np.zeros((get_samples, num_frames, keypoints_num, 3))
  data_T = np.zeros((get_samples, num_frames, 3))
  data_R = np.zeros((get_samples, num_frames, 3))
  data_beta = np.zeros((get_samples, num_frames, bases_num))
  data_J = np.zeros((get_samples, num_frames, keypoints_num, 3))
  data_J_2d = np.zeros((get_samples, num_frames, keypoints_num, 2))
  data_image = np.zeros((get_samples, num_frames, image_size_h, image_size_w, 3), dtype = np.uint8)
  data_seg = np.zeros((get_samples, num_frames, image_size_h, image_size_w), dtype=np.bool_)
  data_f = np.zeros((get_samples, num_frames, 2))
  data_gender = np.zeros((get_samples), dtype=np.int32)

  for sample_id in tqdm(range(get_samples)):
    filename, t = files[sample_id].split("#")
    print filename, t, int(t)
    output = dict()
    output[0] = read_syn_to_bin(filename, int(t))
    output[1] = read_syn_to_bin(filename, int(t) + 1)
    print "start!!@@@@@"
    scipy.io.savemat('tf_test1.mat', {'output':output[0]})
    for frame_id in range(num_frames): 
      pose = np.reshape(output[frame_id]['pose'], [keypoints_num, 3]) 
      data_pose[sample_id, frame_id, :, :] = pose
      data_gender[sample_id] = int(output[frame_id]['gender'])
      data_T[sample_id, frame_id, :] = output[frame_id]['T']
      data_R[sample_id, frame_id, :] = output[frame_id]['R']
      data_beta[sample_id, frame_id, :] = output[frame_id]['beta']
      data_J[sample_id, frame_id, :, :] = output[frame_id]['J']
      data_J_2d[sample_id, frame_id, :, :] = output[frame_id]['J_2d']
      data_image[sample_id, frame_id, :, :, :] = output[frame_id]['image']
      #print np.max(output['seg'][sample_id, :, :])
      data_seg[sample_id, frame_id, :, :] = output[frame_id]['seg']
      data_f[sample_id, frame_id, :] = output[frame_id]['f']


  np.savez(npz_filename, pose=data_pose, T=data_T, R=data_R, beta=data_beta, J=data_J,
           J_2d=data_J_2d, image=data_image, seg=data_seg, f=data_f, gender=data_gender)

def loadBatchSurreal_fromString(file_string, image_size=128, num_frames=2, \
                                keypoints_num=24, bases_num=10, chamfer_scale=0.5):
  
  filename, t = file_string.split("#")
  output = dict()
  output[0] = read_syn_to_bin2(filename, int(t))
  #print "start!!@@@@@"
  scipy.io.savemat('tf_test1.mat', {'output':output[0]}) 
  output[1] = read_syn_to_bin2(filename, int(t) + 1)
  #output[0] = read_syn_to_bin(filename, int(t))
  #output[1] = read_syn_to_bin(filename, int(t) + 1)
    
  data_pose = np.zeros((num_frames, keypoints_num * 3))
  data_T = np.zeros((num_frames, 3))
  data_R = np.zeros((num_frames, 6))
  data_beta = np.zeros((num_frames, bases_num))
  data_J = np.zeros((num_frames, keypoints_num, 3))
  data_J_2d = np.zeros((num_frames, keypoints_num, 2))
  data_image = np.zeros((num_frames, image_size, image_size, 3))
  data_seg = np.zeros((num_frames, image_size, image_size))
  small_image_size = int(chamfer_scale * image_size)
  data_chamfer = np.zeros((num_frames, small_image_size, small_image_size))
  data_f = np.zeros((num_frames, 2))
  data_c = np.zeros((num_frames, 2))
  data_resize_scale = np.zeros((num_frames))
  data_gender = np.zeros(())
  # Cropping
  old_2d_center = output[0]['c']#np.array([(320 - 1)/2.0, (240-1)/2.0])
  # Use keypoint 0 in frame1 as center
  J_2d = output[0]['J_2d']
  new_2d_center = np.round(J_2d[0, :]) + 0.5*np.ones((2))
  s = 1.2 #1.3 + 0.1 * np.random.rand()
  crop_size = np.round(s * np.max(np.abs(J_2d - np.reshape(new_2d_center, [1, 1, -1]))))
  new_image_size = int(2*crop_size)
  x_min = int(math.ceil(new_2d_center[0] - crop_size)) 
  x_max = int(math.floor(new_2d_center[0] + crop_size)) 
  y_min = int(math.ceil(new_2d_center[1] - crop_size))
  y_max = int(math.floor(new_2d_center[1] + crop_size))
  resize_scale = float(image_size)/(crop_size * 2.0) 
  data_resize_scale[:] = resize_scale
  new_origin = np.array([x_min, y_min]) 
  data_c[:, :] = np.reshape(old_2d_center - new_origin, [-1, 2])
  
  for frame_id in range(num_frames): 
      data_pose[frame_id, :] = output[frame_id]['pose']
      data_gender = int(output[frame_id]['gender'])
      data_R[frame_id, :3] = np.sin(output[frame_id]['R'])
      data_R[frame_id, 3:6] = np.cos(output[frame_id]['R'])
      #data_beta[frame_id, :] = output[frame_id]['beta']
      data_f[frame_id, :] = output[frame_id]['f']
      #data_c[frame_id, :] = output[frame_id]['c']
      data_T[frame_id, :] = output[frame_id]['T']
      data_J[frame_id, :, :] = output[frame_id]['J']
      data_J_2d[frame_id, :, :] = resize_scale * (output[frame_id]['J_2d'] - np.reshape(new_origin, [1, -1]))
      
      # crop image
      image = output[frame_id]['image']
      #print image
      h, w, _ = image.shape
      img_x_min = max(x_min, 0) 
      img_x_max = min(x_max, w -1)
      img_y_min = max(y_min, 0)
      img_y_max = min(y_max, h -1)          
      crop_image = np.zeros((new_image_size, new_image_size, 3), dtype=np.float32) 
      crop_image[max(0, -y_min):max(0, -y_min) + img_y_max - img_y_min + 1, \
                 max(0, -x_min):max(0, -x_min) + img_x_max - img_x_min + 1, :] \
                 = image[img_y_min:img_y_max + 1, img_x_min:img_x_max +1, :] 
      data_image[frame_id, :, :, :] = scipy.misc.imresize(crop_image, [image_size, image_size])
      seg_float = output[frame_id]['seg'].astype(np.float32)
      crop_seg = np.zeros((new_image_size, new_image_size, 3), dtype=np.float32)
      crop_seg[max(0, -y_min):max(0, -y_min) + img_y_max - img_y_min + 1, \
               max(0, -x_min):max(0, -x_min) + img_x_max - img_x_min + 1] \
               = np.expand_dims(seg_float[img_y_min:img_y_max + 1, \
                                img_x_min:img_x_max +1], 2)   
      seg = scipy.misc.imresize(crop_seg, [image_size, image_size])
      seg[seg < 0.5] = 0
      seg[seg >= 0.5] = 1 
      
      #print np.max(output['seg'][sample_id, :, :])
      data_seg[frame_id, :, :] = seg[:, :, 0]
      data_chamfer[frame_id, :, :], _, _ = get_chamfer(seg[:,:,0],
                                                        chamfer_scale)
#  return data_pose, data_T, data_R, data_J, data_J_2d,\
#         data_f, data_c, data_gender
  #print "resize",data_resize_scale*3.0
  return data_pose, data_T, data_R, data_beta, data_J, data_J_2d, data_image/255.0,\
         data_seg, data_f/3.0, data_chamfer, data_c/3.0, data_gender, data_resize_scale*3.0  


def _floatList_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))
def _intList_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))



def convert_to_tfrecords_from_folder(folder_name, tf_filename, get_samples=None, test=False, quo=0, with_idx=False):
  files = get_file_list(folder_name, quo, test=test)
  #print "--files list get!--"
  random.shuffle(files)

  num_files = len(files)
  num_frames = 2
  crop_image_size = 128
  keypoints_num = 24
  bases_num = 10
  if not get_samples:
    get_samples = num_files
  #print "total samples", get_samples
  #print "start load Batch!"
  writer = tf.python_io.TFRecordWriter(tf_filename)
  for sample_id in tqdm(range(get_samples)):
    #print "sample_id", files[sample_id]
    pose, T, R, beta, J, J_2d, image, seg, f, chamfer, c, gender, resize_scale = \
      loadBatchSurreal_fromString(files[sample_id], crop_image_size, num_frames)
    scipy.io.savemat('tf_test.mat', \
       {'pose':pose,'T':T,'R':R,'beta':beta,'J':J,'J_2d':J_2d, 'image':image, 'seg':seg, 'f':f, \
        'chamfer':chamfer, 'c':c, 'gender':gender, 'resize_scale':resize_scale})

    example = tf.train.Example(features=tf.train.Features(feature={ 
            'pose': _floatList_feature(pose.flatten()),
            'beta': _floatList_feature(beta.flatten()),
            'T': _floatList_feature(T.flatten()),
            'R': _floatList_feature(R.flatten()),
            'J': _floatList_feature(J.flatten()),
            'J_2d': _floatList_feature(J_2d.flatten()),
            'image': _floatList_feature(image.flatten()),
            'seg': _floatList_feature(seg.flatten()),
            'f': _floatList_feature(f.flatten()),
            'chamfer': _floatList_feature(chamfer.flatten()),
            'c': _floatList_feature(c.flatten()),
            'resize_scale': _floatList_feature(resize_scale.flatten()),
            'gender': _intList_feature([gender]),
            'idx': _intList_feature([sample_id])}))    
    writer.write(example.SerializeToString())  
  writer.close() 


def read_and_decode_surreal(tfrecord_file):
  num_frames = 2
  image_size = 128
  keypoints_num = 24
  bases_num = 10
  chamfer_scale = 0.5
  small_image_size = int(chamfer_scale * image_size)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(tfrecord_file)
  feature = tf.parse_single_example(
    serialized_example,  
    features={
      'pose': tf.FixedLenFeature([num_frames*keypoints_num*3], tf.float32),
      'beta': tf.FixedLenFeature([num_frames*bases_num], tf.float32),
      'T': tf.FixedLenFeature([num_frames*3], tf.float32),
      'R': tf.FixedLenFeature([num_frames*6], tf.float32),
      'J': tf.FixedLenFeature([num_frames*keypoints_num*3], tf.float32),
      'J_2d': tf.FixedLenFeature([num_frames*keypoints_num*2], tf.float32),
      'image': tf.FixedLenFeature([num_frames*image_size*image_size*3], tf.float32),
      'seg': tf.FixedLenFeature([num_frames*image_size*image_size], tf.float32),
      'f': tf.FixedLenFeature([num_frames*2], tf.float32),
      'chamfer': tf.FixedLenFeature([num_frames*small_image_size*small_image_size], tf.float32),
      'c': tf.FixedLenFeature([num_frames*2], tf.float32),
      'resize_scale': tf.FixedLenFeature([num_frames], tf.float32),
      'gender': tf.FixedLenFeature([], tf.int64),
    })
  feature['pose'] = tf.reshape(feature['pose'], [num_frames, keypoints_num, 3])
  feature['beta'] = tf.reshape(feature['beta'], [num_frames, bases_num])
  feature['T'] = tf.reshape(feature['T'], [num_frames, 3])
  feature['R'] = tf.reshape(feature['R'], [num_frames, 6])
  feature['J'] = tf.reshape(feature['J'], [num_frames, keypoints_num, 3])
  feature['J_2d'] = tf.reshape(feature['J_2d'], [num_frames, keypoints_num, 2])
  feature['image'] = tf.reshape(feature['image'], [num_frames, image_size, image_size, 3])
  feature['seg'] = tf.reshape(feature['seg'], [num_frames, image_size, image_size])
  feature['chamfer'] = tf.reshape(feature['chamfer'], [num_frames, small_image_size, small_image_size])
  feature['c'] = tf.reshape(feature['c'], [num_frames, 2])
  feature['f'] = tf.reshape(feature['f'], [num_frames, 2])
  feature['resize_scale'] = tf.reshape(feature['resize_scale'], [num_frames])
    
  return feature['pose'], feature['beta'], feature['T'], feature['R'], feature['J'], feature['J_2d'],\
           feature['image'], feature['seg'], feature['chamfer'], feature['c'], \
           feature['f'], feature['resize_scale'], feature['gender']
 

def read_and_decode_surreal_with_idx(tfrecord_file):
  num_frames = 2
  image_size = 128
  keypoints_num = 24
  bases_num = 10
  chamfer_scale = 0.5
  small_image_size = int(chamfer_scale * image_size)
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(tfrecord_file)
  feature = tf.parse_single_example(
    serialized_example,  
    features={
      'pose': tf.FixedLenFeature([num_frames*keypoints_num*3], tf.float32),
      'beta': tf.FixedLenFeature([num_frames*bases_num], tf.float32),
      'T': tf.FixedLenFeature([num_frames*3], tf.float32),
      'R': tf.FixedLenFeature([num_frames*6], tf.float32),
      'J': tf.FixedLenFeature([num_frames*keypoints_num*3], tf.float32),
      'J_2d': tf.FixedLenFeature([num_frames*keypoints_num*2], tf.float32),
      'image': tf.FixedLenFeature([num_frames*image_size*image_size*3], tf.float32),
      'seg': tf.FixedLenFeature([num_frames*image_size*image_size], tf.float32),
      'f': tf.FixedLenFeature([num_frames*2], tf.float32),
      'chamfer': tf.FixedLenFeature([num_frames*small_image_size*small_image_size], tf.float32),
      'c': tf.FixedLenFeature([num_frames*2], tf.float32),
      'resize_scale': tf.FixedLenFeature([num_frames], tf.float32),
      'gender': tf.FixedLenFeature([], tf.int64),
      'idx': tf.FixedLenFeature([], tf.int64),
    })
  feature['pose'] = tf.reshape(feature['pose'], [num_frames, keypoints_num, 3])
  feature['beta'] = tf.reshape(feature['beta'], [num_frames, bases_num])
  feature['T'] = tf.reshape(feature['T'], [num_frames, 3])
  feature['R'] = tf.reshape(feature['R'], [num_frames, 6])
  feature['J'] = tf.reshape(feature['J'], [num_frames, keypoints_num, 3])
  feature['J_2d'] = tf.reshape(feature['J_2d'], [num_frames, keypoints_num, 2])
  feature['image'] = tf.reshape(feature['image'], [num_frames, image_size, image_size, 3])
  feature['seg'] = tf.reshape(feature['seg'], [num_frames, image_size, image_size])
  feature['chamfer'] = tf.reshape(feature['chamfer'], [num_frames, small_image_size, small_image_size])
  feature['c'] = tf.reshape(feature['c'], [num_frames, 2])
  feature['f'] = tf.reshape(feature['f'], [num_frames, 2])
  feature['resize_scale'] = tf.reshape(feature['resize_scale'], [num_frames])
    
  return feature['pose'], feature['beta'], feature['T'], feature['R'], feature['J'], feature['J_2d'],\
           feature['image'], feature['seg'], feature['chamfer'], feature['c'], \
           feature['f'], feature['resize_scale'], feature['gender'], feature['idx']

def inputs_surreal(tf_filenames, batch_size):
  with tf.name_scope('surreal_input'):
    filename_queue = tf.train.string_input_producer(tf_filenames)
    pose, beta, T, R, J, J_2d, image, seg, chamfer, c, f, resize_scale, gender = read_and_decode_surreal(filename_queue) 
    
    return tf.train.shuffle_batch([pose, beta, T, R, J, J_2d, image, seg, chamfer, c, f, 
             resize_scale, gender], 
             batch_size=batch_size, 
             num_threads=2,capacity=5000,min_after_dequeue=2000)

def inputs_surreal_with_idx(tf_filenames, batch_size, shuffle=True):
  with tf.name_scope('surreal_input'):
    filename_queue = tf.train.string_input_producer(tf_filenames, shuffle=shuffle)
    pose, beta, T, R, J, J_2d, image, seg, chamfer, c, f, resize_scale, gender, idx = read_and_decode_surreal_with_idx(filename_queue) 
   
    if not shuffle:
      return tf.train.batch([pose, beta, T, R, J, J_2d, image, seg, chamfer, c, f, 
             resize_scale, gender, idx], 
             batch_size=batch_size, 
             num_threads=2) #,capacity=80,min_after_dequeue=50)

    else: 
      return tf.train.shuffle_batch([pose, beta, T, R, J, J_2d, image, seg, chamfer, c, f, 
             resize_scale, gender, idx], 
             batch_size=batch_size, 
             num_threads=2,capacity=80,min_after_dequeue=50)

