import math
import numpy as np 
import tensorflow as tf

from tensorflow.python.framework import ops

#from utils import *

try:
  image_summary = tf.image_summary
  scalar_summary = tf.scalar_summary
  histogram_summary = tf.histogram_summary
  merge_summary = tf.merge_summary
  SummaryWriter = tf.train.SummaryWriter
except:
  image_summary = tf.summary.image
  scalar_summary = tf.summary.scalar
  histogram_summary = tf.summary.histogram
  merge_summary = tf.summary.merge
  SummaryWriter = tf.summary.FileWriter

if "concat_v2" in dir(tf):
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat_v2(tensors, axis, *args, **kwargs)
else:
  def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)
def eud_loss(a, b):
  return tf.reduce_mean(tf.square(a-b))

def l1_loss(a, b):
  return tf.reduce_mean(tf.abs(a-b))

def smoothLoss2(flow):
    with tf.name_scope("smoothLoss2"):
        shape = flow.get_shape()
        bs = shape[0]
        h = shape[1]
        w = shape[2]
        kernel = tf.transpose(tf.constant([[[[0,0,0],[0,1,-1],[0,0,0]]],
                                           [[[0,0,0],[0,1,0],[0,-1,0]]]],
                                          dtype=tf.float32),perm=[3,2,1,0],
                              name="kernel")
        [u,v] = tf.unstack(flow,axis=3)
        u = tf.expand_dims(u,3,name="u")
        v = tf.expand_dims(v,3,name="v")
        diff_u = tf.nn.conv2d(u,kernel,[1,1,1,1],padding="SAME",name="diff_u")
        diff_v = tf.nn.conv2d(v,kernel,[1,1,1,1],padding="SAME",name="diff_v")
        diffs = tf.concat([diff_u,diff_v],3,name="diffs")

        # make mask with ones everywhere but the bottom and right borders
        mask = tf.ones([bs,h-1,w-1,1],name="mask")
        mask = tf.concat([mask,tf.zeros([bs,1,w-1,1])], 1,name="mask2")
        mask = tf.concat([mask,tf.zeros([bs,h,1,1])], 2,name="mask3")
        loss = tf.reduce_mean(tf.abs(diffs*mask),name="loss")
        return loss




def per_joint_loss(a, b):
  return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(a - b), 2)))

def merge_bf(value):
  shape = value.get_shape().as_list()
  return tf.reshape(value, [-1] + shape[2:])

def split_bf(value, batch_size, nframes):
  shape = value.get_shape().as_list()
  return tf.reshape(value, [batch_size, nframes] + shape[1:])

def norm(value):
  shape = value.get_shape()
  return tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(value), len(shape)-1)), len(shape)-1)

def get_scalar_summary(name, value):
  summ = dict()
  summ["syn_train"] = tf.summary.scalar(name + " (syn train)", value)
  summ["syn_test"] = tf.summary.scalar(name + " (syn test)", value)
  summ["h3.6_train"] = tf.summary.scalar(name + " (h3.6 train)", value)
  summ["h3.6_test"] = tf.summary.scalar(name + " (h3.6 test)", value)
  return summ

def get_hist_summary(name, value):
  summ = dict()
  summ["syn_train"] = tf.summary.histogram(name + " (syn train)", value)
  summ["syn_test"] = tf.summary.histogram(name + " (syn test)", value)
  summ["h3.6_train"] = tf.summary.histogram(name + " (h3.6 train)", value)
  summ["h3.6_test"] = tf.summary.histogram(name + " (h3.6 test)", value)
  return summ

def get_image_summary(name, value, n=4):
  summ = dict()
  summ["syn_train"] = tf.summary.image(name + " (syn train)", value, n)
  summ["syn_test"] = tf.summary.image(name + " (syn test)", value, n)
  summ["h3.6_train"] = tf.summary.image(name + " (h3.6 train)", value, n)
  summ["h3.6_test"] = tf.summary.image(name + " (h3.6 test)", value, n)
  return summ 
 
def getIdxMap(batch_size, height, width):
  IdxMap = np.zeros((batch_size, height, width, 2), dtype=np.float32)  
  for h in range(height):
    IdxMap[:, h, :, 1] = h  
  for w in range(width):
    IdxMap[:, :, w, 0] = w  
  return IdxMap 

def repeat(x, n_repeats):
  rep = tf.transpose(
      tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
  rep = tf.cast(rep, 'int32')
  x = tf.matmul(tf.reshape(x, (-1, 1)), rep) 
  return tf.reshape(x, [-1])

def split(tensor, dim):
  shape = tensor.get_shape().as_list()
  #final_shape = [item for item in shape[:dim]] + [item for item in shape[dim+1:]] 
  return [tf.squeeze(item, dim) for item in tf.split(tensor, shape[dim], axis=dim)]

def print_shape(t):
  print(t.name, t.get_shape().as_list())

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)

def conv_cond_concat(x, y):
  """Concatenate conditioning vector on feature map axis."""
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return concat(3, [
    x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])])

def conv2d(input_, output_dim, 
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="conv2d"):
  with tf.variable_scope(name):
    w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv

def deconv2d(input_, output_shape,
       k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
       name="deconv2d", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))
    
    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv
     
def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
  shape = input_.get_shape().as_list()

  with tf.variable_scope(scope or "Linear"):
    matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                 tf.random_normal_initializer(stddev=stddev))
    bias = tf.get_variable("bias", [output_size],
      initializer=tf.constant_initializer(bias_start))
    if with_w:
      return tf.matmul(input_, matrix) + bias, matrix, bias
    else:
      return tf.matmul(input_, matrix) + bias
