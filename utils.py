import tensorflow as tf
import math
import numpy as np 
from tensorflow.python.framework import ops
from flow_transformer import transformer
#import hyperparams as hyp

def print_shape(t):
    print(t.name, t.get_shape().as_list())

def split_rt(rt):
    r = tf.slice(rt,[0,0,0],[-1,3,3])
    t = tf.reshape(tf.slice(rt,[0,0,3],[-1,3,1]),[hyp.bs,3])
    return r, t

def merge_rt(r,t):
    bottom_row = tf.tile(tf.reshape(tf.pack([0.,0.,0.,1.]),[1,1,4]),
                         [hyp.bs,1,1],name="bottom_row")
    rt = tf.concat(2,[r,tf.expand_dims(t,2)],name="rt_3x4")
    rt = tf.concat(1,[rt,bottom_row],name="rt_4x4")
    return rt

def random_crop(t,crop_h,crop_w,h,w):
    def off_h(): return tf.random_uniform([], minval=0, maxval=h-crop_h, dtype=tf.int32)
    def off_w(): return tf.random_uniform([], minval=0, maxval=w-crop_w, dtype=tf.int32)
    def z(): return tf.constant(0)
    offset_h = tf.cond(tf.less(crop_h, h), off_h, z)
    offset_w = tf.cond(tf.less(crop_w, w), off_w, z)
    t_crop = tf.slice(t,[offset_h,offset_w,0],[crop_h,crop_w,-1],name="cropped_tensor")
    return t_crop, offset_h, offset_w

def compute_distance(transform):
    """
    Compute the distance of the translational component of a 4x4 homogeneous matrix.
    """
    # return numpy.linalg.norm(transform[0:3,3])
    # t = tf.reshape(tf.slice(transform,[0,0,3],[-1,3,1]),[-1,3])
    t = tf.reshape(tf.slice(transform,[0,0,3],[-1,3,1]),[-1,3])
    # t should now be bs x 3  
    return tf.sqrt(tf.reduce_sum(tf.square(t),axis=1))

def compute_angle(transform):
    """
    Compute the rotation angle from a 4x4 homogeneous matrix.
    """
    # an invitation to 3-d vision, p 27
    # return numpy.arccos( min(1,max(-1, (numpy.trace(transform[0:3,0:3]) - 1)/2) ))
    r = tf.slice(transform,[0,0,0],[-1,3,3])
    return tf.acos(tf.minimum(1.,tf.maximum(-1.,(tf.trace(r)-1.)/2.)))

def compute_t_diff(rt1, rt2):
    """
    Compute the difference between the magnitudes of the translational components of the two transformations. 
    """
    t1 = tf.reshape(tf.slice(rt1,[0,0,3],[-1,3,1]),[-1,3])
    t2 = tf.reshape(tf.slice(rt2,[0,0,3],[-1,3,1]),[-1,3])
    # each t should now be bs x 3  
    mag_t1 = tf.sqrt(tf.reduce_sum(tf.square(t1),axis=1))
    mag_t2 = tf.sqrt(tf.reduce_sum(tf.square(t2),axis=1))
    return tf.abs(mag_t1-mag_t2)

def compute_t_ang(rt1, rt2):
    """
    Compute the angle between the translational components of two transformations.
    """
    t1 = tf.reshape(tf.slice(rt1,[0,0,3],[-1,3,1]),[-1,3])
    t2 = tf.reshape(tf.slice(rt2,[0,0,3],[-1,3,1]),[-1,3])
    # each t should now be bs x 3  
    mag_t1 = tf.sqrt(tf.reduce_sum(tf.square(t1),axis=1))
    mag_t2 = tf.sqrt(tf.reduce_sum(tf.square(t2),axis=1))
    dot = tf.reduce_sum(t1*t2,axis=1)
    return tf.acos(dot/(mag_t1*mag_t2 + hyp.eps))

def safe_inverse(a):
    """ 
    safe inverse for rigid transformations
    should be equivalent to 
      a_inv = tf.matrix_inverse(a)
    for well-behaved matrices
    """
    shape = a.get_shape()
    bs = int(shape[0])
    Ra = tf.slice(a,[0,0,0],[-1,3,3])
    Ta = tf.reshape(tf.slice(a,[0,0,3],[-1,3,1]),[bs,3])
    Ra_t = tf.transpose(Ra,[0,2,1])
    bottom_row = tf.tile(tf.reshape(tf.pack([0.,0.,0.,1.]),[1,1,4]),[bs,1,1])
    a_inv = tf.concat(2,[Ra_t,-tf.batch_matmul(Ra_t, tf.expand_dims(Ta,2))])
    a_inv = tf.concat(1,[a_inv,bottom_row])
    return a_inv

def ominus(a,b):
    """
    Compute the relative 3D transformation between a and b.
    
    Input:
    a -- first pose (homogeneous 4x4 matrix)
    b -- second pose (homogeneous 4x4 matrix)
    
    Output:
    Relative 3D transformation from a to b.
    https://github.com/liruihao/tools-for-rgbd-SLAM-evaluation/blob/master/evaluate_rpe.py
    """
    with tf.name_scope("ominus"):
        a_inv = safe_inverse(a)
        return tf.batch_matmul(a_inv,b)

def sinabg2r(sina,sinb,sing):
    shape = sina.get_shape()
    bs = int(shape[0])
    one = tf.ones([bs],name="one")
    zero = tf.zeros([bs],name="zero")
    cosa = tf.sqrt(1 - tf.square(sina))
    cosb = tf.sqrt(1 - tf.square(sinb))
    cosg = tf.sqrt(1 - tf.square(sing))
    Rz = tf.reshape(tf.pack([cosa, -sina, zero,
                             sina, cosa, zero,
                             zero, zero, one],
                    axis=1),[bs, 3, 3])
    Ry = tf.reshape(tf.pack([cosb, zero, sinb,
                             zero, one, zero,
                             -sinb, zero, cosb],
                    axis=1),[bs, 3, 3])
    Rx = tf.reshape(tf.pack([one, zero, zero,
                             zero, cosg, -sing,
                             zero, sing, cosg],
                    axis=1),[bs, 3, 3])
    Rcam=tf.batch_matmul(tf.batch_matmul(Rx,Ry),Rz,name="Rcam")
    return Rcam
    
def sinabg2r_fc(sina,sinb,sing):
    shape = sina.get_shape()
    bs = int(shape[0])
    hw = int(shape[1])
    one = tf.ones([bs,hw],name="one")
    zero = tf.zeros([bs,hw],name="zero")
    cosa = tf.sqrt(1 - tf.square(sina))
    cosb = tf.sqrt(1 - tf.square(sinb))
    cosg = tf.sqrt(1 - tf.square(sing))
    Rz = tf.reshape(tf.pack([cosa, -sina, zero,
                             sina, cosa, zero,
                             zero, zero, one],
                            axis=2),[bs, hw, 3, 3])
    Ry = tf.reshape(tf.pack([cosb, zero, sinb,
                             zero, one, zero,
                             -sinb, zero, cosb],
                            axis=2),[bs, hw, 3, 3])
    Rx = tf.reshape(tf.pack([one, zero, zero,
                             zero, cosg, -sing,
                             zero, sing, cosg],
                            axis=2),[bs, hw, 3, 3])

    Rcam=tf.batch_matmul(tf.batch_matmul(Rx,Ry),Rz,name="Rcam")
    
    Rcam = tf.reshape(tf.pack([one, zero, zero,
                               zero, one, zero,
                               zero, zero, one],
                              axis=2),[bs, hw, 3, 3])
    return Rcam
    
def abg2r(a,b,g,bs):
    one = tf.ones([bs],name="one")
    zero = tf.zeros([bs],name="zero")
    sina = tf.sin(a)
    sinb = tf.sin(b)
    sing = tf.sin(g)
    cosa = tf.cos(a)
    cosb = tf.cos(b)
    cosg = tf.cos(g)
    Rz = tf.reshape(tf.pack([cosa, -sina, zero,
                             sina, cosa, zero,
                             zero, zero, one],
                    axis=1),[bs, 3, 3])
    Ry = tf.reshape(tf.pack([cosb, zero, sinb,
                             zero, one, zero,
                             -sinb, zero, cosb],
                    axis=1),[bs, 3, 3])
    Rx = tf.reshape(tf.pack([one, zero, zero,
                             zero, cosg, -sing,
                             zero, sing, cosg],
                    axis=1),[bs, 3, 3])
    Rcam=tf.batch_matmul(tf.batch_matmul(Rx,Ry),Rz,name="Rcam")
    return Rcam

def r2abg(r):
    # r is 3x3. i want to get out alpha, beta, and gamma
    # a = atan2(R(3,2), R(3,3));
    # b = atan2(-R(3,1), sqrt(R(3,2)*R(3,2) + R(3,3)*R(3,3)));
    # g = atan2(R(2,1), R(1,1));

    # x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
    # y = atan2(-R.at<double>(2,0), sy);
    # z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    
    r11 = r[:,0,0]
    r21 = r[:,1,0]
    r31 = r[:,2,0]
    r32 = r[:,2,1]
    r33 = r[:,2,2]
    a = atan2(r32,r33)
    b = atan2(-r31,tf.sqrt(r32*r32+r33*r33))
    g = atan2(r21,r11)
    return a, b, g

def zrt2flow_helper(Z1, rt12, fy, fx, y0, x0):
    r12, t12 = split_rt(rt12)
    if hyp.dataset_name == 'KITTI':
        flow = zrt2flow_kitti(Z1, r12, t12, fy, fx, y0, x0)
    else:
        flow = zrt2flow(Z1, r12, t12, fy, fx, y0, x0)
    return flow

def zrt2flow_kitti(Z, R, T, oh, ow, fy, fx, y0, x0):
    if hyp.do_debug:
        Z = tf.check_numerics(Z, 'util 195')
        R = tf.check_numerics(R, 'util 196')
        T = tf.check_numerics(T, 'util 197')

        fx = tf.check_numerics(fx, 'util 200')
        fy = tf.check_numerics(fy, 'util 201')
        x0 = tf.check_numerics(x0, 'util 202')
        y0 = tf.check_numerics(y0, 'util 203')

    print('*'*100)
    print(Z)
    print(R)
    print(T)
    print(oh)
    print(ow)
    print(fx)
    print(fy)
    print(x0)
    print(y0)
    print('*'*100)

    ed = lambda x : tf.expand_dims(x, axis = 0)
    upk = lambda x : tf.unstack(x, axis = 0)
    upked = lambda x : map(ed,upk(x))
    Zu = upked(Z)
    Ru = upked(R)
    Tu = upked(T)
    ohu = upked(oh)
    owu = upked(ow)
    fxu = upk(fx)
    fyu = upk(fy)
    x0u = upked(x0)
    y0u = upked(y0)

    result1 = []
    result2 = []

    for i in range(hyp.bs):
        Zs = Zu[i]
        Rs = Ru[i]
        Ts = Tu[i]
        ohs = ohu[i]
        ows = owu[i]
        fxs = fxu[i]
        fys = fyu[i]
        x0s = x0u[i]
        y0s = y0u[i]

        r1, r2 = zrt2flow(Zs, Rs, Ts, ohs, ows, fys, fxs, y0s, x0s)
        result1.append(r1)
        result2.append(r2)

    flow = tf.concat(0, result1)
    XYZ2 = tf.concat(0, result2)

    if hyp.do_debug:
        flow = tf.check_numerics(flow, 'util 240')
        XYZ2 = tf.check_numerics(XYZ2, 'util 241')

    print(flow)
    # print XYZ2
    return flow

def zrt2flow(Z, r, t, fy, fx, y0, x0):
    with tf.variable_scope("zrt2flow"):
        shape = Z.get_shape()
        bs = int(shape[0])
        h = int(shape[1])
        w = int(shape[2])

        # get pointcloud1
        [grid_x1,grid_y1] = meshgrid2D(bs, h, w)
        Z = tf.reshape(Z,[bs,h,w],name="Z")
        XYZ = Camera2World(grid_x1,grid_y1,Z,fx,fy,x0,y0)

        # transform pointcloud1 using r and t, to estimate pointcloud2
        t_tiled = tf.tile(tf.expand_dims(t,dim=1),[1,h*w,1],name="t_tiled")
        XYZ_t = tf.transpose(XYZ,perm=[0,2,1],name="XYZ_t")
        XYZ_mm = tf.batch_matmul(r,XYZ_t,name="XYZ_mm")
        XYZ_rot = tf.transpose(XYZ_mm,perm=[0,2,1],name="XYZ_rot")
        XYZ2 = tf.add(XYZ_rot,t_tiled,name="XYZ2")

        # project pointcloud2 down, so that we get the 2D location of all of these pixels
        [X2,Y2,Z2] = tf.split(2, 3, XYZ2, name="splitXYZ")
        x2y2_flat = World2Camera(X2,Y2,Z2,fx,fy,x0,y0)
        [x2_flat,y2_flat]=tf.split(2,2,x2y2_flat,name="splitxyz_flat")

        # subtract the new 2D locations from the old ones to get optical flow
        x1_flat = tf.reshape(grid_x1,[bs,-1,1],name="x1")
        y1_flat = tf.reshape(grid_y1,[bs,-1,1],name="y1")
        flow_flat = tf.concat(2,[x2_flat-x1_flat,y2_flat-y1_flat],name="flow_flat")
        flow = tf.reshape(flow_flat,[bs,h,w,2],name="flow")
        return flow

def warper(frame, flow, name="warper", is_train=True, reuse=False):
    with tf.variable_scope(name):
        shape = flow.get_shape()
        bs, h, w, c = shape
        if reuse:
            tf.get_variable_scope().reuse_variables()
        warp, occ = transformer(frame, flow, (int(h), int(w)))
        return warp, occ
