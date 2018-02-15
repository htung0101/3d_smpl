import numpy as np
import struct
w = 320
h = 240
def write_syn_to_bin(parsed_data, filename):
  # gender: 1 
  # beta: 100 x 10
  # pose: 100 x 72
  # f : 100 x2
  # R : 100 x 3
  # T : 100 x 3
  # J : 100x24x3
  # J_2d : 100 x 24 x2
  # image: 100 x 24 x 320 x 3 # np.unit8
  # seg: 100 x 240 x 320 #bool
  num_frames = parsed_data['pose'].shape[0]
  # gender[int32], num_frames[int32]
  with open(filename, "wb") as f_:
    f_.write(struct.pack('i', parsed_data['gender'])) 
    f_.write(struct.pack('i', num_frames)) 
    for frame_id in range(num_frames):
      beta = list(parsed_data['beta'][frame_id, :])
      pose = list(parsed_data['pose'][frame_id, :])
      f = list(parsed_data['f'][frame_id, :])
      R = list(parsed_data['R'][frame_id, :])
      T = list(parsed_data['T'][frame_id, :])
      J = list(np.reshape(parsed_data['J'][frame_id, :, :], [-1]))
      J_2d = list(np.reshape(parsed_data['J_2d'][frame_id, :, :], [-1]))
      image = list(np.reshape(parsed_data['image'][frame_id, :, :, :].astype(np.float32), [-1]))
      params = beta + pose + f + R + T + J + J_2d + image
      num_elements = len(params)
      f_.write(struct.pack('f' * num_elements, *params))
      seg = list(np.reshape(parsed_data['seg'][frame_id, :, :], [-1])) 
      f_.write(struct.pack('?' * h * w, *seg))
    """
    frame_id = 23
    print "gender", parsed_data['gender']
    print "beta", parsed_data['beta'][frame_id, :4]
    print "pose", parsed_data['pose'][frame_id, :4]
    print "f", parsed_data['f'][frame_id, :]
    print "R", parsed_data['R'][frame_id, :]
    print "T", parsed_data['T'][frame_id, :]
    print "J", parsed_data['J'][frame_id, :2, :]
    print "J_2d", parsed_data['J_2d'][frame_id, :2, :]
    print "image", parsed_data['image'][frame_id, 120:125,180, :]
    print "seg", parsed_data['seg'][frame_id, 120:125,170:175]
    """
       

def read_syn_to_bin(filename, frame_id):

  with open(filename, 'rb') as f_:
    line = f_.read(4)
    gender = struct.unpack('i', line)[0] 
    line = f_.read(4)
    num_frames = struct.unpack('i', line)[0] 
    num_elements_in_line = 10 + 72 + 2 + 3 + 3 + 24 * 3 + 24 * 2 + h * w * 3
    # get to the head of requested frame
    _ = f_.read((4 * (num_elements_in_line) + h * w) * frame_id)
    line = f_.read(4 * num_elements_in_line)
    params = struct.unpack('f' * num_elements_in_line, line) 
    line = f_.read(1 * h * w)
    seg = struct.unpack('?' * h * w, line) 
    output = dict()
    output['gender'] = gender
    output['beta'] = params[:10]
    output['pose'] = params[10: 82]
    output['f'] = params[82:84]
    output['R'] = params[84:87]
    output['T'] = params[87:90]
    output['J'] = np.reshape(params[90:90 + 72], [24, 3])
    output['J_2d'] = np.reshape(params[162:162 + 48], [24, 2])
    output['image'] = np.reshape(params[210:210 + h * w * 3], [h, w, 3])
    output['seg'] = np.reshape(seg, [h, w])
    return output 
    """
    print gender, num_frames
    print "beta", beta[:4]
    print "pose", pose[:4]
    print "f", f
    print "R", R
    print "T", T
    print "J", J[:2, :] 
    print "J_2d", J_2d[:2, :] 
    print "image", image[120:125, 180, :]
    print "seg", seg[120:125, 180]
    """
