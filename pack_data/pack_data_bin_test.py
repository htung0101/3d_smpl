import numpy as np
import pickle
import os 
import random
from write_utils import read_syn_to_bin
import struct 
import sys
from tfrecord_utils import convert_to_tfrecords_from_folder
with_idx=True
dataset = "h36m"
train = True
test = not train
run = 0
data_path = "/home/htung/Documents/2017/Spring/3D_pose/data/scenario"

data_path = os.path.join(data_path, dataset)
if train:
  data_path = os.path.join(data_path, "train")
else:
  data_path = os.path.join(data_path, "test")

data_path = os.path.join(data_path, "run" + str(run))

print data_path
 
quo = sys.argv[1]

print "quo", quo

#a = b
#filename = "/mnt/ssd/fish_tmp/surreal/tfrecords2/surreal2_1.35quo" + str(quo) + ".tfrecords"
#filename = "/mnt/ssd/fish_tmp/surreal/tfrecords2/surreal2_100_test_quo" + str(quo) + ".tfrecords"
filename = "/mnt/ssd/fish_tmp/surreal/tfrecords2/surreal2_100_testseq_quo" + str(quo) + ".tfrecords"
print filename
convert_to_tfrecords_from_folder(data_path, filename, quo = quo, test=True, get_samples=100, with_idx=with_idx, shuffle=False)

print filename
