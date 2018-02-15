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
run = 0
data_path = "/PATH/TO/YOUR/SURREAL_bin/data"
num_samples = 10000
is_test = False
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
filename = "/path/to/your/tfrecords/folder/surreal_" + str(quo) + ".tfrecords"

convert_to_tfrecords_from_folder(data_path, filename, quo = quo, test=is_test, get_samples=num_samples, with_idx=with_idx)
print filename
