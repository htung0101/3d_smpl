# This file preparsed surreal data into binary files
# For fast tfrecord read-write
from load_surreal import get_training_params
import os
import numpy as np
import pickle as pkl
from write_utils import write_syn_to_bin, read_syn_to_bin

data_root_dir = '/PATH/TO/YOUR/SURREAL_DATA/SURREAL'
write_root_dir = "PATH/TO/YOUR/OUTPUT_DIRECTORY" 
dataset_name = "h36m"
subdir = 'data/h36m/train/run0'


# actions to parse
actions = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning']
#actions = ['Photo', 'Posing', 'Purchases', 'Sitting', 'SittingDown']
#actions = ['Smoking', 'Waiting', 'WalkDog', 'Walking', 'WalkTogether']   

# actors to parse for each action
subjects = [1] #, 5, 6, 7, 8, 9, 11]
subactions = ['', '1', '2', '3']

def check_dir(path):
  if not os.path.exists(path):
    os.mkdir(path)

full_data_path = os.path.join(data_root_dir, subdir)
full_write_path = os.path.join(write_root_dir, subdir)
write_split_paths = full_write_path.split('/')
current_path = ""
for pp in write_split_paths:
  current_path = os.path.join(current_path, pp)
  check_dir(current_path)

for action in actions:
  for subject in subjects:
    for subact in subactions: 
      filename = dataset_name + "_S" + str(subject) + "_" + action + subact
      if os.path.exists(os.path.join(full_data_path, filename)):
        data_folder = os.path.join(full_data_path, filename)
        output_folder = os.path.join(full_write_path, filename)
        check_dir(output_folder)
        
        subfiles = [fname[:-4] for fname in os.listdir(data_folder) if fname.endswith('.mp4')]
        for sfile in subfiles:
              #if "c0028" in sfile:
              print sfile
              parsed_data = get_training_params(sfile, data_dir = full_data_path) 
              write_syn_to_bin(parsed_data, os.path.join(output_folder, sfile) + ".bin")
              #read_syn_to_bin(os.path.join(output_folder, sfile) + ".bin")
              
          
