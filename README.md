### Self-supervised Learning of Motion Capture ###

This is code for the paper:
Hsiao-Yu Fish Tung, Hsiao-Wei Tung, Ersin Yumer, Katerina Fragkiadaki, 
[Self-supervised Learning of Motion Capture](https://arxiv.org/abs/1712.01337), NIPS2017 (Spotlight)

Check the [project page](https://sites.google.com/view/selfsupervisedlearningofmotion/) for more results.


### Content ###
* Environment setup and Dataset
* Data preprocessing
* Pretrained model and small tfrecords
* Training
* Citation
* License

### 1. Environment setup and Dataset  ###

* python
We use python2.7.13 from Anaconda and Tensorflow 1.1

* SMPL model
We need rest body template from SMPL model. 

You can download it from [here](http://smpl.is.tue.mpg.de/).

* SURREAL Dataset
If you plan to pretrain or test on surreal dataset. 

Please download surreal from [here](https://github.com/gulvarol/surreal)

* H36M Dataset
If you plan to test on real video with some groundtruth (to evaluate). 

Please download H3.6M Dataset from [here](http://vision.imar.ro/human3.6m/description.php)

### 2. Data preprocessing ###
* Parse Surreal Dataset into binary files

In order to speed up the read write for tfrecords, we parse surreal dataset into binary files. 
Open file 

    data/preparsed/main_parse_surreal 

and change the data path and output path.

* Build up tfrecords

change the data path to the path you built in the previous step in 

    pack_data/pack_data_bin.py

and run it.
You can specify how many examples you want to have in each tfrecords by changing value for num_samples.
If "is_test" is False, we use sequences generated from actor 1, 5, 6, 7, 8 as training samples.
If "is_test" is True, we use only sequence "" from actor 9 as validation.
You can change this split by modifying the "get_file_list" function in tfrecords_utils.py

### 3. Pretrained model and small tfrecords ###

You can downdload a pretrained model using supervision from [here](https://drive.google.com/drive/folders/1MB0ATtSfQ7qbvMq49UPYhP2ubd1BfAK-?usp=sharing)
surreal_quo0.tfrecords is a small training data and surreal2_100_test_quo1.tfrecords

Note: To make this code pack, I calculate 2d flow directly from 3d groundtruth during testing.
But you should replace this with your own predicted flow and keypoints.

### 4. Train model ###
open up pretrained.sh, there is one commend for pretraining using supervision,
and one commend for finetuning with testing data.
Commend out the line that you need


### Citation ###
If you use this code, please cite:

@incollection{NIPS2017_7108,
title = {Self-supervised Learning of Motion Capture},
author = {Tung, Hsiao-Yu and Tung, Hsiao-Wei and Yumer, Ersin and Fragkiadaki, Katerina},
booktitle = {Advances in Neural Information Processing Systems 30},
editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
pages = {5236--5246},
year = {2017},
publisher = {Curran Associates, Inc.},
url = {http://papers.nips.cc/paper/7108-self-supervised-learning-of-motion-capture.pdf}
}




