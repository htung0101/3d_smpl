import os
from model2 import _3DINN
import tensorflow as tf

"""
Define flags
"""
flags = tf.app.flags
flags.DEFINE_integer("gpu", 1, "gpu_id")
flags.DEFINE_string("name", "1", "name of this version")
flags.DEFINE_float("learning_rate", 0.0001, "Learning rate")
flags.DEFINE_float("init", 0.01, "std of param init") #? 
flags.DEFINE_integer("max_iter", 100000, "Iterations times")
flags.DEFINE_integer("batch_size", 16, "The size of batch images")
flags.DEFINE_integer("num_frames", 2, "The size of batch images")
flags.DEFINE_integer("gf_dim", 32, "The size of batch images")
flags.DEFINE_integer("flow_dim", 32, "The size of batch images")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_float("alpha", 0.5, "Weights for silhouette loss S_M * C_I[0.5]")
flags.DEFINE_float("sr", 1.0, "sampling rate for visibility")
flags.DEFINE_float("chamfer_scale", 0.5, "size of chamfer image. Reduce to save memory" 
                   "and computation")
flags.DEFINE_integer("image_size_h", 128, "")
flags.DEFINE_integer("image_size_w", 128, "")
flags.DEFINE_integer("keypoints_num", 24, "number of keypoints") 
flags.DEFINE_integer("mesh_num", 6890, "number of keypoints") 
flags.DEFINE_integer("bases_num", 10, "number of Base Shapes") # 1 (mean)+ 96

flags.DEFINE_integer("gWidth", 7, "width of Gaussian kernels")
flags.DEFINE_float("gStddev", 0.25, "std for 2d Gaussian heatmaps")
flags.DEFINE_string("data_dir", "../src/output", "Directory name to save the preprocessed data [data]")
flags.DEFINE_string("checkpoint_dir", "checkpoint/", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("model_dir", None, "Directory name to save the checkpoints [checkpoint]")
#flags.DEFINE_string("flow_model_dir", None, "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples/", "Directory name to save the image samples [samples]")
flags.DEFINE_string("logs_dir", "logs/", "Directory name to save logs [logs]")
flags.DEFINE_boolean("is_train", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_debug_basic_function", False, "True for debugging basic function, should have d2, d3=0")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
flags.DEFINE_boolean("is_sup_train", True, "True for supervised training on training data,"
                                           "False for unsupervised training [True]")
flags.DEFINE_boolean("is_dryrun", False, "Run one batch to see whether visibility is correct"
                                         "You need to reduce batch size to fit memory [False]")
#flags.DEFINE_boolean("sup_loss", True, "True for using supervised loss")
flags.DEFINE_boolean("key_loss", False, "True for using unsupervised keypoint loss")
flags.DEFINE_boolean("silh_loss", False, "True for using unsupervised segmentation loss")
flags.DEFINE_boolean("pixel_loss", False, "True for using unsupervised flow matching loss")
flags.DEFINE_boolean("pretrained_flownet", False, "True for using supervised loss")
FLAGS = flags.FLAGS

def main(_):
    checkpoint_dir_ = os.path.join(FLAGS.checkpoint_dir, FLAGS.name)
    sample_dir_ = os.path.join(FLAGS.sample_dir, FLAGS.name)
    logs_dir_ = os.path.join(FLAGS.logs_dir, FLAGS.name)
    if not os.path.exists(checkpoint_dir_):
        os.makedirs(checkpoint_dir_)
    if not os.path.exists(sample_dir_):
        os.makedirs(sample_dir_)
    if not os.path.exists(logs_dir_):
        os.makedirs(logs_dir_)
    with tf.Session() as sess:
        my3DINN = _3DINN(sess,
		         config=FLAGS,
                         checkpoint_dir=checkpoint_dir_,
                         logs_dir=logs_dir_,
                         sample_dir=sample_dir_)
        if FLAGS.is_train:
            my3DINN.train(FLAGS)
        else:
            my3DINN.load(checkpoint_dir)
if __name__ == '__main__':
	tf.app.run()
