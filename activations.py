"""Evaluation script for the DeepLab-ResNet network on the validation subset
   of PASCAL VOC dataset.

This script evaluates the model on 1449 validation images.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel, ImageReader, prepare_label

n_classes = 21

DATA_DIRECTORY = '/home/VOCdevkit'
ACTV_SAVE_DIR = './tmp_activations'
DATA_LIST_PATH = './dataset/train.txt'
NUM_STEPS = 10582 
RESTORE_FROM = './deeplab_resnet.ckpt'

RANDOM_SEED = 1234

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main(data_dir=DATA_DIRECTORY, data_list=DATA_LIST_PATH, save_dir=ACTV_SAVE_DIR, num_steps=NUM_STEPS, restore_from=RESTORE_FROM, seed=RANDOM_SEED, input_size=(321,321)):
    """Create the model and start the evaluation process."""

    graph = tf.Graph()

    with graph.as_default():

        tf.set_random_seed(seed)

        # Create queue coordinator.
        coord = tf.train.Coordinator()
    
        # Load reader.
        with tf.name_scope("create_inputs"):
            reader = ImageReader(
                data_dir,
                data_list,
                input_size,
                seed,
                True, # random scale.
                True, # random mirror.
                coord)
            image, label = reader.image, reader.label

        # Define a placeholder for temperature variable
        Temp = tf.placeholder(shape=None, dtype=tf.float32)

        image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0) # Add one batch dimension.
        h_orig, w_orig = tf.to_float(tf.shape(image_batch)[1]), tf.to_float(tf.shape(image_batch)[2])
        image_batch075 = tf.image.resize_images(image_batch, tf.pack([tf.to_int32(tf.mul(h_orig, 0.75)), tf.to_int32(tf.mul(w_orig, 0.75))]))
        image_batch05 = tf.image.resize_images(image_batch, tf.pack([tf.to_int32(tf.mul(h_orig, 0.5)), tf.to_int32(tf.mul(w_orig, 0.5))]))
    
        # Create network.
        with tf.variable_scope('', reuse=False):
            net = DeepLabResNetModel({'data': image_batch}, is_training=False)
        with tf.variable_scope('', reuse=True):
            net075 = DeepLabResNetModel({'data': image_batch075}, is_training=False)
        with tf.variable_scope('', reuse=True):
            net05 = DeepLabResNetModel({'data': image_batch05}, is_training=False)

        # Which variables to load.
        restore_var = tf.global_variables()
    
        # Predictions.
        raw_output100 = net.layers['fc1_voc12']
        raw_output075 = tf.image.resize_images(net075.layers['fc1_voc12'], tf.shape(raw_output100)[1:3,])
        raw_output05 = tf.image.resize_images(net05.layers['fc1_voc12'], tf.shape(raw_output100)[1:3,])
    
        raw_output = tf.reduce_max(tf.stack([raw_output100, raw_output075, raw_output05]), axis=0)
        raw_output = tf.squeeze(raw_output, squeeze_dims=0) # Remove the batch dimension

        # Soft targets at increased temperature
        raw_activations = tf.nn.softmax(raw_output/ Temp)

    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config, graph=graph) as sess:

        # Initialize the model parameters
        tf.global_variables_initializer().run()
    
        # Load weights.
        loader = tf.train.Saver(var_list=restore_var)
        if restore_from is not None:
            load(loader, sess, restore_from)
    
        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        # Get the file names from the list
        f = open(data_list, 'r')
        image_names = []
        for line in f:
            try:
                img, _ = line.strip("\n").split(' ')
            except ValueError:
                img = _ = line.strip("\n")
            image_names.append(img)
        f.close()

        # Iterate over training steps.
        for step in range(num_steps):
            activations, img = sess.run([raw_activations, image], feed_dict={Temp:2})
            base_fname = image_names[step].strip("\n").rsplit('/', 1)[1].replace('jpg', 'npz')
            f_name = save_dir + "/" + base_fname
            np.savez(f_name, actv=activations)

            if (step % 100 == 0):
                print('Processed {:d}/{:d}'.format(step, num_steps))

        # Store an activation of all zeros for non-hard samples
        arr_zero = np.zeros((10, 10, 21)) #TODO: How to determine the dims
        f_name = save_dir + "/" + '0000_000000.npz'
        np.savez(f_name, actv=arr_zero)

        print('Activations are stored at: %s'%(save_dir))

        coord.request_stop()
        coord.join(threads)
    
if __name__ == '__main__':
    main()
