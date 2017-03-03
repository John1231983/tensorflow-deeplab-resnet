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

DATA_DIRECTORY = '/home/mac/Downloads/PASCAL_VOC_Aug/'
DATA_LIST_PATH = './dataset/5k_data_valid_removal.txt'
SAVE_LIST_PATH = './dataset/5k_data_valid_removal_hard.txt'
NUM_STEPS = 5236
RESTORE_FROM = 'snapshots_curriculum_5k_data/model.ckpt-5000'
MAX_SAVE = 500

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

def main(data_dir=DATA_DIRECTORY, data_list=DATA_LIST_PATH, save_list=SAVE_LIST_PATH, num_steps=NUM_STEPS, max_save=MAX_SAVE, restore_from=RESTORE_FROM):
    """Create the model and start the evaluation process."""

    graph = tf.Graph()

    with graph.as_default():

        tf.set_random_seed(RANDOM_SEED)

        # Create queue coordinator.
        coord = tf.train.Coordinator()
    
        # Load reader.
        with tf.name_scope("create_inputs"):
            reader = ImageReader(
                data_dir,
                data_list,
                None, # No defined input size.
                None, # No defined seed
                False, # No random scale.
                False, # No random mirror.
                coord)
            image, label = reader.image, reader.label

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
        raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    
        # Calculate hard-samples based on entropy of the predictions
        logits = tf.reshape(raw_output_up, [-1, n_classes])
        pred_probs = tf.nn.softmax(logits)
        pred_entropy = tf.reduce_mean(-tf.reduce_sum(pred_probs *
                                                 tf.log(pred_probs), axis=1))

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
            image_names.append(line)
        f.close()

        # Array to store the entropies
        preds_entropy = np.zeros(num_steps)

        # Iterate over training steps.
        for step in range(num_steps):
            preds_entropy[step] = sess.run(pred_entropy)
            if (step % 100 == 0):
                print('Processed {:d}/{:d}'.format(step, num_steps))

        # Sort the entropy list in descending order
        indices = preds_entropy.argsort()[::-1]

        # Save the difficult samples to a file
        if os.path.isfile(save_list):
            print('File %s is already present. Deleting it...'%(save_list))
            os.remove(save_list)

        hard_samples_list = open(save_list, "w")

        for i in range(max_save):
            img = image_names[indices[i]]
            hard_samples_list.write(img)

        hard_samples_list.close()

        print('Hard samples list is saved at: %s'%(save_list))

        coord.request_stop()
        coord.join(threads)
    
if __name__ == '__main__':
    main()

