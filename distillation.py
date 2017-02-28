"""Training script with multi-scale inputs for the DeepLab-ResNet network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC,
which contains approximately 10000 images for training and 1500 images for validation.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel, ImageReader_Distill, decode_labels, inv_preprocess, prepare_label

n_classes = 21

BATCH_SIZE = 1
DATA_DIRECTORY = '/home/VOCdevkit'
DATA_LIST_PATH = './dataset/train.txt'
GRAD_UPDATE_EVERY = 10
INPUT_SIZE = '321,321'
LEARNING_RATE = 2.5e-4
MOMENTUM = 0.9
NUM_STEPS = 20001
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = './deeplab_resnet.ckpt'
SAVE_NUM_IMAGES = 1
SAVE_PRED_EVERY = 1000
SNAPSHOT_DIR = './snapshots/'
WEIGHT_DECAY = 0.0005

ACTV_DIR = './activations'

def save(saver, sess, logdir, step):
   '''Save weights.
   
   Args:
     saver: TensorFlow Saver object.
     sess: TensorFlow session.
     logdir: path to the snapshots directory.
     step: current training step.
   '''
   model_name = 'model.ckpt'
   checkpoint_path = os.path.join(logdir, model_name)
    
   if not os.path.exists(logdir):
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step)
   print('The checkpoint has been created.')

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow Saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main(data_dir=DATA_DIRECTORY, actv_dir=ACTV_DIR, data_list=DATA_LIST_PATH, num_steps=NUM_STEPS, restore_from=RESTORE_FROM, snapshot_dir=SNAPSHOT_DIR):
    """Create the model and start the training."""

    h, w = map(int, INPUT_SIZE.split(','))
    input_size = (h, w)

    graph = tf.Graph()

    with graph.as_default():
    
        tf.set_random_seed(RANDOM_SEED)
    
        # Create queue coordinator.
        coord = tf.train.Coordinator()
    
        # Load reader.
        with tf.name_scope("create_inputs"):
            reader = ImageReader_Distill(
                data_dir,
                actv_dir,
                data_list,
                input_size,
                True, # Random scale
                True, # Random mirror
                coord)
            image_batch, label_batch, activation_batch = reader.dequeue(BATCH_SIZE)
            image_batch075 = tf.image.resize_images(image_batch, [int(h * 0.75), int(w * 0.75)])
            image_batch05 = tf.image.resize_images(image_batch, [int(h * 0.5), int(w * 0.5)])
    
        # Create network.
        with tf.variable_scope('', reuse=False):
            net = DeepLabResNetModel({'data': image_batch}, is_training=False)
        with tf.variable_scope('', reuse=True):
            net075 = DeepLabResNetModel({'data': image_batch075}, is_training=False)
        with tf.variable_scope('', reuse=True):
            net05 = DeepLabResNetModel({'data': image_batch05}, is_training=False)
        # For a small batch size, it is better to keep 
        # the statistics of the BN layers (running means and variances)
        # frozen, and to not update the values provided by the pre-trained model. 
        # If is_training=True, the statistics will be updated during the training.
        # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
        # if they are presented in var_list of the optimiser definition.

        # Predictions.
        raw_output100 = net.layers['fc1_voc12']
        raw_output075 = net075.layers['fc1_voc12']
        raw_output05 = net05.layers['fc1_voc12']
        raw_output = tf.reduce_max(tf.stack([raw_output100,
                                             tf.image.resize_images(raw_output075, tf.shape(raw_output100)[1:3,]),
                                             tf.image.resize_images(raw_output05, tf.shape(raw_output100)[1:3,])]), axis=0)
        # Which variables to load. Running means and variances are not trainable,
        # thus all_variables() should be restored.
        restore_var = tf.global_variables()
        all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]
        fc_trainable = [v for v in all_trainable if 'fc' in v.name]
        conv_trainable = [v for v in all_trainable if 'fc' not in v.name] # lr * 1.0
        fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name] # lr * 10.0
        fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name] # lr * 20.0
        assert(len(all_trainable) == len(fc_trainable) + len(conv_trainable))
        assert(len(fc_trainable) == len(fc_w_trainable) + len(fc_b_trainable))

        # Add histogram of all variables
        for v in conv_trainable + fc_w_trainable + fc_b_trainable:
            tf.summary.histogram(v.name.replace(":", "_"), v)
    
    
        # Predictions: ignoring all predictions with labels greater or equal than n_classes
        raw_prediction = tf.reshape(raw_output, [-1, n_classes])
        raw_prediction100 = tf.reshape(raw_output100, [-1, n_classes])
        raw_prediction075 = tf.reshape(raw_output075, [-1, n_classes])
        raw_prediction05 = tf.reshape(raw_output05, [-1, n_classes])

        activation_proc = tf.image.resize_images(activation_batch, tf.shape(raw_output)[1:3,])
        activation_proc075 = tf.image.resize_images(activation_batch, tf.shape(raw_output075)[1:3,]) 
        activation_proc05 = tf.image.resize_images(activation_batch, tf.shape(raw_output05)[1:3,])

        raw_activation = tf.reshape(activation_proc, [-1, n_classes])
        raw_activation075 = tf.reshape(activation_proc075, [-1, n_classes])
        raw_activation05 = tf.reshape(activation_proc05, [-1, n_classes])

    
        label_proc = prepare_label(label_batch, tf.pack(raw_output.get_shape()[1:3]), one_hot=False) # [batch_size, h, w]
        label_proc075 = prepare_label(label_batch, tf.pack(raw_output075.get_shape()[1:3]), one_hot=False)
        label_proc05 = prepare_label(label_batch, tf.pack(raw_output05.get_shape()[1:3]), one_hot=False)
    
        raw_gt = tf.reshape(label_proc, [-1,])
        raw_gt075 = tf.reshape(label_proc075, [-1,])
        raw_gt05 = tf.reshape(label_proc05, [-1,])
    
        indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, n_classes - 1)), 1)
        indices075 = tf.squeeze(tf.where(tf.less_equal(raw_gt075, n_classes - 1)), 1)
        indices05 = tf.squeeze(tf.where(tf.less_equal(raw_gt05, n_classes - 1)), 1)
    
        gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
        gt075 = tf.cast(tf.gather(raw_gt075, indices075), tf.int32)
        gt05 = tf.cast(tf.gather(raw_gt05, indices05), tf.int32)
    
        prediction = tf.gather(raw_prediction, indices)
        prediction100 = tf.gather(raw_prediction100, indices)
        prediction075 = tf.gather(raw_prediction075, indices075)
        prediction05 = tf.gather(raw_prediction05, indices05)

        activation = tf.gather(raw_activation, indices)
        activation075 = tf.gather(raw_activation075, indices075)
        activation05 = tf.gather(raw_activation05, indices05)

        # Define a placeholder for temperature variable
        Temp = tf.placeholder(shape=None, dtype=tf.float32)

        # Calculate softmax probs at higher temperature
        pred_probs = tf.nn.softmax(prediction/ Temp)
        pred_probs100 = tf.nn.softmax(prediction100/ Temp)
        pred_probs075 = tf.nn.softmax(prediction075/ Temp)
        pred_probs05 = tf.nn.softmax(prediction05/ Temp)

        # Calculate distillation loss
        loss_distill = tf.reduce_mean(-tf.reduce_sum(activation * tf.log(pred_probs), axis=1))
        loss_distill100 = tf.reduce_mean(-tf.reduce_sum(activation * tf.log(pred_probs100), axis=1))
        loss_distill075 = tf.reduce_mean(-tf.reduce_sum(activation075 * tf.log(pred_probs075), axis=1))
        loss_distill05 = tf.reduce_mean(-tf.reduce_sum(activation05 * tf.log(pred_probs05), axis=1))

        # Pixel-wise softmax loss.
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        loss100 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction100, labels=gt)
        loss075 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction075, labels=gt075)
        loss05 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction05, labels=gt05)
        l2_losses = [WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
        reduced_loss = tf.reduce_mean(loss) + tf.reduce_mean(loss100) + tf.reduce_mean(loss075) + tf.reduce_mean(loss05) +\
                loss_distill + loss_distill100 + loss_distill075 + loss_distill05 + tf.add_n(l2_losses)
   
        # Add loss to summary
        tf.summary.scalar("loss", reduced_loss)

        # Processed predictions: for visualisation.
        raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
        raw_output_up = tf.argmax(raw_output_up, dimension=3)
        pred = tf.expand_dims(raw_output_up, dim=3)
    
        # Image summary.
        images_summary = tf.py_func(inv_preprocess, [image_batch, SAVE_NUM_IMAGES], tf.uint8)
        labels_summary = tf.py_func(decode_labels, [label_batch, SAVE_NUM_IMAGES], tf.uint8)
        preds_summary = tf.py_func(decode_labels, [pred, SAVE_NUM_IMAGES], tf.uint8)
    
        total_summary = tf.summary.image('images', 
                                        tf.concat(2, [images_summary, labels_summary, preds_summary]), 
                                        max_outputs=SAVE_NUM_IMAGES) # Concatenate row-wise.

        merged_summary = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(snapshot_dir,
                                               graph=tf.get_default_graph())
   
        # Define loss and optimisation parameters.
        base_lr = tf.constant(LEARNING_RATE)
        step_ph = tf.placeholder(dtype=tf.float32, shape=())
        learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / num_steps), POWER))
    
        opt_conv = tf.train.MomentumOptimizer(learning_rate, MOMENTUM)
        opt_fc_w = tf.train.MomentumOptimizer(learning_rate * 10.0, MOMENTUM)
        opt_fc_b = tf.train.MomentumOptimizer(learning_rate * 20.0, MOMENTUM)

        # Define a variable to accumulate gradients.
        accum_grads = [tf.Variable(tf.zeros_like(v.initialized_value()),
                                   trainable=False) for v in conv_trainable + fc_w_trainable + fc_b_trainable]

        # Define an operation to clear the accumulated gradients for next batch.
        zero_op = [v.assign(tf.zeros_like(v)) for v in accum_grads]

        # Compute gradients.
        grads = tf.gradients(reduced_loss, conv_trainable + fc_w_trainable + fc_b_trainable)
   
        # Accumulate and normalise the gradients.
        accum_grads_op = [accum_grads[i].assign_add(grad / GRAD_UPDATE_EVERY) for i, grad in
                           enumerate(grads)]

        grads_conv = accum_grads[:len(conv_trainable)]
        grads_fc_w = accum_grads[len(conv_trainable) : (len(conv_trainable) + len(fc_w_trainable))]
        grads_fc_b = accum_grads[(len(conv_trainable) + len(fc_w_trainable)):]

        # Apply the gradients.
        train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable))
        train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
        train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))

        train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)
    
    
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config, graph=graph) as sess:

        # Initialize the model parameters
        tf.global_variables_initializer().run()

    
        # Saver for storing checkpoints of the model.
        saver = tf.train.Saver(var_list=restore_var, max_to_keep=10)
    
        # Load variables if the checkpoint is provided.
        if restore_from is not None:
            loader = tf.train.Saver(var_list=restore_var)
            load(loader, sess, restore_from)
    
        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        # Iterate over training steps.
        for step in range(num_steps):
            start_time = time.time()
            feed_dict = {step_ph:step, Temp:2}
            loss_value = 0

            # Clear the accumulated gradients.
            sess.run(zero_op, feed_dict=feed_dict)
       
            # Accumulate gradients.
            for i in range(GRAD_UPDATE_EVERY):
                _, l_val = sess.run([accum_grads_op, reduced_loss], feed_dict=feed_dict)
                loss_value += l_val

            # Normalise the loss.
            loss_value /= GRAD_UPDATE_EVERY

            # Apply gradients.
            if step % SAVE_PRED_EVERY == 0:
                images, labels, summary, _ = sess.run([image_batch, label_batch,
                                                       merged_summary, train_op], feed_dict=feed_dict)
                summary_writer.add_summary(summary, step)
                save(saver, sess, snapshot_dir, step)
            else:
                _, summary = sess.run([train_op, merged_summary], feed_dict=feed_dict)
                summary_writer.add_summary(summary, step)

            duration = time.time() - start_time
            print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
            
        coord.request_stop()
        coord.join(threads)
    
if __name__ == '__main__':
    main()
