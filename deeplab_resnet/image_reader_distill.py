import os

import numpy as np
import tensorflow as tf
from utils import load_npz, get_actv_shape

IGNORE_LABEL = 255
IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

n_classes = 21
def image_scaling(img, label, seed):
    """
    Randomly scales the images between 0.5 to 1.5 times the original size.

    Args:
      img: Training image to scale.
      label: Segmentation mask to scale.
    """
    
    scale = tf.random_uniform([1], minval=0.5, maxval=1.5, dtype=tf.float32, seed=seed)
    h_new = tf.to_int32(tf.mul(tf.to_float(tf.shape(img)[0]), scale))
    w_new = tf.to_int32(tf.mul(tf.to_float(tf.shape(img)[1]), scale))
    new_shape = tf.squeeze(tf.pack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize_images(img, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])
   
    return img, label

def image_mirroring(img, label, seed):
    """
    Randomly mirrors the images.

    Args:
      img: Training image to mirror.
      label: Segmentation mask to mirror.
    """
    
    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32, seed=seed)[0]
    mirror = tf.less(tf.pack([1.0, distort_left_right_random, 1.0]), 0.5)
    img = tf.reverse(img, mirror)
    label = tf.reverse(label, mirror)
    return img, label

def random_crop_and_pad_image_and_labels(image, label, crop_h, crop_w, seed, ignore_label=255):
    """
    Randomly crop and pads the input images.

    Args:
      image: Training image to crop/ pad.
      label: Segmentation mask to crop/ pad.
      crop_h: Height of cropped segment.
      crop_w: Width of cropped segment.
      ignore_label: Label to ignore during the training.
    """

    label = tf.cast(label, dtype=tf.float32)
    label = label - ignore_label # Needs to be subtracted and later added due to 0 padding.
    combined = tf.concat(2, [image, label]) 
    image_shape = tf.shape(image)
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, tf.maximum(crop_h, image_shape[0]), tf.maximum(crop_w, image_shape[1]))
    
    last_image_dim = tf.shape(image)[-1]
    combined_crop = tf.random_crop(combined_pad, [crop_h,crop_w,4], seed=seed)
    img_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop = label_crop + ignore_label
    label_crop = tf.cast(label_crop, dtype=tf.uint8)
    
    # Set static shape so that tensorflow knows shape at compile time. 
    img_crop.set_shape((crop_h, crop_w, 3))
    label_crop.set_shape((crop_h, crop_w, 1))
    return img_crop, label_crop

def read_labeled_image_list(data_dir, actv_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.
    
    Args:
      data_dir: path to the directory with images and masks.
      actv_dir: path to the directory with activations.
      data_list: path to the file with lines of the form '/path/to/image /path/to/mask /path/to/activation'.
       
    Returns:
      Three lists with all file names for images, masks and activations, respectively.
    """
    f = open(data_list, 'r')
    images = []
    masks = []
    activations = []
    for line in f:
        try:
            image, mask, activation = line.strip("\n").split(' ')
        except ValueError: # Adhoc for test.
            image = mask = activation = line.strip("\n")
        images.append(data_dir + image)
        masks.append(data_dir + mask)
        activations.append(actv_dir + activation)
    return images, masks, activations

def read_images_from_disk(input_queue, input_size, random_scale, random_mirror, seed): # optional pre-processing arguments
    """Read one image and its corresponding mask with optional pre-processing.
    
    Args:
      input_queue: tf queue with paths to the image and its mask.
      input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
      random_scale: whether to randomly scale the images prior
                    to random crop.
      random_mirror: whether to randomly mirror the images prior
                    to random crop.
      
    Returns:
      Three tensors: the decoded image, its mask and activation.
    """

    img_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])
    
    img = tf.image.decode_jpeg(img_contents, channels=3)
    img_r, img_g, img_b = tf.split(split_dim=2, num_split=3, value=img)
    img = tf.cast(tf.concat(2, [img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN

    label = tf.image.decode_png(label_contents, channels=1)

    activation = tf.py_func(load_npz, [input_queue[2]], [tf.double])
    shape = tf.py_func(get_actv_shape, activation, [tf.int64])
    shape = tf.to_int32(tf.reshape(shape, [3]))
    activation = tf.to_float(tf.reshape(activation, shape))
    #shp = activation.get_shape()
    #h_new = tf.to_int32(tf.mul(tf.to_float(tf.shape(activation)[0]), 1))
    #w_new = tf.to_int32(tf.mul(tf.to_float(tf.shape(activation)[1]), 1))
    #new_shape = tf.squeeze(tf.pack([h_new, w_new]), squeeze_dims=[1])
    #new_shape = tf.squeeze(tf.pack([tf.shape(activation)[0], tf.shape(activation)[1]]), squeeze_dims=[1])
    #activation = tf.image.resize_images(activation, [shp[0], shp[1]])
    #activation.set_shape((shp[0], shp[1], 21))

    if input_size is not None:
        h, w = input_size

        # Randomly scale the images and labels.
        if random_scale:
            img, label = image_scaling(img, label, seed)

        # Randomly mirror the images and labels.
        if random_mirror:
            img, label = image_mirroring(img, label, seed)

        # Randomly crops the images and labels.
        img, label = random_crop_and_pad_image_and_labels(img, label, h, w, seed, IGNORE_LABEL)

        # Pad the activation with extra zeros to give it a fixed size.
        # This is a bug in tensorflow pipeline, as shuffle APIs for queues
        # do not accept dynamic size inputs.
        activation = tf.image.resize_image_with_crop_or_pad(activation, h, w)
        activation.set_shape((h, w, n_classes))

    return img, label, activation

class ImageReader_Distill(object):
    '''Generic ImageReader_Distill which reads images and corresponding segmentation
       masks from the disk, and enqueues them into a TensorFlow queue.
    '''

    def __init__(self, data_dir, actv_dir, data_list, input_size, seed, random_scale,
                 random_mirror):
        '''Initialise an ImageReader_Distill.
        
        Args:
          data_dir: path to the directory with images and masks.
          actv_dir: path to the directory with activations for hard-samples.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask /path/to/activations'.
          input_size: a tuple with (height, width) values, to which all the images will be resized.
          random_scale: whether to randomly scale the images prior to random crop.
          random_mirror: whether to randomly mirror the images prior to random crop.
          coord: TensorFlow queue coordinator.
        '''
        self.data_dir = data_dir
        self.actv_dir = actv_dir
        self.data_list = data_list
        self.input_size = input_size
        self.seed = seed
        #self.coord = coord
        
        self.image_list, self.label_list, self.actv_list = read_labeled_image_list(self.data_dir, self.actv_dir, self.data_list)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        self.activations = tf.convert_to_tensor(self.actv_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images, self.labels, self.activations],\
                                                   shuffle=input_size is not None) # not shuffling if it is val
        self.image, self.label, self.activation = read_images_from_disk(self.queue, self.input_size, random_scale, random_mirror, self.seed) 

    def dequeue(self, num_elements):
        '''Pack images and labels into a batch.
        
        Args:
          num_elements: the batch size.
          
        Returns:
          Three tensors of size (batch_size, h, w, {3, 1, 21}) for images, masks and activations'''
        image_batch, label_batch, activation_batch = tf.train.batch([self.image, self.label, self.activation],
                                                  num_elements)
        return image_batch, label_batch, activation_batch
