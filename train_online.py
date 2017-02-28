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
import shutil

# import modules from different scripts
import hard_samples
import activations
import distillation

# Defaults of commandline variables
DATA_DIRECTORY = '/home/VOCdevkit'
TMP_LIST_DIR = './tmp_list_dir'
DATA_LIST = './dataset/train.txt'
NUM_STEPS = 20001
RESTORE_FROM = './deeplab_resnet.ckpt'
SNAPSHOT_DIR = './snapshots/'
MAX_SAVE = 100

SAVE_LIST_PATH_OLD = './dataset/universal_hard_old.txt'
SAVE_LIST_PATH_NEW = './dataset/universal_hard_new.txt'

# Variables for temp directory - No need to get them from commandline
NEW_HARD_SAMPLES_PREV_MODEL = 'new_hard_samples_prev_model.txt'
COMBINED_HARD_SAMPLES_PREV_MODEL = 'comb_hard_samples_prev_model.txt'
TOTAL_SAMPLES_FOR_NEW_MODEL = 'total_samples_new_model.txt'
COMBINED_ACTV_WITH_IMG_AND_MSK = 'combined_actv_with_img_and_msk.txt'
TEMP_ACTV_SAVE_DIR = 'activations'

DEFAULT_ACTV_FILE = '0000_000000.npz'

def get_file_len(f_name):
    """Get the length of a file

    Inputs:
        f_name          File name/path

    Outputs:
        Length of a file
    """
    f = open(f_name, 'r')
    length = sum(1 for _ in f)
    f.close()
    return length

def get_file_path(data_dir, data_list):
    """Get a file path

    Inputs:
        data_dir        Path to a directory
        data_list       File name

    Outputs:
        Path to the file
    """
    return data_dir + "/" + data_list 

def combine_files(output_file, input_file1, intput_file2):
    """Combines any two files into a third file

    Inputs:
        input_file1     Path to file1
        intput_file2    Path to file2

    Outputs:
        output_file     Combined file
    """
    output_images = []

    f = open(input_file1, 'r')
    for line in f:
        output_images.append(line)
    f.close()

    f = open(intput_file2, 'r')
    for line in f:
        output_images.append(line)
    f.close()

    f = open(output_file, 'w')
    for i in range(len(output_images)):
        f.write(output_images[i])
    f.close()

def combine_files_with_collisions(output_file, input_file1, intput_file2):
    """Combines any two files into a third file with removing collisions

    Inputs:
        input_file1     Path to file1
        intput_file2    Path to file2

    Outputs:
        output_file     Combined file with collisions removed
    """
    output_images = []

    f = open(input_file1, 'r')
    for line in f:
        output_images.append(line)
    f.close()

    f = open(intput_file2, 'r')
    for line in f:
        if not any(line in s for s in output_images):
            output_images.append(line)
    f.close()

    f = open(output_file, 'w')
    for i in range(len(output_images)):
        f.write(output_images[i])
    f.close()

def combine_activations_with_image_and_mask(out_file, input_file1, input_file2):
    """Create a file with list of following type:
        /path/to/image /path/to/mask /path/to/activations

    Inputs:
        input_file1: Path to file1
        input_file2: Path to file2 -> This file contains hard-samples

    Outputs:
        out_file: Output file with the listings of images, mask and activations
    """
    input1_images = []
    input2_images = []
    output_images = []
    f = open(input_file1, 'r')
    for line in f:
        input1_images.append(line)
    f.close()

    f = open(input_file2, 'r')
    for line in f:
        input2_images.append(line)
    f.close()

    for line in input1_images:
        if any(line in s for s in input2_images):
            sub_str = line.strip("\n").rsplit('/', 1)[1].replace('png', 'npz')
        else:
            sub_str = DEFAULT_ACTV_FILE
            
        line = line.strip("\n") + " /" + sub_str + "\n"
        output_images.append(line)

    f = open(out_file, 'w')
    for line in output_images:
        f.write(line)
    f.close()

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--tmp-list-dir", type=str, default=TMP_LIST_DIR,
                        help="Path to the directory where intermediate results are stored.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST,
                        help="Path to the file containing new data list.")
    parser.add_argument("--old-data-list", type=str, default=SAVE_LIST_PATH_OLD,
                        help="Path to the file containing list of hardsamples of old data.")
    parser.add_argument("--new-data-list", type=str, default=SAVE_LIST_PATH_NEW,
                        help="Path to the file where hardsamples on new model will be stored.")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--max-save", type=int, default=MAX_SAVE,
                        help="Maximum number of hardsamples to store.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    return parser.parse_args()

def main():
    """Main function to run online training algorithm.

    Output:
        A trained model on new and old data-sets
        A list containing hardsamples on trained model
    """
    # Get the command-line arguments
    args = get_arguments()

    # Create tmp data directory to store intermediate lists if not present
    if not os.path.exists(args.tmp_list_dir):
        os.makedirs(args.tmp_list_dir)

    #######################################################################
    # Calculate hard-samples from new-data on old model
    #######################################################################
    print('Calculating entropy of new data on previous model...')
    new_data_hard_list_pmodel = get_file_path(args.tmp_list_dir, NEW_HARD_SAMPLES_PREV_MODEL)
    iterations = get_file_len(args.data_list)

    hard_samples.main(data_dir=args.data_dir, data_list=args.data_list,\
                      save_list=new_data_hard_list_pmodel, num_steps=iterations,\
                      max_save=args.max_save, restore_from=args.restore_from)

    #######################################################################
    # Calculate activations of old model on all the hard-samples
    #######################################################################
    # Combine the hard-samples from new and old data in one file
    combined_list_hard_prev_model = get_file_path(args.tmp_list_dir, COMBINED_HARD_SAMPLES_PREV_MODEL)
    combine_files(combined_list_hard_prev_model, args.old_data_list, new_data_hard_list_pmodel)
    iterations = get_file_len(combined_list_hard_prev_model)
    actv_dir = get_file_path(args.tmp_list_dir, TEMP_ACTV_SAVE_DIR)
    print('Calculating and storing activations in %s ...'%(actv_dir))
    if not os.path.exists(actv_dir):
        os.makedirs(actv_dir)

    activations.main(data_dir=args.data_dir, data_list=combined_list_hard_prev_model,\
                    save_dir=actv_dir, num_steps=iterations, restore_from=args.restore_from)

    #######################################################################
    # Train the new model using classification and distillation loss
    #######################################################################
    # Append the activation files to a list [/path/to/image /path/to/masks /path/to/activations]
    combined_list_actv_with_img_and_mask = get_file_path(args.tmp_list_dir, COMBINED_ACTV_WITH_IMG_AND_MSK)
    combine_activations_with_image_and_mask(combined_list_actv_with_img_and_mask,\
                                            args.data_list, combined_list_hard_prev_model)
    print('Started training of the model using classification and distillation losses ...')
    distillation.main(data_dir=args.data_dir, actv_dir=actv_dir, data_list=combined_list_actv_with_img_and_mask,\
                     num_steps=args.num_steps, restore_from=args.restore_from, snapshot_dir=args.snapshot_dir)

    #######################################################################
    # Get the hard-samples on newly trained model
    #######################################################################
    print('Calculating entropy of new data on new model...')
    total_samples_list_new_model = get_file_path(args.tmp_list_dir, TOTAL_SAMPLES_FOR_NEW_MODEL)
    combine_files_with_collisions(total_samples_list_new_model, args.data_list, combined_list_hard_prev_model)
    iterations = get_file_len(total_samples_list_new_model)
    model = args.snapshot_dir + '/' + 'model.ckpt-5000' #TODO: Fix this
    hard_samples.main(data_dir=args.data_dir, data_list=total_samples_list_new_model,\
                     save_list=args.new_data_list, num_steps=iterations, max_save=args.max_save,\
                     restore_from=model)

    # Clean the intermediata temporary directory
    print('Cleaning the temporary directory %s...'%(args.tmp_list_dir))
    shutil.rmtree(args.tmp_list_dir)

if __name__ == '__main__':
    main()
    
