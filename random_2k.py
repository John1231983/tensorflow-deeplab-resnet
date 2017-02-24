import os
import numpy as np

DATA_LIST_PATH = './dataset/train.txt'
SAVE_LIST_PATH = {0:'./dataset/2k_random_samples_1.txt',
                  1:'./dataset/2k_random_samples_2.txt',
                  2:'./dataset/2k_random_samples_3.txt',
                  3:'./dataset/2k_random_samples_4.txt',
                  4:'./dataset/2k_random_samples_5.txt',
                  5:'./dataset/2k_random_samples_6.txt'}

MAX_ITER = 10582
MAX_SAVE = 2000
def main():
    
    # Get the file names from the list
    f = open(DATA_LIST_PATH, 'r')
    image_name = []
    for line in f:
        image_name.append(line)

    # Generate random indices
    indices = np.arange(MAX_ITER)
    np.random.shuffle(indices)

    for j in range(len(SAVE_LIST_PATH)):
        print(SAVE_LIST_PATH[j])
    
        # Store the random sample list to a file
        random_samples_list = open(SAVE_LIST_PATH[j], 'w')

        for i in range(j*MAX_SAVE, (j+1)*MAX_SAVE):
            if(i>(MAX_ITER-1)):
                break
            index = indices[i]
            img = image_name[index]
            random_samples_list.write(img)

        random_samples_list.close()

if __name__ == '__main__':
    main()

