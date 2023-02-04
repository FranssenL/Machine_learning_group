import numpy as np
import pandas as pd
import sys
import os
import cv2
from skimage.transform import rotate
import random

def read_data(path):
    
    with open(path) as f:
        i = 0
        data = np.zeros((2000,240))
        for line in f:
            digits = "".join(line.split())
            for j in range(len(digits)):
                data[i][j] = pd.to_numeric(digits[j])
            i = i + 1 
    #normalize values from 0 to 1
    max_value = np.amax(data)
    data = data/max_value
    data = data.reshape(2000,16,15)
    return(data)

def augment_data(data_x):
    aug_data = np.zeros((2000,240))
    aug_data = aug_data.reshape(2000,16,15)
    for i in range(len(data_x)):
        aug_data[i] = rotate(data_x[i], angle=random.randrange(1,20))
    return aug_data

def erode(data): 
    eroded_data = np.zeros((2000,240))
    eroded_data = eroded_data.reshape(2000,16,15)
    kernel = np.ones((3, 3), np.uint8)
    for i in range(len(data)):
        eroded_data[i] = cv2.erode(data[i], kernel, iterations=1)
    return eroded_data

def dilute(data): 
    diluted_data = np.zeros((2000,240))
    diluted_data = diluted_data.reshape(2000,16,15)
    kernel = np.ones((3, 3), np.uint8)
    for i in range(len(data)):
        diluted_data[i] = cv2.erode(data[i], kernel, iterations=1)
    return diluted_data

def write_to_txt(data):
    data = data.reshape(8000,240)
    file = open("final.txt", "w+")
    for i in range(len(data)):
        for item in data[i]:
            d_string = str(item)
            file.write(d_string+" ")
        file.write("\n")
    file.close()

def main():
    os.chdir(os.path.dirname(sys.argv[0]))
    data = read_data('mfeat_pix.txt')
    aug_data = augment_data(data)
    new_data = np.concatenate((data,aug_data),axis=0)
    eroded_data =  erode(data)
    diluted_data = dilute(data)
    new_data = np.concatenate((new_data, eroded_data),axis=0)
    final_data = np.concatenate((new_data, diluted_data), axis=0)
    write_to_txt(final_data)
    
if __name__ == "__main__":
    main()
