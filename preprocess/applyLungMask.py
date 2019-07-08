# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 17:24:16 2019

@author: Lab806_J
"""

import cv2
from tqdm import tqdm
#import numpy as np
#from matplotlib import pyplot as plt
import os
from glob import glob

img_root = 'cleanNodules'

def apply_mask(file_path='test'):
     if 'train' in file_path:
          img_files = glob(os.path.join(img_root, 'train', '*.png'))
          mask_files = glob(os.path.join(img_root, 'lung_mask_train', '*.png'))
          save_path = 'lung_mask_train'
     elif 'test' in file_path:
          img_files = glob(os.path.join(img_root, 'test', '*.png'))
          mask_files = glob(os.path.join(img_root, 'lung_mask_test', '*.png'))
          save_path = 'lung_mask_test'
     
     for img_file, mask_file in tqdm(zip(img_files, mask_files)):
          image = cv2.imread(img_file, 0)
          mask = cv2.imread(mask_file, 0)
          mask[mask==3] = 1
          mask[mask==4] = 1
          mask[mask!=1] = 0
          image = image * mask
          image[mask==0] = 170
          cv2.imwrite(os.path.join(img_root, save_path, '{}'.format(os.path.basename(img_file))), image)
     
     print('*'*25, 'Done', '*'*25)
     

if __name__ == '__main__':
     apply_mask()
          
