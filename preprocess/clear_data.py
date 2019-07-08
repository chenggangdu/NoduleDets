# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:30:38 2019

@author: Lab806_J
"""

import os
from tqdm import tqdm
from luna16preprocess import readCSV
from glob import glob
import cv2
from matplotlib import pyplot as plt
import shutil
import csv

anno_path = './CSVFILES/cleanAnnotations.csv'
annotations = readCSV(anno_path)

image_files = glob('Images/*.png')
image_files.sort()

def draw_bbox(image, bbox):
     x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]),int(bbox[2]),int(bbox[3])
     cv2.rectangle(image, (x1, y1), (x2, y2), 255, 2)
     return image
def saveImagesWithGT(image_files, annotations):
     for img_file, ann in tqdm(zip(image_files, annotations)):
          img = cv2.imread(img_file, 0)
          bbox = ann[1:-1]
          img = draw_bbox(img, bbox)
          cv2.imwrite('ImagesWithGT/{}'.format(os.path.basename(img_file)), img)
     print('*'*25, 'Done', '*'*25)
     
def get_ignore_anns(annotations):
     anns = []
     for ann in annotations:
          ignore = ann[-2]
          if ignore == '1':
               continue
          anns.append(ann)
     return anns

if __name__ == '__main__':

     for i, ann in tqdm(enumerate(annotations)):
          if ann[-2] == '1':
               continue
          if '{:06}.png'.format(i+1) != ann[-1]:
               continue
          img_file = os.path.join('noduleMasks', ann[-1])
          shutil.copy(img_file, 'cleanNoduleMasks')