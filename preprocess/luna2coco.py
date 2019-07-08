# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 21:18:14 2019

@author: Lab806_J
"""

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import cv2
import numpy as np
from pycococreatortools import pycococreatortools
from collections import OrderedDict
import csv
from random import shuffle
from glob import glob
from tqdm import tqdm
from matplotlib import pyplot as plt
import shutil
from pycococreatortools.pycococreatortools import create_annotation_info

class COCOINFO(dict):
     def __init__(self):
          super(COCOINFO, self).__init__()
          INFO = {"description": "Nodule Dataset",
                  "url": "https://github.com/waspinator/pycococreator",
                  "version": "0.1.0",
                  "year": 2018,
                  "contributor": "waspinator",
                  "date_created": datetime.datetime.now().isoformat(' ')
                  }  

          LICENSES = [{"id": 1,
                       "name": "Attribution-NonCommercial-ShareAlike License",
                       "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
                       }]

          CATEGORIES = [{'id': 1,
                         'name': 'nodule',
                         'supercategory': 'nodule',
                         'iscrowd': 0,
                         },]

          self.CocoInfo = {"info": INFO,
                           "licenses": LICENSES,
                           "categories": CATEGORIES,
                           "images": [],
                           "annotations": []
                           }
          
     def update(self, info=None, licenses=None, categories=None, image=None, annotation=None):
          if info is not None:
               self.CocoInfo['info'] = info
          if licenses is not None:
               self.CocoInfo['licenses'] = licenses
          if categories is not None:
               self.CocoInfo['categories'] = categories
          if image is not None:
               self.CocoInfo['images'].append(image)
          if annotation is not None:
               self.CocoInfo['annotations'].append(annotation)

def readCSV(filename='./CSVFILES/newAnnotations.csv'):
     lines = []
     with open(filename, "r") as f:
          csvreader = csv.reader(f)
          for line in csvreader:
               lines.append(line)
     return lines

def getImageInfo(image_id, image_file):
     return {"id": image_id,
             "file_name": os.path.basename(image_file),
             "height": 512,
             "width": 512,
               }
     
def getAnnotations(id, image_file, annotations, binary_mask=None):
     ann_id = int(os.path.basename(image_file).split('.')[0]) - 1
     x1, y1, x2, y2 = (int(i) for i in annotations[ann_id][1:-1])
     h = x2-x1
     w = y2-y1
     
     if binary_mask == None:
          area = (x2-x1) * (y2 - y1)
     
     return {"id":id,
             "image_id":id,
             "category_id":1,
             "ignore":0,
             "iscrowd":1,
             "bbox":[x1,y1,h,w],
             "area":area,
             "segmentation":[], # rle or polygon 
             }

def copy_file(file_path, train=True):
     file_path = os.path.basename(file_path)
     image_file = os.path.join('cleanImages', file_path)
     lungseg_file = os.path.join('cleanLungMasks', file_path)
     nodule_save_path = os.path.join('cleanNodules')
     lungseg_save_path = os.path.join('cleanNodules')
     if train:
          nodule_save_path = os.path.join(nodule_save_path, 'train')
          lungseg_save_path = os.path.join(lungseg_save_path, 'lung_mask_train')
     else:
          nodule_save_path = os.path.join(nodule_save_path, 'test')
          lungseg_save_path = os.path.join(lungseg_save_path, 'lung_mask_test')
     shutil.copy(image_file, nodule_save_path)
     shutil.copy(lungseg_file, lungseg_save_path)
     

def get_binary_mask(file_path):
     file_path = os.path.join('cleanInstanceNoduleMasks', os.path.basename(file_path))
     binary_mask = np.asarray(Image.open(file_path)
                        .convert('1')).astype(np.uint8)
     return binary_mask
     

def getAnnImages(img_file):
     ann_root = 'cleanInstanceNoduleMasks'
     img_basename = os.path.basename(img_file)
     img_id = os.path.splitext(img_basename)[0]
     annmasks = glob(ann_root+'/'+img_id+'_*.png')
     annmasks.sort()
     return annmasks

def main():
     """debug:
          bbox = cocoInfo.CocoInfo['annotations'][-1]['bbox']
          img = cv2.imread(image_files[i], 0)
          img = cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]), 255, 2)
          plt.figure()
          plt.imshow(img, cmap='gray')
          if i > 10:
               return
     return cocoInfo.CocoInfo['images'], cocoInfo.CocoInfo['annotations']
     """
     coco_train = COCOINFO()
     coco_test = COCOINFO()
#     csv_annos = readCSV()
     image_files = glob('cleanImages/*.png')
     shuffle(image_files)
     ntrain = np.round(len(image_files)*0.8).astype(np.int)
     segmentation_id = 0
     for i in tqdm(range(ntrain)):
          img_id = i + 1
          copy_file(image_files[i], train=True)
          imageInfo = getImageInfo(img_id, image_files[i])
          annmasks = getAnnImages(image_files[i])
          for annmask in annmasks:
               binary_mask = get_binary_mask(annmask)
               segmentation_id += 1
               annoInfo = pycococreatortools.create_annotation_info(
                        segmentation_id, img_id, category_info={'id': 1, 'is_crowd': 0}, binary_mask=binary_mask,
                        image_size=(512, 512), tolerance=2)
               coco_train.update(image=imageInfo)
               coco_train.update(annotation=annoInfo)
          
     # write train.json file
     with open('cleanNodules/annotations/instance_nodule_train.json', 'a+', newline='') as f:
          json.dump(coco_train.CocoInfo, f)
          
     for i in tqdm(range(ntrain, len(image_files))):
          img_id = i + 1
          copy_file(image_files[i], train=False)
          imageInfo = getImageInfo(img_id, image_files[i])
          annmasks = getAnnImages(image_files[i])
          for annmask in annmasks:
               binary_mask = get_binary_mask(annmask)
               segmentation_id += 1
               annoInfo = pycococreatortools.create_annotation_info(
                        segmentation_id, img_id, category_info={'id': 1, 'is_crowd': 0}, binary_mask=binary_mask,
                        image_size=(512, 512), tolerance=2)
               coco_test.update(image=imageInfo)
               coco_test.update(annotation=annoInfo)
          
     # write test.json file
     with open('cleanNodules/annotations/instance_nodule_test.json', 'a+', newline='') as f:
          json.dump(coco_test.CocoInfo, f)
     
     
     print('*'*25, 'All Done', '*'*25)

def main1():
     coco_train = COCOINFO()
     coco_test = COCOINFO()
     image_files = glob('cleanImages/*.png')
     shuffle(image_files)
     ntrain = np.round(len(image_files)*0.7).astype(np.int)
     
     for i in tqdm(range(ntrain)):
          copy_file(image_files[i], train=True)
          binary_mask = get_binary_mask(image_files[i])
          imageInfo = getImageInfo(i+1, image_files[i])
#          annoInfo = getAnnotations(i+1, image_files[i], csv_annos)
          segmentation_id = i + 1
          image_id = i + 1
          annoInfo = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info={'id': 1, 'is_crowd': 0}, binary_mask=binary_mask,
                        image_size=(512, 512), tolerance=2)
          coco_train.update(image=imageInfo)
          coco_train.update(annotation=annoInfo)
          
     # write train.json file
     with open('cleanNodules/annotations/instance_nodule_train.json', 'a+', newline='') as f:
          json.dump(coco_train.CocoInfo, f)
          
     for i in tqdm(range(ntrain, len(image_files))):
          copy_file(image_files[i], train=False)
          binary_mask = get_binary_mask(image_files[i])
          imageInfo = getImageInfo(i+1, image_files[i])
#          annoInfo = getAnnotations(i+1, image_files[i], csv_annos)
          segmentation_id = i + 1
          image_id = i + 1
          annoInfo = pycococreatortools.create_annotation_info(
                        segmentation_id, image_id, category_info={'id': 1, 'is_crowd': 0}, binary_mask=binary_mask,
                        image_size=(512, 512), tolerance=2)
          coco_test.update(image=imageInfo)
          coco_test.update(annotation=annoInfo)
          
     # write test.json file
     with open('cleanNodules/annotations/instance_nodule_test.json', 'a+', newline='') as f:
          json.dump(coco_test.CocoInfo, f)
     
     
     print('*'*25, 'All Done', '*'*25)     

if __name__ == '__main__':
     main()
     from applyLungMask import apply_mask
     apply_mask('train')
     apply_mask('test')
     
     
     