# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 10:06:10 2019

@author: Lab806_J
"""

import SimpleITK as sitk
import numpy as np
import csv
import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import shutil

def load_itk_image(filename):
     itkimage = sitk.ReadImage(filename)
     numpyImage = sitk.GetArrayFromImage(itkimage)
     numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
     numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     return numpyImage, numpyOrigin, numpySpacing

def load_mhd_lungseg(filename):
     itkimage = sitk.ReadImage(filename)
     numpyImage = sitk.GetArrayFromImage(itkimage)
     return numpyImage

def readCSV(filename):
     lines = []
     with open(filename, "r") as f:
          csvreader = csv.reader(f)
          for line in csvreader:
               lines.append(line)
     return lines

def worldToVoxelCoord(worldCoord, origin, spacing):
     stretchedVoxelCoord = np.absolute(worldCoord - origin)
     voxelCoord = stretchedVoxelCoord / spacing
     return voxelCoord

def VoxelToWorldCoord(voxelCoord, origin, spacing):
    strechedVocelCoord = voxelCoord * spacing
    worldCoord = strechedVocelCoord + origin
    return worldCoord

def normalizePlanes(npzarray, maxHU = 400., minHU = -1000.):
     npzarray = (npzarray - minHU) / (maxHU - minHU)
     npzarray[npzarray>1] = 1.
     npzarray[npzarray<0] = 0.
     return npzarray

def getSubsetFiles(seglung_path=glob('subset*')):
     total_files = []
     for i in range(len(seglung_path)):
          file = glob(os.path.join(seglung_path[i], '*.mhd'))
          for f in file:
               total_files.append(f)
     return total_files

def getLungSegFiles(path='seg-lungs-LUNA16'):
     lungseg_files = glob(os.path.join(path, '*.mhd'))
     return lungseg_files

def match(annotations, mhd_files, tif_save_path='Images', lungseg_path='lungSegImages', noduleMask_path='noduleMasks'):
     for i, annline in tqdm(enumerate(annotations)):
          for fmhd in mhd_files:
               if annline[0] in fmhd.split('\\')[1].split('.mhd')[0]:
                    diameter = float(annline[-1])
                    # get worldCoord
                    worldCoord = np.array([annline[3], annline[2], annline[1]]).astype(np.float16)
                    # load mhd images
                    mhdImage, Origin, Spacing = load_itk_image(fmhd)
                    # normalizePlanes
                    mhdImage = normalizePlanes(mhdImage)
                    
                    # get voxelCoord
                    voxelCoord = worldToVoxelCoord(worldCoord, Origin, Spacing)
                    diameter = np.ceil(diameter / Spacing[1])+5
                    # get nodule bbox location
                    idZ = np.round(voxelCoord[0]).astype(np.uint8)
                    posY1 = np.round(voxelCoord[1] - diameter / 2).astype(np.int16)
                    posX1 = np.round(voxelCoord[2] - diameter / 2).astype(np.int16)
                    posY2 = np.round(voxelCoord[1] + diameter / 2).astype(np.int16)
                    posX2 = np.round(voxelCoord[2] + diameter / 2).astype(np.int16)
                    
                    label = [fmhd, posX1, posY1, posX2, posY2, idZ]
                    with open('CSVFILES/newAnnotations.csv', 'a', newline='') as f:
                         writer = csv.writer(f)
                         writer.writerows([label])
                    image = mhdImage[idZ]*255     
                    flungseg = os.path.join('seg-lungs-LUNA16', annline[0]+'.mhd')
                    lungsegImage = load_mhd_lungseg(flungseg)[idZ]
                    cv2.imwrite(os.path.join(tif_save_path, '{:06}.png'.format(i+1)), image)
                    cv2.imwrite(os.path.join(lungseg_path, '{:06}.png'.format(i+1)), lungsegImage)
                    extractNoduleMask(i, image, label[1:-1], noduleMask_path)
     
     print('*'*20, 'Done.', '*'*20)

def extractNoduleMask(image_id, image, bbox, noduleMask_path):
     """debug:
          plt.figure()
          plt.subplot(1,3,1)
          plt.imshow(image, cmap='gray')
          plt.subplot(1,3,2)
          plt.imshow(img, cmap='gray')
          plt.subplot(1,3,3)
          plt.imshow(mask, cmap='gray')
     """
     mask = np.zeros_like(image).astype(np.uint8)
     y1, x1, y2, x2 = bbox[0], bbox[1], bbox[2], bbox[3]
     mask[x1:x2, y1:y2] = 1
     img = image.copy()
     img = img * mask
     mean = np.sum(img) / np.sum(mask)
     idx = img >= 0.5*mean
     mask[:] = 0
     mask[idx] = 1
     
     cv2.imwrite(os.path.join(noduleMask_path, '{:06}.png'.format(image_id+1)), mask*255)

# get clean nodule masks
def draw_nodule_masks(image, bboxes):
    mask = np.zeros_like(image).astype(np.uint8)
    for bbox in bboxes:
        y1, x1, y2, x2 = bbox[0], bbox[1], bbox[2], bbox[3]
        patch = image[int(x1):int(x2), int(y1):int(y2)]
        mean = np.mean(patch)
        idx = image[int(x1):int(x2), int(y1):int(y2)] > 0.6 * mean
        mask[int(x1):int(x2), int(y1):int(y2)][idx] = 255
    return mask

# get gtbbox on images
def get_cleanImagesGT(image_files, annotations, save_path='cleanImagesWithGT'):
    for img_file in tqdm(image_files):
        img = cv2.imread(img_file, 0)
        img_id = os.path.basename(img_file).split('.')[0]
        
        for ann in annotations:
            if img_id in ann[-1]:
                x1, y1, x2, y2 = int(ann[1]), int(ann[2]), int(ann[3]), int(ann[4])
                cv2.rectangle(img, (x1, y1), (x2, y2), 255, 1)
        cv2.imwrite(os.path.join(save_path, os.path.basename(img_file)), img)
    print('All done.')
        
def CalBoxSizeDistribution(image_files, annotations):
    bboxes = []
    for img_file in tqdm(image_files):
        img_id = os.path.basename(img_file).split('.')[0]
        for ann in annotations:
            if img_id in ann[-1]:
                x1, y1, x2, y2 = int(ann[1]), int(ann[2]), int(ann[3]), int(ann[4])
                h = y2 - y1
                w = x2 - x1
                area = w * h
                bboxes.append([area, h, w])
    bboxes = np.array(bboxes)
    # plot hist
    sizes = np.sqrt(bboxes[:, 0])
    bins = len(set(sizes)) # best value is 35, for this time
    plt.hist(sizes, bins=bins, normed=0, facecolor="red", edgecolor="black", alpha=0.6)
    # 显示横轴标签
    plt.xlabel("nodule sizes")
    # 显示纵轴标签
    plt.ylabel("number of nodule per size")
    # 显示图标题
    plt.title("distributions of nodule size")
    plt.show()
    
def draw_single_nodule_mask(image, bbox):
    mask = np.zeros_like(image).astype(np.uint8)
    y1, x1, y2, x2 = bbox[0], bbox[1], bbox[2], bbox[3]
    patch = image[int(x1):int(x2), int(y1):int(y2)]
    mean = np.mean(patch)
    idx = image[int(x1):int(x2), int(y1):int(y2)] > 0.6 * mean
    mask[int(x1):int(x2), int(y1):int(y2)][idx] = 255
    return mask
  
def extractCleanNoduleMask(image_files, annotations, nodule_save_path='cleanInstanceNoduleMasks'):
    for img_file in tqdm(image_files):
        img = cv2.imread(img_file, 0)
        img_id = os.path.basename(img_file).split('.')[0]
        bboxes = []
        for ann in annotations:
            if img_id in ann[-1]:
                bboxes.append(ann[1:-3])
        for i, bbox in enumerate(bboxes):
             mask = draw_single_nodule_mask(img, bbox)
             cv2.imwrite(os.path.join(nodule_save_path, img_id + '_{:02}.png'.format(i+1)), mask)
    print('*'*25, 'Done', '*'*25)

def newAnn2cleanAnn(newAnn_path='./CSVFILES/newAnnotations1.csv', 
                    cleanAnn_path='./CSVFILES/cleanAnnotations1.csv',
                    cleanImage_path='cleanImages',
                    cleanImagesWithGT_path='cleanImagesWithGT',
                    cleanInstanceNoduleMasks_path='cleanInstanceNoduleMasks',
                    cleanLungMasks_path='cleanLungMasks',
                    cleanNodules_path='cleanNodules',):
    if os.path.exists(cleanAnn_path):
        os.remove(cleanAnn_path)
    # had done: newAnnotations.csv to cleanAnnotations1.csv
    newAnn = readCSV(newAnn_path)
    for i, ann in tqdm(enumerate(newAnn, 1)):
        if ann[-1] == '0':
            ann[-1] = '{:06}.png'.format(i)
        elif ann[-1] != '0':
            ann[-1] = '{:06}.png'.format(int(ann[-1]))
        with open(cleanAnn_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows([ann])
    
    cleanAnn = readCSV(cleanAnn_path)
    for ann in tqdm(cleanAnn):
        if ann[-2] == '1':
            continue
        img_srcfile = os.path.join('Images', ann[-1])
        img_dstfile = os.path.join(cleanImage_path, ann[-1])
        shutil.copy(img_srcfile, img_dstfile)
        lungmask_srcfile = os.path.join('lungMasks', ann[-1])
        lungmask_dstfile = os.path.join(cleanLungMasks_path, ann[-1])
        shutil.copy(lungmask_srcfile, lungmask_dstfile)
        
           
if __name__ == '__main__':
#    newAnn2cleanAnn() 
    anno_path = './CSVFILES/cleanAnnotations1.csv'
    annotations = readCSV(anno_path)
     
    image_files = glob('cleanImages/*.png')
    image_files.sort()
#    get_cleanImagesGT(image_files, annotations)
#    extractCleanNoduleMask(image_files, annotations)
    
    CalBoxSizeDistribution(image_files, annotations)
    
     
     
     
     
