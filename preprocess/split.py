#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 02:40:23 2019

@author: felllix
@modified by akshitac8

"""

from pathlib import Path
import cv2
import os
import numpy as np
from natsort import natsorted
from glob import glob
import re
from shutil import copyfile
import sys

import argparse

import tqdm

PATCH_SIZE = 800


def main(source_path: Path, target_path: Path, modes1, mode2, patch_H, patch_W, overlap):
    for i in modes1:
        if i == 'train' or i == 'val':
            extras = ['','_instance_color_RGB','_instance_id_RGB']
        elif i == 'test':
            extras = ['']
        else:
            print('Invalid input')

    for mode1 in modes1:
        print(f'Splitting {mode1} images')
        source_path = source_path / mode1 / mode2
        target_path = target_path / mode1 / mode2

        os.makedirs(target_path, exist_ok=True)

        files = source_path.glob("*.png")
        files = [os.path.split(i)[-1].split('.')[0] for i in files if '_' not in os.path.split(i)[-1]]
        files = natsorted(files)
        
        if len(files) == 0:
            print(f'No files found at {source_path}')
        else:
            for file_ in tqdm.tqdm(files):
                # print(f'Splitting {file_}')
                if file_ == 'P1527' or file_ == 'P1530':
                    continue
                for extra in extras:
                    filename = file_ + extra + '.png'
                    full_filename = source_path / filename
                    img = cv2.imread(str(full_filename))
                    img_H, img_W, _ = img.shape
                    X = np.zeros_like(img, dtype=float)
                    h_X, w_X,_ = X.shape
                    if img_H > PATCH_SIZE and img_W > PATCH_SIZE:
                        for x in range(0, img_W, patch_W-overlap):
                            for y in range(0, img_H, patch_H-overlap):
                                x_str = x
                                x_end = x + patch_W
                                if x_end > img_W:
                                    diff_x = x_end - img_W
                                    x_str-=diff_x
                                    x_end = img_W
                                y_str = y
                                y_end = y + patch_H
                                if y_end > img_H:
                                    diff_y = y_end - img_H
                                    y_str-=diff_y
                                    y_end = img_H
                                patch = img[y_str:y_end,x_str:x_end,:]
                                image = file_+'_'+str(y_str)+'_'+str(y_end)+'_'+str(x_str)+'_'+str(x_end)+extra+'.png'
                                save_path = target_path / image
                                if not os.path.isfile(save_path):
                                    cv2.imwrite(str(save_path), patch)
                    else:
                        copyfile(full_filename, target_path / filename)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Splitting the Images')
    parser.add_argument('--source', default='./dataset/iSAID', type=str, help='path for the original dataset')
    parser.add_argument('--target', default='./dataset/iSAID_patches', type=str, help='path for saving the new dataset')
    parser.add_argument('--image_sub_folder', default='images', type=str, help='name of subfolder inside the training, validation and test folders')
    parser.add_argument('--set', default="train,val,test", type=str, help='evaluation mode')
    parser.add_argument('--patch_width', default=800, type=int, help='Width of the cropped image patch')
    parser.add_argument('--patch_height', default=800, type=int, help='Height of the cropped image patch')
    parser.add_argument('--overlap_area', default=200, type=int, help='Overlap area')


    args = parser.parse_args()

    source_path = Path(args.source)
    target_path = Path(args.target)
    modes1 = args.set.split(',')
    mode2 = args.image_sub_folder
    patch_H, patch_W = args.patch_width, args.patch_height # image patch width and height
    overlap = args.overlap_area #overlap area
    extras = []

    main(source_path, target_path, modes1, mode2, patch_H, patch_W, overlap)

