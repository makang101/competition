# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 12:15:11 2018

@author: makang

__project__ = data_augmentation

"""
import os
import os.path

filepath = 'C:\\Users\Administrator\\jupyter\\data_augmentation\\蝴蝶图库'
filepath = filepath.replace('\\','/')
#print(filepath)
filepathsets = []
filepathsets = gothrough(filepath,filepathsets)
for i in range(len(filepathsets)):
    print('{}:'.format(i) + filepathsets[i])