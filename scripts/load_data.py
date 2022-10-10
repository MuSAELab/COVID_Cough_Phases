# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 11:02:51 2022

@author: richa
"""

import os,re,glob
import numpy as np
import pandas as pd
import textgrid

def numericalSort(value):
    
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def load_data_compare(label_path:str,
                      ant_path:str,
                      split:str):
    
    df = pd.read_csv(os.path.join(label_path,'%s_new.csv'%(split)))
    codes = {'negative':0, 'positive':1}
    df['label'] = df['label'].map(codes)
    label = np.array(df['label'])
    
    tg = []
    ad = []
    for file in sorted(glob.glob(os.path.join(ant_path, '%s_*.TextGrid'%(split))),key=numericalSort):
        tg_segment = textgrid.TextGrid.fromFile(file)
        tg.append(tg_segment)
    
    return tg,label


def load_data_dicova(label_path:str,
                     ant_path:str):
    
    df = pd.read_csv(label_path,delimiter=r'\s+')
    codes = {'n':0, 'p':1}
    df['COVID_STATUS'] = df['COVID_STATUS'].map(codes)
    
    tg = []
    name = []
    label = []
    
    for file in sorted(glob.glob(os.path.join(ant_path, '*.TextGrid')),key=numericalSort):
        filename = os.path.split(file)[1][:-9]
        name.append(filename)
        tg_segment = textgrid.TextGrid.fromFile(file)
        tg.append(tg_segment)
        file_label = df[df['SUB_ID'] == filename]['COVID_STATUS']
        label.append(np.array(file_label))
    
    label = np.concatenate(label).ravel()
    
    return tg,name,label