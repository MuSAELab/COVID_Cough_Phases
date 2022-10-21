# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 11:02:51 2022

@author: richa
"""

import os,re,glob
import numpy as np
import pandas as pd
import textgrid
import pickle as pkl

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


def save_data(data_path,data_name,data):
    
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        
    # Store data (serialize)
    with open(os.path.join(data_path,'%s.pkl'%(data_name)), 'wb') as handle:
        pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)
    
    print("--------")
    print('Data saved in '+data_path)
    
    return 0

def load_feature(feature_path:str):
    
    with open(feature_path, 'rb') as f:
        data = pkl.load(f)
    
    return data