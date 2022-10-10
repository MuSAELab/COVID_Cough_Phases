# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 08:46:02 2022

@author: richa
"""


""" 
Given the path of an audio file and its corresponding annotation file, 
segment the audio into chunks, extract acoustic features from each chunk, then
calculate LLD across chunks.
"""

import re
import numpy as np
import textgrid
import opensmile
import librosa

def numericalSort(value):
    
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


def get_segment(audio:np.array,
                fs:int,
                onset:float,
                offset:float):
    
    seg = audio[int(onset*fs):int(offset*fs)]
    
    return seg


def get_stamp(ant_path:str):
    
    tg_segment = textgrid.TextGrid.fromFile(ant_path)
    
    phase_stamps = {}
    phase_stamps['inhale_list'] = []
    phase_stamps['compress_list'] = []
    phase_stamps['cough_list'] = []
    
    for z in tg_segment[0]:
        
        if(z.mark=="inhale"):
            phase_stamps['inhale_list'].append([z.minTime,z.maxTime])
        if(z.mark=="compress"):
            phase_stamps['compress_list'].append([z.minTime,z.maxTime])
        if(z.mark=="cough"):
            phase_stamps['cough_list'].append([z.minTime,z.maxTime])
    
    return phase_stamps
    

def separate_phase(audio:np.array,
                   fs:int,
                   stamps:list):
    
    phase_ad = []
    
    for seg in stamps:

        seg_ad = get_segment(audio=audio,
                             fs=fs,
                             onset=seg[0],
                             offset=seg[1])
        phase_ad.append(seg_ad)
    
    return phase_ad


smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

def get_SMILE(audio_file:str):
    
    smile_fea = smile.process_file(audio_file)
    smile_fea = np.array(smile_fea)
    smile_fea = smile_fea.reshape((6373))
    
    return smile_fea


def load_one_ad(ad_name,fs=16000):
    
    ad, _ = librosa.load(ad_name,sr=fs)
    ad = ad/np.max(abs(ad)) #amplitude normalize between -1 and 1
    return ad