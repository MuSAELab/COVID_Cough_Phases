# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:21:32 2022

@author: Yi Zhu
@e-mail: Yi.Zhu@inrs.ca
"""

"""
This script is used for calculating cough-to-nosie ratio (CNR).
"""
import os,glob
import numpy as np
import textgrid
import feature_func as ffunc
import librosa
import soundfile as sf
import pickle as pkl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy


def ave_power(audio:np.array,
              fs:int,
              phase_stamps:tuple):
    
    ave_p = 1e-5
    if len(phase_stamps) != 0:
        
        phase_ad = ffunc.separate_phase(audio=audio,
                                        fs=fs,
                                        stamps=phase_stamps)
        
        power = np.zeros((len(phase_stamps,)))
        for i,ad in enumerate(phase_ad):
            # normalized power
            power[i] = np.sum(np.abs(ad)**2)/len(ad)
        
        ave_p = np.mean(power)
        
    return ave_p



def get_cnr(audio_path:str,
            fs:str,
            ant_path:str):
    
    audio = ffunc.load_one_ad(ad_name=audio_path, fs=fs)
    all_phase_stamps = ffunc.get_stamp(ant_path=ant_path)
    phase = ['cough','noise']
    cough_stamps = all_phase_stamps['%s_list'%(phase[0])]
    noise_stamps = all_phase_stamps['%s_list'%(phase[1])]
    p_cough = ave_power(audio,fs,cough_stamps)
    if len(noise_stamps) != 0: 
        p_noise = ave_power(audio,fs,noise_stamps[:2])
    elif len(noise_stamps) == 0:
        p_noise = ave_power(audio,fs,noise_stamps)
    CNR = 10*np.log(p_cough) - 10*np.log10(p_noise)
    
    return CNR



def cnr_compare(tg_folder_path:str,
                ad_folder_path:str,
                fs:int,
                split:str):
    
    ot = []
    
    for file in sorted(glob.glob(os.path.join(tg_folder_path, '%s_*.TextGrid'%(split))),key=ffunc.numericalSort):
    
        subject_id = os.path.split(file)[1][:-9]
        audio_path = os.path.join(ad_folder_path,'%s.wav'%(subject_id))
        ot.append(get_cnr(audio_path,fs,file))
    
    ot = np.asarray(ot)
    
    return ot


def cnr_compare_all(tg_folder_path:str,
                    ad_folder_path:str,
                    fs:int):
    
    ot_train = cnr_compare(tg_folder_path,
                           ad_folder_path,
                           fs,
                           'train')
    
    ot_devel = cnr_compare(tg_folder_path,
                           ad_folder_path,
                           fs,
                           'devel')
    
    ot_test = cnr_compare(tg_folder_path,
                           ad_folder_path,
                           fs,
                           'test')
    
    ot = np.concatenate((ot_train,ot_devel,ot_test),axis=0)
    
    return ot


def cnr_dicova2(tg_folder_path:str,
                ad_folder_path:str,
                fs:int):
    
    ot = []
    
    for file in sorted(glob.glob(os.path.join(tg_folder_path, '*.TextGrid')),key=ffunc.numericalSort):

        subject_id = os.path.split(file)[1][:-9]
        audio_path = os.path.join(ad_folder_path,'%s.flac'%(subject_id))
        ot.append(get_cnr(audio_path,fs,file))
    
    ot = np.asarray(ot)
    
    return ot


if __name__ == '__main__':
    
    cnr_c = cnr_compare_all(r'ROOT_PATH\COVID_Cough_Phases\data\ComParE_2021\annotation',
                            r'PATH_TO_COMPARE_AUDIO_FOLDER',
                            16000)
    
    cnr_d = cnr_dicova2(r'ROOT_PATH\GitHub\COVID_Cough_Phases\data\DiCOVA2\annotation',
                        r'PATH_TO_DICOVA2_AUDIO_FOLDER',
                        16000)
   
