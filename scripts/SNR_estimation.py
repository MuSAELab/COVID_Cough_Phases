# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 10:42:18 2022

@author: richa
"""

import os,glob
import numpy as np
import pandas as pd
import textgrid
import feature_func as ffunc
import librosa
import soundfile as sf
import pickle as pkl
from blind_SNR import wada_snr
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

# %% Blind SNR estimation

def SNR_compare(tg_folder_path:str,
                ad_folder_path:str,
                fs:int,
                split:str):

    ot = []
    
    for file in sorted(glob.glob(os.path.join(tg_folder_path, '%s_*.TextGrid'%(split))),key=ffunc.numericalSort):
    
        subject_id = os.path.split(file)[1][:-9]
        audio_path = os.path.join(ad_folder_path,'%s.wav'%(subject_id))
        audio = ffunc.load_one_ad(audio_path,fs)
        ot.append(wada_snr(audio[-16000:]))
    
    return ot


def SNR_dicova2(tg_folder_path:str,
                ad_folder_path:str,
                fs:int):
    
    ot = []
    
    for file in sorted(glob.glob(os.path.join(tg_folder_path, '*.TextGrid')),key=ffunc.numericalSort):

        subject_id = os.path.split(file)[1][:-9]
        audio_path = os.path.join(ad_folder_path,'%s.flac'%(subject_id))
        audio = ffunc.load_one_ad(audio_path,fs)
        ot.append(wada_snr(audio[-16000:]))

    return ot

# %% SNR main
if __name__ == '__main__':
    
    snr_compare_train = SNR_compare(r'C:\Users\richa\OneDrive\files\GitHub\COVID_Cough_Phases\data\ComParE_2021\annotation',
                        r'C:\Users\richa\OneDrive\desktop\PROJECTS\COVID\ComPare\Cough\wav_new',
                        16000,
                        'train')
    
    snr_compare_devel = SNR_compare(r'C:\Users\richa\OneDrive\files\GitHub\COVID_Cough_Phases\data\ComParE_2021\annotation',
                    r'C:\Users\richa\OneDrive\desktop\PROJECTS\COVID\ComPare\Cough\wav_new',
                    16000,
                    'devel')
    
    snr_compare_test = SNR_compare(r'C:\Users\richa\OneDrive\files\GitHub\COVID_Cough_Phases\data\ComParE_2021\annotation',
                    r'C:\Users\richa\OneDrive\desktop\PROJECTS\COVID\ComPare\Cough\wav_new',
                    16000,
                    'test')

    snr_dicova = SNR_dicova2(r'C:\Users\richa\OneDrive\files\GitHub\COVID_Cough_Phases\data\DiCOVA2\annotation',
                            r'C:\Users\richa\OneDrive\desktop\PROJECTS\COVID\Dicova\Second_DiCOVA_Challenge_Dev_Data_Release\Second_DiCOVA_Challenge_Dev_Data_Release\AUDIO\cough',
                            16000)
    
    snr_compare = np.concatenate((snr_compare_train,snr_compare_devel,snr_compare_test),axis=0)
    snr_compare = np.nan_to_num(snr_compare)
    snr_dicova = np.nan_to_num(snr_dicova)
    
    snr_all = np.concatenate((snr_compare,snr_dicova),axis=0)   
    # snr_all = snr_dicova
    snr_all = pd.DataFrame(snr_all)
    snr_all['label'] = label
    snr_all.columns = ['snr','label']
    
        
    plt.figure(dpi=300)
    sns.histplot(snr_all[(snr_all['label']==1)],
                 x='snr',
                 stat='density',
                 color = 'red',
                 kde = True)
    
    sns.histplot(snr_all[(snr_all['label']==0)],
                 x='snr',
                 stat='density',
                 color = 'blue',
                 kde = True)
    plt.legend(labels=['Positive','Negative'])
    
    scipy.stats.ttest_ind(snr_all[(snr_all['label']==1)]['snr'],
                          snr_all[(snr_all['label']==0)]['snr'],
                          equal_var=False)