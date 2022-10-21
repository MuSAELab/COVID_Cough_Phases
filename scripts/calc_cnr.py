# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 18:21:32 2022

@author: richa
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
    
    cnr_c = cnr_compare_all(r'C:\Users\richa\OneDrive\files\GitHub\COVID_Cough_Phases\data\ComParE_2021\annotation',
                            r'C:\Users\richa\OneDrive\desktop\PROJECTS\COVID\ComPare\Cough\wav_new',
                            16000)
    
    cnr_d = cnr_dicova2(r'C:\Users\richa\OneDrive\files\GitHub\COVID_Cough_Phases\data\DiCOVA2\annotation',
                        r'C:\Users\richa\OneDrive\desktop\PROJECTS\COVID\Dicova\Second_DiCOVA_Challenge_Dev_Data_Release\Second_DiCOVA_Challenge_Dev_Data_Release\AUDIO\cough',
                        16000)
    
    cnr_all = np.concatenate((cnr_c,cnr_d),axis=0)
    cnr_all_df = pd.DataFrame(cnr_all)
    cnr_all_df['label'] = label
    cnr_all_df.columns = ['cnr','label']
    pos_idx = cnr_all_df.index[cnr_all_df['label'] == 1].tolist()
    neg_idx = cnr_all_df.index[cnr_all_df['label'] == 0].tolist()
    
    cnr_c_df = pd.DataFrame(cnr_c)
    cnr_c_df['label'] = label_c
    cnr_c_df.columns = ['cnr','label']
    pos_idx = cnr_c_df.index[cnr_c_df['label'] == 1].tolist()
    neg_idx = cnr_c_df.index[cnr_c_df['label'] == 0].tolist()
    
    cnr_d_df = pd.DataFrame(cnr_d)
    cnr_d_df['label'] = label_d
    cnr_d_df.columns = ['cnr','label']
    pos_idx = cnr_d_df.index[cnr_d_df['label'] == 1].tolist()
    neg_idx = cnr_d_df.index[cnr_d_df['label'] == 0].tolist()
    
    """ number of cough per second per recording """
    plt.figure(dpi=300)
    sns.histplot(cnr_all_df[(cnr_all_df['label']==1) & (cnr_all_df['cnr']>-20)],
                 x='cnr',
                 stat='density',
                 color = 'red',
                 kde = True)
    
    sns.histplot(cnr_all_df[(cnr_all_df['label']==0) & (cnr_all_df['cnr']>-20)],
                 x='cnr',
                 stat='density',
                 color = 'blue',
                 kde = True)
    plt.legend(labels=['Positive','Negative'])
    
    scipy.stats.ttest_ind(cnr_d_df[(cnr_d_df['label']==0)]['cnr'],
                          cnr_d_df[(cnr_d_df['label']==1)]['cnr'],
                          equal_var=False)