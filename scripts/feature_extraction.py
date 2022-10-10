# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 12:53:15 2022

@author: richa
"""

import os,glob
import numpy as np
import textgrid
import feature_func as ffunc
import librosa
import soundfile as sf
import pickle as pkl


def feature_from_phase(audio:np.array,
                       fs:int,
                       phase_stamps:tuple,
                       save_path:str):
    
    ot = np.zeros((6373))
    
    if len(phase_stamps) != 0:
        
        phase_ad = ffunc.separate_phase(audio=audio,
                                        fs=fs,
                                        stamps=phase_stamps)
        
        smile_fea = np.zeros((len(phase_stamps),6373))
        
        for i in range(len(phase_stamps)):
    
            save_path_ = save_path+'_%s.wav'%(i)
            # librosa.output.write_wav(save_path,phase_ad[i],fs)
            sf.write(save_path_, phase_ad[i], fs)
            smile_fea[i,:] = ffunc.get_SMILE(save_path_)
        
        ot = np.mean(smile_fea,axis=0)
        
    return ot


def feature_from_audio(audio_path:str,
                       fs:int,
                       ant_path:str,
                       save_path:str):

    audio = ffunc.load_one_ad(ad_name=audio_path, fs=fs)
    all_phase_stamps = ffunc.get_stamp(ant_path=ant_path)
    phase = ['inhale', 'cough']
    ot = np.zeros((len(phase),6373))
    for i,p in enumerate(phase):
        phase_stamps = all_phase_stamps['%s_list'%(p)]
        # print(phase_stamps)
        save_path_ = save_path+'_%s'%(p)
        ot[i,:] = feature_from_phase(audio=audio,
                                   fs=fs,
                                   phase_stamps=phase_stamps,
                                   save_path=save_path_
                                   )
    
    assert np.nan not in ot, "NaN in features"
    
    return ot
    

def feature_extraction_compare(tg_folder_path:str,
                               ad_folder_path:str,
                               save_folder_path:str,
                               fs:int,
                               split:str):
    
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    
    ot = []
    
    for file in sorted(glob.glob(os.path.join(tg_folder_path, '%s_*.TextGrid'%(split))),key=ffunc.numericalSort):
    
        subject_id = os.path.split(file)[1][:-9]
        audio_path = os.path.join(ad_folder_path,'%s.wav'%(subject_id))
        save_path = os.path.join(save_folder_path,'%s'%(subject_id))
        ot.append(feature_from_audio(audio_path=audio_path,
                                     fs=fs,
                                     ant_path=file,
                                     save_path=save_path))
    
    return ot


def feature_extraction_dicova2(tg_folder_path:str,
                               ad_folder_path:str,
                               save_folder_path:str,
                               fs:int):
    
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    
    ot = []
    
    for file in sorted(glob.glob(os.path.join(tg_folder_path, '*.TextGrid')),key=ffunc.numericalSort):

        subject_id = os.path.split(file)[1][:-9]
        audio_path = os.path.join(ad_folder_path,'%s.flac'%(subject_id))
        save_path = os.path.join(save_folder_path,'%s'%(subject_id))
        ot.append(feature_from_audio(audio_path=audio_path,
                                     fs=fs,
                                     ant_path=file,
                                     save_path=save_path))

    return ot




if __name__ == '__main__':
    
    smile_compare = feature_extraction_compare(r'C:\Users\richa\OneDrive\files\GitHub\COVID_Cough_Phases\data\ComParE_2021\annotation',
                                               r'C:\Users\richa\OneDrive\desktop\PROJECTS\COVID\ComPare\Cough\wav_new',
                                               r'C:\Users\richa\OneDrive\desktop\PROJECTS\COVID\ComPare\Cough\wav_phase',
                                               16000,
                                               'train')
    
    smile_dicova2 = feature_extraction_dicova2(r'C:\Users\richa\OneDrive\files\GitHub\COVID_Cough_Phases\data\DiCOVA2\annotation',
                                               r'C:\Users\richa\OneDrive\desktop\PROJECTS\COVID\Dicova\Second_DiCOVA_Challenge_Dev_Data_Release\Second_DiCOVA_Challenge_Dev_Data_Release\AUDIO\cough',
                                               r'C:\Users\richa\OneDrive\desktop\PROJECTS\COVID\Dicova\Second_DiCOVA_Challenge_Dev_Data_Release\Second_DiCOVA_Challenge_Dev_Data_Release\AUDIO\cough_phase',
                                               16000)
    
    # feature_path = r'C:\Users\richa\OneDrive\files\GitHub\COVID_Cough_Phases\feature'
    # with open(os.path.join(feature_path,'smile_compare.pkl'), 'wb') as handle:
    #     pkl.dump(smile_compare, handle, protocol=pkl.HIGHEST_PROTOCOL)