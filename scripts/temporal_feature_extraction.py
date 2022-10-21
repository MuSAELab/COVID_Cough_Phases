# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 17:49:56 2021

@author: richa
"""


"""
Sanity check of annotations
"""
import textgrid
import os,re,glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy
import load_data as ld

# %% Import annotations of DiCOVA2
ant_path_1 = r'C:\Users\richa\OneDrive\files\GitHub\COVID_Cough_Phases\data\DiCOVA2\annotation'
label_path_1 = r'C:\Users\richa\OneDrive\files\GitHub\COVID_Cough_Phases\data\DiCOVA2\label\metadata.csv'

tg_d,_,label_d = ld.load_data_dicova(label_path_1,ant_path_1)

# %% Import annotations of ComParE
ant_path_2 = r'C:\Users\richa\OneDrive\files\GitHub\COVID_Cough_Phases\data\ComParE_2021\annotation'
label_path_2 = r'C:\Users\richa\OneDrive\files\GitHub\COVID_Cough_Phases\data\ComParE_2021\label'

tg_train,label_train = ld.load_data_compare(label_path_2,
                                            ant_path_2,
                                            'train')
tg_devel,label_devel = ld.load_data_compare(label_path_2,
                                            ant_path_2,
                                            'devel')
tg_test,label_test = ld.load_data_compare(label_path_2,
                                          ant_path_2,
                                          'test')
tg_c = tg_train+tg_devel+tg_test
label_c = np.concatenate((label_train,label_devel,label_test),axis=0)

tg = tg_c+tg_d
label = np.concatenate((label_c,label_d),axis=0)

# %% Check number of inhalation, compression, and cough
def check_num(tg):
    
    num_cough = 0
    num_inh = 0
    num_other = 0

    for i in range(0,len(tg)):

        num_inh += int(tg[i].tiers[2].intervals[0].mark)
        num_cough = int(tg[i].tiers[1].intervals[0].mark)
        n_seg = len(tg[i].tiers[0].intervals)
        
        for j in range(0,n_seg):
            seg = tg[i].tiers[0].intervals
            if seg[j].mark == 'other':
                num_other += 1
            if seg[j].mark == 'throatclear':
                num_other += 1
        
    return num_cough,num_inh,num_other

# %% Check the consistency of cough annotations
def check_consistency(tg):
    
    num_cough = np.zeros((1,len(tg)))
    num_inh = np.zeros((1,len(tg)))

    for i in range(0,len(tg)):

        n_inhale = int(tg[i].tiers[2].intervals[0].mark)
        n_cough = int(tg[i].tiers[1].intervals[0].mark)
        n_seg = len(tg[i].tiers[0].intervals)
        
        num_1 = 0
        num_2 = 0
        for j in range(0,n_seg):
            seg = tg[i].tiers[0].intervals
            if seg[j].mark == 'cough':
                num_1 += 1
            if seg[j].mark == 'inhale':
                num_2 += 1
        
        if num_1 != n_cough: num_cough[0,i] = 1
        if num_2 != n_inhale: num_inh[0,i] = 1
        if num_1>30:
            print(i)
        
    return num_cough,num_inh

# Check the number of inconsistent annotations
num_cough,num_inh = check_consistency(tg)
print(np.where(num_cough == 1)[1])
print(np.where(num_inh == 1)[1])

# %% Statistical analysis on number of events
def count_num(tg):
    
    stat_cough = np.zeros((4,len(tg)))

    for i in range(0,len(tg)):
        ad_len = tg[i].maxTime
        n_inhale = int(tg[i].tiers[2].intervals[0].mark)
        n_cough = int(tg[i].tiers[1].intervals[0].mark)
        n_seg = len(tg[i].tiers[0].intervals)

        num = 0
        for j in range(0,n_seg):
            seg = tg[i].tiers[0].intervals
            if seg[j].mark == 'throatclear' or seg[j].mark == 'other':
                num += 1
        
        stat_cough[2,i] = num/ad_len
        stat_cough[1,i] = n_inhale/ad_len
        stat_cough[0,i] = n_cough/ad_len
        if (n_inhale != 0) and (n_cough != 0):
            stat_cough[3,i] = n_inhale/n_cough
        
    return stat_cough.T

stat_cough_fea = count_num(tg)
stat_cough = pd.DataFrame(stat_cough_fea)
stat_cough.columns = ['num_cough/rec','num_inh/rec','num_other/rec','inh_cough_ratio']
stat_cough['label'] = label

pos_idx = stat_cough.index[stat_cough['label'] == 1].tolist()
neg_idx = stat_cough.index[stat_cough['label'] == 0].tolist()

""" number of cough per second per recording """
plt.figure(dpi=300)
sns.histplot(stat_cough.iloc[pos_idx],
             x='num_cough/rec',
             stat='density',
             color = 'red',
             kde = True)

sns.histplot(stat_cough.iloc[neg_idx],
             x='num_cough/rec',
             stat='density',
             color = 'blue',
             kde = True)
plt.legend(labels=['Positive','Negative'])

scipy.stats.ttest_ind(stat_cough.iloc[pos_idx]['num_cough/rec'],
                      stat_cough.iloc[neg_idx]['num_cough/rec'],
                      equal_var=False)


""" number of inhalation per second per recording """
plt.figure(dpi=300)
sns.histplot(stat_cough.iloc[pos_idx],
             x='num_inh/rec',
             stat='density',
             color = 'red',
             kde = True)

sns.histplot(stat_cough.iloc[neg_idx],
             x='num_inh/rec',
             stat='density',
             color = 'blue',
             kde = True)
plt.legend(labels=['Positive','Negative'])

scipy.stats.ttest_ind(stat_cough.iloc[pos_idx]['num_inh/rec'],
                      stat_cough.iloc[neg_idx]['num_inh/rec'],
                      equal_var=False)


""" number of articulatory sounds per second per recording """
plt.figure(dpi=300)
sns.histplot(stat_cough.iloc[pos_idx],
             x='num_other/rec',
             stat='density',
             color = 'red',
             kde = True)

sns.histplot(stat_cough.iloc[neg_idx],
             x='num_other/rec',
             stat='density',
             color = 'blue',
             kde = True)
plt.legend(labels=['Positive','Negative'])

scipy.stats.ttest_ind(stat_cough.iloc[pos_idx]['num_other/rec'],
                      stat_cough.iloc[neg_idx]['num_other/rec'],
                      equal_var=False)


""" cough/inhalation ratio per recording """
plt.figure(dpi=300)
sns.histplot(stat_cough[(stat_cough['label']==1) & \
                        (stat_cough['inh_cough_ratio']!=0)],
             x='inh_cough_ratio',
             stat='density',
             color = 'red',
             log_scale=True,
             kde = True)

sns.histplot(stat_cough[(stat_cough['label']==0) & \
                        (stat_cough['inh_cough_ratio']!=0)],
             x='inh_cough_ratio',
             stat='density',
             color = 'blue',
             log_scale=True,
             kde = True)
plt.legend(labels=['Positive','Negative'])

scipy.stats.ttest_ind(stat_cough[(stat_cough['label']==1) & \
                        (stat_cough['inh_cough_ratio']!=0)]['inh_cough_ratio'],
                      stat_cough[(stat_cough['label']==0) & \
                        (stat_cough['inh_cough_ratio']!=0)]['inh_cough_ratio'],
                      equal_var=False)

# %%
def segment_duration_lld(tg):
    
    ot = np.ones((len(tg),9))*1e-5

    for i in range(0,len(tg)):
        n_seg = len(tg[i].tiers[0].intervals)
        cough_fea = []
        inh_fea = []
        com_fea = []
        for j in range(0,n_seg):

            current_seg = tg[i].tiers[0].intervals[j]
            seg_duration = current_seg.maxTime - current_seg.minTime
            
            if current_seg.mark == 'cough':
                cough_fea.append(seg_duration)
            elif current_seg.mark == 'inhale':
                inh_fea.append(seg_duration)
            elif current_seg.mark == 'compress':
                com_fea.append(seg_duration)
        
        if len(cough_fea) != 0:
            ot[i,0] = np.mean(cough_fea)
            ot[i,1] = np.std(cough_fea)
        
        if len(inh_fea) != 0:
            ot[i,2] = np.mean(inh_fea)
            ot[i,3] = np.std(inh_fea)

        if len(com_fea) != 0:
            ot[i,4] = np.mean(com_fea)
            ot[i,5] = np.std(com_fea)
        
        if (len(cough_fea) != 0) and (len(inh_fea) != 0):
            ot[i,6] = ot[i,2]/ot[i,0]
            ot[i,7] = ot[i,2]/ot[i,4]
            ot[i,8] = ot[i,4]/ot[i,0]
    
    return ot

lld = segment_duration_lld(tg)
lld_df = pd.DataFrame(lld)
lld_df.columns = ['ave_cough_dur','std_cough_dur',
               'ave_inh_dur','std_inh_dur',
               'ave_com_dur','std_com_dur',
               'ave_dur_inh_cough_ratio',
               'ave_dur_inh_com_ratio',
               'ave_dur_com_cough_ratio']
lld_df['label'] = label

pos_idx = lld_df.index[lld_df['label'] == 1].tolist()
neg_idx = lld_df.index[lld_df['label'] == 0].tolist()

""" cough dur """
plt.figure(dpi=300)
sns.histplot(lld_df.iloc[pos_idx],
             x='ave_cough_dur',
             stat='density',
             color = 'red',
             kde = True)

sns.histplot(lld_df.iloc[neg_idx],
             x='ave_cough_dur',
             stat='density',
             color = 'blue',
             kde = True)
plt.legend(labels=['Positive','Negative'])

scipy.stats.ttest_ind(lld_df.iloc[pos_idx]['ave_cough_dur'],
                      lld_df.iloc[neg_idx]['ave_cough_dur'],
                      equal_var=False)

plt.figure(dpi=300)
sns.histplot(lld_df.iloc[pos_idx],
             x='std_cough_dur',
             stat='density',
             color = 'red',
             kde = True)

sns.histplot(lld_df.iloc[neg_idx],
             x='std_cough_dur',
             stat='density',
             color = 'blue',
             kde = True)
plt.legend(labels=['Positive','Negative'])

scipy.stats.ttest_ind(lld_df.iloc[pos_idx]['std_cough_dur'],
                      lld_df.iloc[neg_idx]['std_cough_dur'],
                      equal_var=False)

""" inh dur """
plt.figure(dpi=300)
sns.histplot(lld_df[(lld_df['label']==1) & (lld_df['ave_inh_dur']!=1e-5)],
             x='ave_inh_dur',
             stat='density',
             color = 'red',
             kde = True)

sns.histplot(lld_df[(lld_df['label']==0) & (lld_df['ave_inh_dur']!=1e-5)],
             x='ave_inh_dur',
             stat='density',
             color = 'blue',
             kde = True)
plt.legend(labels=['Positive','Negative'])

scipy.stats.ttest_ind(lld_df[(lld_df['label']==1) & (lld_df['ave_inh_dur']!=1e-5)]['ave_inh_dur'],
                      lld_df[(lld_df['label']==0) & (lld_df['ave_inh_dur']!=1e-5)]['ave_inh_dur'],
                      equal_var=False)

plt.figure(dpi=300)
sns.histplot(lld_df[(lld_df['label']==1) & (lld_df['std_inh_dur']!=1e-5)],
             x='std_inh_dur',
             stat='density',
             color = 'red',
             kde = True)

sns.histplot(lld_df[(lld_df['label']==0) & (lld_df['std_inh_dur']!=1e-5)],
             x='std_inh_dur',
             stat='density',
             color = 'blue',
             kde = True)
plt.legend(labels=['Positive','Negative'])

scipy.stats.ttest_ind(lld_df[(lld_df['label']==1) & (lld_df['std_inh_dur']!=1e-5)]['std_inh_dur'],
                      lld_df[(lld_df['label']==0) & (lld_df['std_inh_dur']!=1e-5)]['std_inh_dur'],
                      equal_var=False)

""" compression duration """
plt.figure(dpi=300)
sns.histplot(lld_df[(lld_df['label']==1) & (lld_df['ave_com_dur']!=1e-5)],
             x='ave_com_dur',
             stat='density',
             color = 'red',
             kde = True)

sns.histplot(lld_df[(lld_df['label']==0) & (lld_df['ave_com_dur']!=1e-5)],
             x='ave_com_dur',
             stat='density',
             color = 'blue',
             kde = True)
plt.legend(labels=['Positive','Negative'])

scipy.stats.ttest_ind(lld_df[(lld_df['label']==1) & (lld_df['ave_com_dur']!=1e-5)]['ave_com_dur'],
                      lld_df[(lld_df['label']==0) & (lld_df['ave_com_dur']!=1e-5)]['ave_com_dur'],
                      equal_var=False)

plt.figure(dpi=300)
sns.histplot(lld_df.iloc[pos_idx],
             x='std_com_dur',
             stat='density',
             color = 'red',
             kde = True)

sns.histplot(lld_df.iloc[neg_idx],
             x='std_com_dur',
             stat='density',
             color = 'blue',
             kde = True)
plt.legend(labels=['Positive','Negative'])

scipy.stats.ttest_ind(lld_df.iloc[pos_idx]['std_com_dur'],
                      lld_df.iloc[neg_idx]['std_com_dur'],
                      equal_var=False)

""" cross-event duration features """
plt.figure(dpi=300)
sns.histplot(lld_df[(lld_df['label']==1) & (lld_df['ave_dur_inh_cough_ratio']!=1e-5)],
             x='ave_dur_inh_cough_ratio',
             stat='density',
             color = 'red',
             kde = True)

sns.histplot(lld_df[(lld_df['label']==0) & (lld_df['ave_dur_inh_cough_ratio']!=1e-5)],
             x='ave_dur_inh_cough_ratio',
             stat='density',
             color = 'blue',
             kde = True)
plt.legend(labels=['Positive','Negative'])

scipy.stats.ttest_ind(lld_df[(lld_df['label']==1) & \
                             (lld_df['ave_dur_inh_cough_ratio']!=1e-5)]['ave_dur_inh_cough_ratio'],
                      lld_df[(lld_df['label']==0) & \
                             (lld_df['ave_dur_inh_cough_ratio']!=1e-5)]['ave_dur_inh_cough_ratio'],
                      equal_var=False)
    

plt.figure(dpi=300)
sns.histplot(lld_df[(lld_df['label']==1) & (lld_df['ave_dur_inh_com_ratio']!=1e-5)],
             x='ave_dur_inh_com_ratio',
             stat='density',
             color = 'red',
             kde = True)

sns.histplot(lld_df[(lld_df['label']==0) & (lld_df['ave_dur_inh_com_ratio']!=1e-5)],
             x='ave_dur_inh_com_ratio',
             stat='density',
             color = 'blue',
             kde = True)
plt.legend(labels=['Positive','Negative'])

scipy.stats.ttest_ind(lld_df[(lld_df['label']==1) & \
                             (lld_df['ave_dur_inh_com_ratio']!=1e-5)]['ave_dur_inh_com_ratio'],
                      lld_df[(lld_df['label']==0) & \
                             (lld_df['ave_dur_inh_com_ratio']!=1e-5)]['ave_dur_inh_com_ratio'],
                      equal_var=False)
    

plt.figure(dpi=300)
sns.histplot(lld_df[(lld_df['label']==1) & (lld_df['ave_dur_com_cough_ratio']!=1e-5)],
             x='ave_dur_com_cough_ratio',
             stat='density',
             color = 'red',
             kde = True)

sns.histplot(lld_df[(lld_df['label']==0) & (lld_df['ave_dur_com_cough_ratio']!=1e-5)],
             x='ave_dur_com_cough_ratio',
             stat='density',
             color = 'blue',
             kde = True)
plt.legend(labels=['Positive','Negative'])

scipy.stats.ttest_ind(lld_df[(lld_df['label']==1) & \
                             (lld_df['ave_dur_com_cough_ratio']!=1e-5)]['ave_dur_com_cough_ratio'],
                      lld_df[(lld_df['label']==0) & \
                             (lld_df['ave_dur_com_cough_ratio']!=1e-5)]['ave_dur_com_cough_ratio'],
                      equal_var=False)
    
# %% Try to project features to 3d space for separability
from sklearn.manifold import TSNE

lld_all = np.concatenate((stat_cough_fea,lld),axis=1)

tsne_lld = TSNE(n_components=3, 
                verbose=1, 
                perplexity=50,
                init='random').fit_transform(lld_all)  

tsne_df = pd.DataFrame()
tsne_df['tsne-3d-one'] = tsne_lld[:,0]
tsne_df['tsne-3d-two'] = tsne_lld[:,1]
tsne_df['tsne-3d-three'] = tsne_lld[:,2]
tsne_df['label'] = label

plt.figure(figsize=(12,10))
axes = plt.axes(projection='3d')

for s in tsne_df['label'].unique():
    axes.scatter3D(tsne_df['tsne-3d-one'][tsne_df['label']==s], 
                   tsne_df['tsne-3d-two'][tsne_df['label']==s],
                   tsne_df['tsne-3d-three'][tsne_df['label']==s],
                   label=s
                   )
# axes.view_init(-140, 60)
plt.show()

# %% save features
# ld.save_data(r'C:\Users\richa\OneDrive\files\GitHub\COVID_Cough_Phases\feature','temporal_c',lld_c)
# ld.save_data(r'C:\Users\richa\OneDrive\files\GitHub\COVID_Cough_Phases\feature','temporal_d',lld_d)

# %% check noise and silence duration

def check_noise(tg):
    
    ratio = np.zeros((len(tg),3))
    
    for i in range(0,len(tg)):
        ad_len = tg[i].maxTime
        n_seg = len(tg[i].tiers[0].intervals)

        noise_len = 0
        silence_len = 0
        for j in range(0,n_seg):
            seg = tg[i].tiers[0].intervals
            if seg[j].mark == 'noise':
                noise_len += seg[j].maxTime - seg[j].minTime
            
            if seg[j].mark == 'silence':
                silence_len += seg[j].maxTime - seg[j].minTime
    
        ratio[i,0] = noise_len/ad_len
        ratio[i,1] = silence_len/ad_len
        ratio[i,2] = (noise_len+silence_len)/ad_len
    
    return ratio

ratio_c = check_noise(tg_c)
ratio_d = check_noise(tg_d)
ratio = np.concatenate((ratio_c,ratio_d),axis=0)

ratio_c = pd.DataFrame(ratio_c)
ratio_c.columns = ['ratio_noise','ratio_silence','ratio_noise_silence','label']
ratio_c['label'] = label_c

ratio_d = pd.DataFrame(ratio_d)
ratio_d['label'] = label_d
ratio_d.columns = ['ratio_noise','ratio_silence','ratio_noise_silence','label']

ratio = pd.DataFrame(ratio)
ratio['label'] = label
ratio.columns = ['ratio_noise','ratio_silence','ratio_noise_silence','label']


plt.figure(dpi=300)
sns.histplot(ratio[ratio['label']==1],
             x='ratio_noise_silence',
             stat='density',
             color = 'red',
             kde = True)

sns.histplot(ratio[ratio['label']==0],
             x='ratio_noise_silence',
             stat='density',
             color = 'blue',
             kde = True)
plt.legend(labels=['Positive','Negative'])

scipy.stats.ttest_ind(ratio[ratio['label']==1]['ratio_noise_silence'],
                      ratio[ratio['label']==0]['ratio_noise_silence'],
                      equal_var=False)