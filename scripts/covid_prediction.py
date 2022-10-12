# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 14:39:22 2022

@author: richa
"""

# %% Performance test
from sklearn.model_selection import train_test_split
import sklearn
import ML_func
from sklearn.decomposition import PCA
import random
import scipy.stats
import numpy as np
from sklearn import metrics


def mean_confidence_interval(data):
    m = np.mean(data)
    std = np.std(data)
    h = 2*std
    return m, m-h, m+h


def train_multi_clf(concat_feature, 
                    label, 
                    model_choice:list, 
                    clf_kwargs:object):
    
    X_train_1 = concat_feature[:,:6373]
    X_train_2 = concat_feature[:, 6373:2*6373]
    X_train_3 = concat_feature[:, 2*6373:]
    # train clf_1
    clf_1 = ML_func.train_model(X_train_1,
                                label,
                                model_choice[0],
                                clf_kwargs)
    # train clf_2
    clf_2 = ML_func.train_model(X_train_2,
                                label,
                                model_choice[1],
                                clf_kwargs)
    # train clf_3
    clf_3 = ML_func.train_model(X_train_3,
                                label,
                                model_choice[2],
                                clf_kwargs)
    
    return clf_1, clf_2, clf_3



def eva_dicova2(feature,
                label,
                train_mode:str,
                model_choice:list):
    
    num_rep = 10
    r = {'auc_list':[],'uar_list':[],'tpr_list':[],'fpr_list':[]}
    for i in range(num_rep):
        # for DiCOVA2
        x_train, x_test, y_train, y_test = train_test_split(feature, 
                                                            label, 
                                                            test_size=0.33, 
                                                            random_state=i)
        
        scaler = sklearn.preprocessing.StandardScaler()
        X_train = scaler.fit_transform(x_train)
        X_test = scaler.transform(x_test)
        # pca = PCA(n_components=100)
        # pca.fit(X_train)
        # X_train = pca.transform(X_train)
        # X_test = pca.transform(X_test)
        
        clf_kwargs = {"pds":"no",
                      "n_repeats":3,
                      "n_splits":3}
        
        if train_mode == 'single_clf':
            clf = ML_func.train_model(X_train,
                                      y_train,
                                      model_choice[0],
                                      clf_kwargs)
        
            preds = clf.predict(X_test)
            probs = clf.predict_proba(X_test)[:,1]
        
        elif train_mode == 'multi_clf':
            
            clfs = train_multi_clf(X_train,
                                   y_train,
                                   model_choice,
                                   clf_kwargs)
            
            preds = np.empty((X_test.shape[0],len(model_choice)))
            probs = np.empty((X_test.shape[0],len(model_choice)))
            
            for i in range(len(model_choice)):
                
                preds[:,i] = clfs[0].predict(X_test)
                probs[:,i] = clfs[0].predict_proba(X_test)[:,1]
            
            preds = np.mean(preds,axis=1)
            probs = np.mean(probs,axis=1)
        
        
        results = ML_func.sys_evaluate(preds,probs,y_test)
        fpr, tpr, _ = metrics.roc_curve(y_test, probs, pos_label=1)
        r['fpr_list'].append(fpr)
        r['tpr_list'].append(tpr)
        r['uar_list'].append(results['UAR'])
        r['auc_list'].append(results['ROC'])
        
    uar_final = mean_confidence_interval(r['uar_list'])
    auc_final = mean_confidence_interval(r['auc_list'])
        
    return auc_final,uar_final
        

def eva_compare(feature,
                label,
                train_mode:str,
                model_choice:list):
        
    # for ComParE
    split_index = [-1]*273 + [0]*222
    x_train = feature[:273+222,:]
    x_test = feature[495:695,:]
    y_train = label[:495]
    y_test = label[495:]
    
    scaler = sklearn.preprocessing.StandardScaler()
    X_train = scaler.fit_transform(x_train)
    X_test = scaler.transform(x_test)
    
    # pca = PCA(n_components=400)
    # pca.fit(X_train)
    # X_train = pca.transform(X_train)
    # X_test = pca.transform(X_test)
    
    clf_kwargs = {"pds":"yes",
                  "split_index":split_index}
    
    if train_mode == 'single_clf':
        clf = ML_func.train_model(X_train,
                                  y_train,
                                  model_choice[0],
                                  clf_kwargs)
        
    elif train_mode == 'multi_clf':
        
        clfs = train_multi_clf(X_train,
                               y_train,
                               model_choice,
                               clf_kwargs)
        
        preds = np.empty((X_test.shape[0],len(model_choice)))
        probs = np.empty((X_test.shape[0],len(model_choice)))
        
        for i in range(len(model_choice)):
            
            preds[:,i] = clfs[0].predict(X_test)
            probs[:,i] = clfs[0].predict_proba(X_test)[:,1]
        
        preds = np.mean(preds,axis=1)
        probs = np.mean(probs,axis=1)
    
    # Bootstrap sampling test 1000*
    bs = 1000
    r = {'auc_list':[],'uar_list':[],'tpr_list':[],'fpr_list':[]}
    for n in range(bs):
        
        idx = list(range(X_test.shape[0]))
        bs_idx = random.choices(idx,k=X_test.shape[0])
        X_test_new = X_test[bs_idx,:]
        y_test_new = y_test[bs_idx]
        
        if train_mode == 'single_clf':
            
            preds = clf.predict(X_test_new)
            probs = clf.predict_proba(X_test_new)[:,1]
        
        elif train_mode == 'multi_clf':
            
            preds = np.empty((X_test.shape[0],len(model_choice)))
            probs = np.empty((X_test.shape[0],len(model_choice)))
        
            for i in range(len(model_choice)):
                preds[:,i] = clfs[0].predict(X_test)
                probs[:,i] = clfs[0].predict_proba(X_test)[:,1]
        
            preds = np.mean(preds,axis=1)
            probs = np.mean(probs,axis=1)
        
        results = ML_func.sys_evaluate(preds,probs,y_test_new)
        fpr, tpr, _ = metrics.roc_curve(y_test_new, probs, pos_label=1)
        r['fpr_list'].append(fpr)
        r['tpr_list'].append(tpr)
        r['uar_list'].append(results['UAR'])
        r['auc_list'].append(results['ROC'])
    
    uar_final = mean_confidence_interval(r['uar_list'])
    auc_final = mean_confidence_interval(r['auc_list'])
    
    return auc_final,uar_final


if __name__ == "__main__":
    
    auc_final,_ = eva_compare(baseline_c,label_c,'multi_clf',['svm','svm','rf'])
    auc_final,_ = eva_dicova2(fuse_d[:,:-13],label_d,'svm')
