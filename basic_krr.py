# initialization and random seed set

import os
import scipy
import scipy.io
import torch
import random
import math
import sklearn
from utils import *
import numpy as np
import pandas as pd
import torch.nn as nn
from collections import Counter
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold
from CBIG_model_pytorch import dnn_4l, dnn_5l
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.patches as mpatches
import warnings
import joblib
warnings.filterwarnings("ignore")




### BASIC KRR 관련한 함수 START ### 
def basic_krr(df, pheno_with_age, k_num, data_file_name, iteration=10, only_test=False):
    test_size=0.2
    corrs = []
    cods = []
    best_node = []

    folder_path = f"D:/meta_matching_data/model_pth/{data_file_name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"{folder_path}가 생성되었습니다!")


    # Training
    for seed in range(1, iteration+1):
        model_pth = f"D:/meta_matching_data/model_pth/{data_file_name}/{seed}_krr_{data_file_name}.pkl"
        print(f'==========================================Iter{seed}==========================================')
        # Random Seed Setting
        set_random_seeds(seed)
        # Train / Val / Kshot/ Test
        train_df, train_pheno, val_df, val_pheno, _, _, _, _ = \
                                    preprocess_data(df, pheno_with_age, test_size, k_num, seed)
        _, _, _, _, kshot_df, kshot_pheno, test_df, test_pheno= preprocess_data(df, pheno_with_age, test_size, k_num, seed)
        train_df = np.concatenate((train_df, val_df), axis=0)
        train_pheno = np.concatenate((train_pheno, train_pheno), axis=0)
        train_age = train_pheno[:, -1]
        train_pheno = train_pheno[:, :-1]
        kshot_age = kshot_pheno[:, -1]#10,1
        kshot_pheno = kshot_pheno[:, :-1]#10,58
        test_age = test_pheno[:, -1]#140,58
        test_pheno = test_pheno[:, :-1]#
        kf = KFold(n_splits=5) # K-fold 설정
        r2_scores = []
        krr_model = KernelRidge()
        alphas = [0.1, 0.7, 1, 5, 10]
        param_grid = {'alpha':alphas}
        krr_models = []
        # K-fold 로 Iteration 돈다.
        for train_index, test_index in kf.split(train_df):
            train_df_fold, val_df_fold = train_df[train_index], train_df[test_index]
            train_pheno_fold, val_pheno_fold = train_pheno[train_index], train_pheno[test_index]
            for i in range(train_pheno_fold.shape[1]):
                krr_model = KernelRidge()
                grid_search = GridSearchCV(krr_model, param_grid, cv = kf)
                grid_search.fit(train_df_fold, train_pheno_fold[:, i])
                best_alpha = grid_search.best_params_['alpha']
                best_krr = KernelRidge(alpha = best_alpha)
                best_krr.fit(train_df_fold, train_pheno_fold[:, i])
                krr_models.append(best_krr) # 모델을 List에 넣는다...?
        # 58개의 phenotype에 대해서 5 fold했고
        # 5개의 fold 중 max cod를 갖는 fold의 krr model만 추출
        max_cod_model = []
        for i in range(58):
            subset = krr_models[i::58]
            r2_scores_kshot = []
            for i in range(len(subset)):
                pheno_pred = subset[i].predict(kshot_df)
                # 여기에서는 Age가 아니라 각 phenotype이지만, 일단은 age라고 해야 한다.
                pheno_df = pd.DataFrame({'prediction' : pheno_pred, 'Age' : kshot_pheno[:, i]})
                # r2 = get_cod_score(kshot_pheno[:, i], pheno_pred)

                r2 = get_cod_score(pheno_df)
                #r2 = get_corr_score(pheno_df)
                r2_scores_kshot.append(r2)
            max_cod_model.append(r2_scores_kshot.index(max(r2_scores_kshot)))
        max_cod_krr_models = []
        for i in range(len(max_cod_model)):
            max_cod_krr_models.append(krr_models[i*5+max_cod_model[i]])
        # best cod 추출된 krr model로 k shot age cod 계산 후 best krr node 추출
        kshot_cods = []
        
        for i in range(len(max_cod_krr_models)):
            k_pred = max_cod_krr_models[i].predict(kshot_df)
            k_pred_df = pd.DataFrame({'prediction' : k_pred, 'Age':kshot_age})
            k_age_cod = get_cod_score(k_pred_df)
            #k_age_cod = get_corr_score(k_pred_df)
            kshot_cods.append(k_age_cod)
        print('max cod index:', kshot_cods.index(max(kshot_cods)))
        max_kshot_cod_idx = kshot_cods.index(max(kshot_cods))
        best_node.append(max_kshot_cod_idx)
        # best krr model로 test
        test_pred = max_cod_krr_models[max_kshot_cod_idx].predict(test_df)
        joblib.dump(max_cod_krr_models[max_kshot_cod_idx], model_pth)
        test_pred_df = pd.DataFrame({'prediction' : test_pred, 'Age' : test_age})
        test_age_cod = get_cod_score(test_pred_df)
        test_age_corr = get_corr_score(test_pred_df)
        corrs.append(test_age_corr)
        cods.append(test_age_cod)
        print(f'cod: {test_age_cod:.4f}')
        print(f'corr: {test_age_corr:.4f}')

        ##model load test
        #loaded_model = joblib.load(model_pth)
        #test_pred = loaded_model.predict(test_df)
        #test_pred_df = pd.DataFrame({'prediction' : test_pred, 'Age' : test_age})
        #test_age_cod = get_cod_score(test_pred_df)
        #test_age_corr = get_corr_score(test_pred_df)       
        #print(f'load cod: {test_age_cod:.4f}')
        #print(f'load corr: {test_age_corr:.4f}')

    print('==========================================학습을 완료하였습니다.==========================================')
    print('\n\n')
    print(f"Average COD : {np.mean(cods):.4f}")
    print(f"STD     COD : {np.std(cods):.4f}")
    print(f"Average Corr : {np.mean(corrs):.4f}")
    print(f"STD     Corr : {np.std(corrs):.4f}")
    return cods, corrs, best_node
   
    
# basic_krr(df, pheno_with_age, k_num=10, batch_size=128, iteration=10)




### BASIC KRR END ### 
