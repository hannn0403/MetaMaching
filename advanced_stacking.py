# initialization and random seed set

import os
import scipy
import scipy.io
import torch
import random
import math
import sklearn
import numpy as np
import pandas as pd
from utils import *
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
warnings.filterwarnings("ignore")


### STACKING START ### 
def get_top_k_indices(kshot_iq_cods, k_num):
    # kshot_iq_cods의 각 요소와 인덱스를 묶어서 리스트로 만듭니다.
    elements_with_indices = list(enumerate(kshot_iq_cods))    
    # 크기를 기준으로 내림차순으로 정렬합니다.
    sorted_elements = sorted(elements_with_indices, key=lambda x: x[1], reverse=True)
    # 상위 k_num개의 요소의 인덱스를 추출합니다.
    top_k_indices = [index for index, value in sorted_elements[:k_num]]
    
    return top_k_indices


def get_elements_by_indices(input_list, indices):
    return [input_list[i] for i in indices]


def get_predict_df(model, df, pheno_with_iq, k_num, generator, seed):
    '''
    K Sample 들을 이용하여, M개의 가장 좋은 예측을 수행하는 Node를 선정하고, 이 노드에 대한 예측 값들만으로 구성한 
    데이터 프레임 반환 (pred_df)
    '''
    device = 'cuda'
    # k_num에 따른 Stacking에 몇가지 Phenotype을 쓸 것인지 선정 
    m = min(58, k_num)

    # Data Loader & Model Initialization 
    _, _, _, _, kshot_df, kshot_pheno, test_df, test_pheno= preprocess_data(df, pheno_with_iq, 0.2, k_num, seed)
    model = model.to(device)
    kshot_iq = kshot_pheno[:, -1]
    kshot_pheno = kshot_pheno[:, :-1]
    test_iq = test_pheno[:, -1]
    test_pheno = test_pheno[:, :-1]
    
    kshot_dataloader = get_dataloader(kshot_df, kshot_pheno, k_num, generator, device)
    test_dataloader = get_dataloader(test_df, test_pheno, test_df.shape[0], generator, device)
    
    # K-sample들에 대한 Prediction을 수행 & 결과 저장 
    model.eval()
    kshot_outputs = []
    with torch.no_grad(): 
        for inputs, targets in kshot_dataloader:
            output = model(inputs) 
            kshot_outputs.append(output)
    kshot_outputs = torch.cat(kshot_outputs).cpu().detach().numpy().T

    # Node의 Corr을 저장하기 위한 `kshot_iq_corrs` 리스트 생성 
    kshot_iq_corrs = []
    kshot_iq_cods = []
    
    kshot_pred_df = pd.DataFrame()
    for i in range(len(kshot_outputs)):
        if i == 0:
            kshot_pred_df = pd.DataFrame({'IQ':kshot_iq, f'prediction_{i}':kshot_outputs[i]})
        else: 
            kshot_pred_df[f"prediction_{i}"]=kshot_outputs[i]
        pred = pd.DataFrame({'prediction':kshot_outputs[i], 'IQ':kshot_iq})
        kshot_iq_cods.append(get_cod_score(pred)) # 각 phenotype을 예측하여, 예측 값과 COD를 계산
        kshot_iq_corrs.append(get_corr_score(pred)) # 각 phenotype을 예측하여, 예측 값과 Corr를 계산

    
    top_k_indices = get_top_k_indices(kshot_iq_cods, m)
    column_list = [f"prediction_{i}" for i in top_k_indices]
    column_list.append('IQ')    
        
    
    # Test set에 대한 
    model.eval()
    test_outputs = []
    with torch.no_grad(): 
        for inputs, targets in test_dataloader: 
            output = model(inputs)
            test_outputs.append(output) 
        test_outputs = torch.cat(test_outputs).cpu().detach().numpy().T
        
        
    test_pred_df = pd.DataFrame()
    for i in range(len(test_outputs)):
        if i == 0: 
            test_pred_df = pd.DataFrame({'IQ': test_iq, f"prediction_{i}":test_outputs[i]})
        else: 
            test_pred_df[f"prediction_{i}"]=test_outputs[i]
    
    kshot_pred_df = kshot_pred_df.loc[:, column_list]
    test_pred_df = test_pred_df.loc[:, column_list]
    
    return kshot_pred_df, test_pred_df, top_k_indices, kshot_iq_corrs
    
    
def advanced_stacking(df, pheno_with_iq, k_num_list, data_file_name, batch_size=128, iteration=10):

    # 예측 성능을 기록할 Dictionary Intialization 
    corr_dict = {'10':[], '30':[], '50':[], '100':[]} 
    cod_dict = {'10':[], '30':[], '50':[], '100':[]} 
    best_node_dict = {'10':[], '30':[], '50':[], '100':[]} 

    iter_corr_dict = {}

    for seed in range(1, iteration+1):
        basic_model_pth = f'D:/meta_matching_data/model_pth/{data_file_name}/{seed}_dnn4l_adamw_{data_file_name}.pth'
        set_random_seeds(seed)
        generator = torch.Generator()
        generator.manual_seed(seed) 
        basic_model = torch.load(basic_model_pth)

        for k_num in k_num_list: 
            print(f"==========================================K : {k_num}==========================================")
            # Secondary Model(KRR)에 들어갈 INPUT (K-shot & test) 
            kshot_pred_df, test_pred_df, top_k_indices, kshot_iq_corrs = get_predict_df(basic_model, df, pheno_with_iq, k_num, generator, seed) 

            # 각 Iteration의 각 Node에 대한 correlation list 
            iter_corr_dict[f'iter_{seed}_k_{k_num}'] = kshot_iq_corrs
            
            # Hyperparam Tuning 
            kshot_x = kshot_pred_df.drop('IQ', axis=1)
            kshot_y = kshot_pred_df['IQ']
            
            krr = KernelRidge(kernel='rbf')
            alphas = [0.1, 0.7, 1, 5, 10]
            param_grid = {'alpha':alphas}
            kf = KFold(n_splits=5)
            grid_search = GridSearchCV(krr, param_grid, cv=kf)
            grid_search.fit(kshot_x, kshot_y)

            best_alpha = grid_search.best_params_['alpha']
            best_krr = KernelRidge(kernel='rbf', alpha=best_alpha)  # 최적의 alpha를 가진 KRR 선언 
            best_krr.fit(kshot_x, kshot_y)

            # Testing 
            test_x = test_pred_df.drop('IQ', axis=1)
            test_y = test_pred_df['IQ']
            
            # prediction
            stacking_pred = best_krr.predict(test_x)

            final_pred_df = pd.DataFrame({'prediction':stacking_pred, 'IQ':test_y})
            iq_corr = get_corr_score(final_pred_df)
            iq_cod = get_cod_score(final_pred_df)

            # Prediction의 COD와 Corr에 대한 결과 리스트 정리

            corr_dict[str(k_num)].append(iq_corr) 
            cod_dict[str(k_num)].append(iq_cod)
            best_node_dict[str(k_num)].append(top_k_indices)               
            print(f"Iteration {seed}, K : {k_num} - Correlation :{iq_corr:.4f}".rjust(50))
            print(f"R2 Score :{iq_cod:.4f}".rjust(50))
        print('\n\n\n')
    
    iter_corr_df = pd.DataFrame(iter_corr_dict) 

    for key in k_num_list: 
        if len(cod_dict[str(key)]) != 0: 
            print(f"K={key} : Average COD : {np.mean(cod_dict[str(key)])}")
            print(f"K={key} : STD     COD : {np.std(cod_dict[str(key)])}")
            print()
            print(f"K={key} : Average Corr : {np.mean(corr_dict[str(key)])}")
            print(f"K={key} : STD     Corr : {np.std(corr_dict[str(key)])}") 
    
    return corr_dict, cod_dict, best_node_dict, iter_corr_df
    
    
# advanced_stacking(df, pheno_with_iq, [10, 30, 50, 100], batch_size=128, iteration=100)


### STACKING END ### 