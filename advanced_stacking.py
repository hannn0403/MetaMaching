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
def get_top_k_indices(kshot_age_cods, k_num):
    # kshot_age_cods의 각 요소와 인덱스를 묶어서 리스트로 만듭니다.
    elements_with_indices = list(enumerate(kshot_age_cods))    
    # 크기를 기준으로 내림차순으로 정렬합니다.
    sorted_elements = sorted(elements_with_indices, key=lambda x: x[1], reverse=True)
    # 상위 k_num개의 요소의 인덱스를 추출합니다.
    top_k_indices = [index for index, value in sorted_elements[:k_num]]
    
    return top_k_indices


def get_elements_by_indices(input_list, indices):
    return [input_list[i] for i in indices]

def get_predict_df(model, df, pheno_with_age, k_num, generator, seed):
    '''
    K Sample 들을 이용하여, M개의 가장 좋은 예측을 수행하는 Node를 선정하고, 이 노드에 대한 예측 값들만으로 구성한 
    데이터 프레임 반환 (pred_df)
    '''
    device = 'cuda'
    m = min(58, k_num)
    _, _, _, _, kshot_df, kshot_pheno, test_df, test_pheno= preprocess_data(df, pheno_with_age, 0.2, k_num, seed)
    model = model.to(device)
    kshot_age = kshot_pheno[:, -1]
    kshot_pheno = kshot_pheno[:, :-1]
    test_age = test_pheno[:, -1]
    test_pheno = test_pheno[:, :-1]
    
    kshot_dataloader = get_dataloader(kshot_df, kshot_pheno, k_num, generator, device)
    test_dataloader = get_dataloader(test_df, test_pheno, test_df.shape[0], generator, device)
    
    
    model.eval()
    kshot_outputs = []
    with torch.no_grad(): 
        for inputs, targets in kshot_dataloader:
            output = model(inputs) 
            kshot_outputs.append(output)
    kshot_outputs = torch.cat(kshot_outputs).cpu().detach().numpy().T
    kshot_age_cods = []
    
    kshot_pred_df = pd.DataFrame()
    for i in range(len(kshot_outputs)):
        if i == 0:
            kshot_pred_df = pd.DataFrame({'Age':kshot_age, f'prediction_{i}':kshot_outputs[i]})
        else: 
            kshot_pred_df[f"prediction_{i}"]=kshot_outputs[i]
        pred = pd.DataFrame({'prediction':kshot_outputs[i], 'Age':kshot_age})
        kshot_age_cods.append(get_cod_score(pred)) # 각 phenotype을 예측하여, 예측 값과 COD를 계산
    
    top_k_indices = get_top_k_indices(kshot_age_cods, m)
    column_list = [f"prediction_{i}" for i in top_k_indices]
    column_list.append('Age')    
        
    
    
    model.eval()
    test_outputs = []
    with torch.no_grad(): 
        for inputs, targets in test_dataloader: 
            output = model(inputs)
            test_outputs.append(output) 
        test_outputs = torch.cat(test_outputs).cpu().detach().numpy().T
        test_age_cods = []
        
        
    test_pred_df = pd.DataFrame()
    for i in range(len(test_outputs)):
        if i == 0: 
            test_pred_df = pd.DataFrame({'Age': test_age, f"prediction_{i}":test_outputs[i]})
        else: 
            test_pred_df[f"prediction_{i}"]=test_outputs[i]
        
    
    
    kshot_pred_df = kshot_pred_df.loc[:, column_list]
    test_pred_df = test_pred_df.loc[:, column_list]
    
    return kshot_pred_df, test_pred_df, top_k_indices
    
    
def advanced_stacking(df, pheno_with_age, k_num_list, data_file_name, batch_size=128, iteration=10):
    device='cuda'
    
    age_corr_10 = []
    age_cod_10 = []
    age_corr_30 = []
    age_cod_30 = []
    age_corr_50 = []
    age_cod_50 = []
    age_corr_100 = []
    age_cod_100 = []
    best_node_10 = []
    best_node_30 = []
    best_node_50 = []
    best_node_100 = []

    for seed in range(1, iteration+1):
        basic_model_pth = f'D:/meta_matching_data/model_pth/{data_file_name}/{seed}_dnn4l_adamw_{data_file_name}.pth'
        set_random_seeds(seed)
        generator = torch.Generator()
        generator.manual_seed(seed) 

        basic_model = torch.load(basic_model_pth)

        for k_num in k_num_list: 
            print(f"==========================================K : {k_num}==========================================")
            # Secondary Model(KRR)에 들어갈 INPUT (K-shot & test) 
            kshot_pred_df, test_pred_df, top_k_indices = get_predict_df(basic_model, df, pheno_with_age, k_num, generator, seed) 
            # Hyperparam Tuning 
            kshot_x = kshot_pred_df.drop('Age', axis=1)
            kshot_y = kshot_pred_df['Age']
            krr = KernelRidge(kernel='rbf')# 하이퍼 파라미터 튜닝? 
            alphas = [0.1, 0.7, 1, 5, 10]
            param_grid = {'alpha':alphas}
            kf = KFold(n_splits=5)
            grid_search = GridSearchCV(krr, param_grid, cv=kf)
            grid_search.fit(kshot_x, kshot_y)

            best_alpha = grid_search.best_params_['alpha']
            best_krr = KernelRidge(kernel='rbf', alpha=best_alpha)  # 최적의 alpha를 가진 KRR 선언 
            best_krr.fit(kshot_x, kshot_y)

            # Testing 
            test_x = test_pred_df.drop('Age', axis=1)
            test_y = test_pred_df['Age']
            # prediction
            stacking_pred = best_krr.predict(test_x)

            final_pred_df = pd.DataFrame({'prediction':stacking_pred, 'Age':test_y})
            age_corr = get_corr_score(final_pred_df)
            age_cod = get_cod_score(final_pred_df)

            if k_num==10:
                age_corr_10.append(age_corr) 
                age_cod_10.append(age_cod)
                best_node_10.append(top_k_indices)
            elif k_num == 30: 
                age_corr_30.append(age_corr) 
                age_cod_30.append(age_cod)
                best_node_30.append(top_k_indices)
            elif k_num == 50: 
                age_corr_50.append(age_corr) 
                age_cod_50.append(age_cod)
                best_node_50.append(top_k_indices)
            elif k_num == 100: 
                age_corr_100.append(age_corr) 
                age_cod_100.append(age_cod)
                best_node_100.append(top_k_indices)
               
            print(f"Iteration {seed}, K : {k_num} - Correlation :{age_corr:.4f}".rjust(50))
            print(f"R2 Score :{age_cod:.4f}".rjust(50))
        print('\n\n\n')
    
    
    
    print(f"K=10 : Average COD : {np.mean(age_cod_10):.4f}")
    print(f"K=10 : STD     COD : {np.std(age_cod_10):.4f}")
    print()
    print(f"K=10 : Average Corr : {np.mean(age_corr_10):.4f}")
    print(f"K=10 : STD     Corr : {np.std(age_corr_10):.4f}")
    print('\n\n')
    print(f"K=30 : Average COD : {np.mean(age_cod_30):.4f}")
    print(f"K=30 : STD     COD : {np.std(age_cod_30):.4f}")
    print()
    print(f"K=30 : Average Corr : {np.mean(age_corr_30):.4f}")
    print(f"K=30 : STD     Corr : {np.std(age_corr_30):.4f}")
    print('\n\n')
    print(f"K=50 : Average COD : {np.mean(age_cod_50):.4f}")
    print(f"K=50 : STD     COD : {np.std(age_cod_50):.4f}")
    print()
    print(f"K=50 : Average Corr : {np.mean(age_corr_50):.4f}")
    print(f"K=50 : STD     Corr : {np.std(age_corr_50):.4f}")
    print('\n\n')
    print(f"K=100 : Average COD : {np.mean(age_cod_100):.4f}")
    print(f"K=100 : STD     COD : {np.std(age_cod_100):.4f}")
    print()
    print(f"K=100 : Average Corr : {np.mean(age_corr_100):.4f}")
    print(f"K=100 : STD     Corr : {np.std(age_corr_100):.4f}")               
    
    
    return age_cod_10, age_corr_10, age_cod_30, age_corr_30, age_cod_50, age_corr_50, age_cod_100, age_corr_100, best_node_10, best_node_30, best_node_50, best_node_100
    
    
# advanced_stacking(df, pheno_with_age, [10, 30, 50, 100], batch_size=128, iteration=100)


### STACKING END ### 

if __name__ == '__main__': 
    print(dnn_4l)