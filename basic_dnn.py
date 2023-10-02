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
warnings.filterwarnings("ignore")



### BASIC DNN 관련한 함수 START ### 
    
# 반복실험을 이 함수 안에서 도는게 아니라, 이 함수에 안에서는 오직 한번의 Training 및 Validation을 실행 
def basic_dnn_training(train_df, train_pheno, val_df, val_pheno, batch_size, generator, seed, data_file_name):
    
    device='cuda'
    best_loss = float('inf')
    
    # Data Loader 
    train_dataloader = get_dataloader(train_df, train_pheno, batch_size, generator, device)
    val_dataloader = get_dataloader(val_df, val_pheno, batch_size, generator, device)
    
    # Model Initialization
    # model = dnn_4l(train_df.shape[1], 87, 386, 313, 0.242, train_pheno.shape[1]).to(device)
    model = dnn_4l(train_df.shape[1], 128, 512, 64, 0.3, train_pheno.shape[1]).to(device)
    
    model.to(device) 
    loss_function = nn.MSELoss()
    epochs_to_decrease_lr = 200
    optimizer = optim.AdamW(model.parameters(), lr=1e-03, weight_decay=0.4)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)
    
    # 모든 Epoch에 대한 train loss와 val loss를 저장할 리스트 
    train_losses = []
    val_losses = []
    
    model_pth = f"D:/meta_matching_data/model_pth/{data_file_name}/{seed}_dnn4l_adamw_{data_file_name}.pth"
    folder_path = f"D:/meta_matching_data/model_pth/{data_file_name}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"{folder_path}가 생성되었습니다!")
        
    
    num_epochs = 200
    
    for epoch in range(num_epochs):
        # Training 
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_dataloader: 
            optimizer.zero_grad() 
            outputs = model(inputs) 
            loss = loss_function(outputs, targets) 
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)

        # Validation 
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_dataloader: 
                outputs = model(inputs) 
                loss = loss_function(outputs, targets) 
                val_loss += loss.item()
                
        val_loss /= len(val_dataloader)
        val_losses.append(val_loss)
        
        if best_loss > val_loss: 
            best_loss = val_loss 
            torch.save(model, model_pth)
            print(f"Epoch : {epoch}   Best Model! \t : Train Loss - {train_loss:.4f} | Val Loss - {val_loss:.4f}")
        
        elif epoch % 50 == 0: 
            print(f"Epoch : {epoch}               \t : Train Loss - {train_loss:.4f} | Val Loss - {val_loss:.4f}")
    
    
    return train_losses, val_losses

# FINE TUNING 에서 사용한 get_kshot_idx (kshot_age를 함수 안에서 나눈다.)
def get_kshot_idx(model, kshot_df, kshot_pheno, k_num, generator, device, seed): 
    '''
    Trained 된 모델을 가져와서 K sample들을 가지고 먼저 어떤 노드가 가장 높은 값을 가지는 지 확인한다. 
    '''
    model = model.to('cuda')
    kshot_age = kshot_pheno[:, -1]
    kshot_pheno = kshot_pheno[:, :-1]
    # to('cuda') & DataLoader 
    kshot_dataloader = get_dataloader(kshot_df, kshot_pheno, k_num, generator, device)
    
    # K Samples로 최대의 COD를 가지는 값
    model.eval()
    kshot_outputs = []
    
    with torch.no_grad():
        for inputs, targets in kshot_dataloader: 
            output = model(inputs)
            kshot_outputs.append(output)
            
    kshot_outputs = torch.cat(kshot_outputs).cpu().detach().numpy().T
    kshot_age_cods = []
    
    for i in range(len(kshot_outputs)):
        pred = pd.DataFrame({'prediction':kshot_outputs[i], 'Age': kshot_age})
        kshot_age_cods.append(get_cod_score(pred)) # 각 phenotype을 예측하여 이것을 Age와 COD를 계산
        # kshot_age_cods.append(get_corr_score(pred))
        
    max_cod_idx = kshot_age_cods.index(max(kshot_age_cods))
    
    return max_cod_idx



def test_model(model, test_df, test_pheno, test_age, max_cod_idx, batch_size, generator):
    device='cuda'
    # to('cuda') & DataLoader 
    test_df, test_pheno = (
        torch.Tensor(test_df).to(device),
        torch.Tensor(test_pheno).to(device)
    )
    test_dataset = torch.utils.data.TensorDataset(test_df, test_pheno)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=generator)
    
    
    # Testing
    test_outputs = []
    with torch.no_grad():
        for inputs, targets in test_dataloader: 
            output = model(inputs)
            test_outputs.append(output)
            
    test_outputs = torch.cat(test_outputs).cpu().detach().numpy()
    test_outputs= test_outputs[:, max_cod_idx].reshape(-1, 1) 
    pred_df = pd.DataFrame({'prediction' : test_outputs.flatten(), 'Age':test_age.flatten()})
    
    age_cod = get_cod_score(pred_df)
    age_corr = get_corr_score(pred_df)
    
    return age_corr, age_cod


def basic_dnn(df, pheno_with_age, k_num_list, data_file_name, batch_size=128, iteration=10, only_test=False):
    device='cuda'
    # 기본 k_num
    k_num=10
    test_size=0.2
    if not only_test: 
        # Training
        for seed in range(1, iteration+1): 
            print(f'==========================================Iter{seed}==========================================')
            # Random Seed Setting 
            set_random_seeds(seed) 
            generator = torch.Generator()
            generator.manual_seed(seed) 

            train_df, train_pheno, val_df, val_pheno, _, _, _, _ = \
                                        preprocess_data(df, pheno_with_age, test_size, k_num, seed)

            train_age = train_pheno[:, -1]
            train_pheno = train_pheno[:, :-1]
            val_age = val_pheno[:, -1]
            val_pheno = val_pheno[:, :-1]

            # 해당 Iteration에 대한 DataLoader 생성 / Model Initialization / Training 및 Validation을 진행하고, Best Model을 저장한다. 
            train_losses, val_losses = \
                    basic_dnn_training(train_df, train_pheno, val_df, val_pheno, batch_size, generator, seed, data_file_name)

            # 해당 Iteration의 train_loss와 val_loss를 가지고 plot을 생성 
            loss_img_pth = f'D:meta_matching_data/model_pth/plot/{data_file_name}/{seed}_dnn4l_adamw_{data_file_name}.png' 
            loss_img_folder_pth = f"d:/meta_matching_data/model_pth/plot/{data_file_name}"
            if not os.path.exists(loss_img_folder_pth):
                os.makedirs(loss_img_folder_pth)
                print(f"{loss_img_folder_pth}가 생성되었습니다!")
            save_iteration_loss_plot(train_losses, val_losses, loss_img_pth, seed)
        print('==========================================학습을 완료하였습니다.==========================================')
        print('\n\n')
    else: 
        print("학습을 건너뜁니다.")
        
        
        
    corrs_10 = []
    corrs_30 = []
    corrs_50 = []
    corrs_100 = []
    cods_10 = []
    cods_30 = []
    cods_50 = [] 
    cods_100 = []

    
    # K-shot / Test 성능 측정 
    for seed in range(1, iteration+1): 
        set_random_seeds(seed) 
        generator = torch.Generator()
        generator.manual_seed(seed) 
        
        # 학습 완료된 모델 Load 
        model_pth = f"D:/meta_matching_data/model_pth/{data_file_name}/{seed}_dnn4l_adamw_{data_file_name}.pth"
        model = torch.load(model_pth)
        
        for k_num in k_num_list: 
            print(f"==========================================K : {k_num}==========================================")
            _, _, _, _, kshot_df, kshot_pheno, test_df, test_pheno= preprocess_data(df, pheno_with_age, test_size, k_num, seed)
            # kshot_age = kshot_pheno[:, -1]
            # kshot_pheno = kshot_pheno[:, :-1]
            test_age = test_pheno[:, -1]
            test_pheno = test_pheno[:, :-1]
            
            # max_cod_idx를 K Samples로 계산 
            max_cod_idx = get_kshot_idx(model, kshot_df, kshot_pheno, k_num, generator, device, seed)
            
            # 계산된 max_cod_idx로 Test 성능 
            test_corr, test_cod = test_model(model, test_df, test_pheno, test_age, max_cod_idx, batch_size, generator)
            
            if k_num == 10 : 
                corrs_10.append(test_corr)
                cods_10.append(test_cod)
            elif k_num == 30: 
                corrs_30.append(test_corr)
                cods_30.append(test_cod)
            elif k_num == 50:
                corrs_50.append(test_corr)
                cods_50.append(test_cod)
            elif k_num == 100: 
                corrs_100.append(test_corr) 
                cods_100.append(test_cod)            
            print(f"Iteration {seed} | K = {k_num} : Corr - {test_corr:.4f} & R2 - {test_cod:.4f}")
            
        print('\n\n')
    
    print(f"K=10 : Average COD : {np.mean(cods_10)}")
    print(f"K=10 : STD     COD : {np.std(cods_10)}")
    print()
    print(f"K=10 : Average Corr : {np.mean(corrs_10)}")
    print(f"K=10 : STD     Corr : {np.std(corrs_10)}")
    print('\n\n')
    print(f"K=30 : Average COD : {np.mean(cods_30)}")
    print(f"K=30 : STD     COD : {np.std(cods_30)}")
    print()
    print(f"K=30 : Average Corr : {np.mean(corrs_30)}")
    print(f"K=30 : STD     Corr : {np.std(corrs_30)}")
    print('\n\n')
    print(f"K=50 : Average COD : {np.mean(cods_50)}")
    print(f"K=50 : STD     COD : {np.std(cods_50)}")
    print()
    print(f"K=50 : Average Corr : {np.mean(corrs_50)}")
    print(f"K=50 : STD     Corr : {np.std(corrs_50)}")
    print('\n\n')
    print(f"K=100 : Average COD : {np.mean(cods_100)}")
    print(f"K=100 : STD     COD : {np.std(cods_100)}")
    print()
    print(f"K=100 : Average Corr : {np.mean(corrs_100)}")
    print(f"K=100 : STD     Corr : {np.std(corrs_100)}")   
            
        
        
    return corrs_10, cods_10, corrs_30, cods_30, corrs_50, cods_50, corrs_100, cods_100
      
    
# corrs_10, cods_10, corrs_30, cods_30, corrs_50, cods_50, corrs_100, cods_100  = \
                # basic_dnn(df, pheno_with_age, k_num_list=[10, 30, 50, 100], batch_size=128, iteration=100, only_test=False)
    
### BASIC DNN 관련한 함수 END ### 