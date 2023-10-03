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
from basic_dnn import get_kshot_idx
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


### ADVANCED FINE_TUNING START ### 
def get_kshot_model(model_pth, max_cod_idx):
    # dnn4l이라는 전제 하에 함수가 짜여졌다. 
    before_model = torch.load(model_pth)
    after_model = torch.load(model_pth)
    
    # 변화시키려는 모델의 마지막 layer를 (64, 1)의 shape을 가지도록 만든다. (다른 조건들은 동일하게 유지)
    after_model.fc4 = nn.Sequential(nn.Dropout(p=0.3, inplace=False),
                          nn.Linear(in_features=64, out_features=1, bias=True))
    
    
    with torch.no_grad():
        after_model.fc4[1].weight = nn.Parameter(before_model.fc4[1].weight[max_cod_idx:max_cod_idx+1, :])
        after_model.fc4[1].bias = nn.Parameter(before_model.fc4[1].bias[max_cod_idx:max_cod_idx+1])
        
        
    for param in after_model.fc3.parameters():
        param.requires_grad = True 
    for param in after_model.fc4.parameters():
        param.requires_grad = True 
    
    after_model = after_model.to('cuda')
    
    return after_model


def kshot_fine_tuning(model, kshot_df, kshot_pheno, max_cod_idx, fine_tuned_model_pth, generator, device, seed, data_file_name): 
    
    # epochs_to_decrease_lr=30
    
    # last 2 layers update
    optimizer = torch.optim.AdamW([{'params': model.fc3.parameters()}, {'params': model.fc4.parameters()}], lr=1e-05, weight_decay=0.4)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)
    loss_function = nn.MSELoss()

    kshot_train_df, kshot_val_df, kshot_train_pheno, kshot_val_pheno = \
                            train_test_split(kshot_df, kshot_pheno, test_size=0.2, shuffle=False, random_state=seed)
    
    # kshot_train_iq = kshot_train_pheno[:, -1] # 이거 안쓰임
    # kshot_train_pheno = kshot_train_pheno[:, :-1]
    kshot_train_pheno = kshot_train_pheno[:, max_cod_idx] # 모델의 구조가 바뀌었으므로 max_cod_idx에 해당하는 phenotype만 남긴다. 
    kshot_val_iq = kshot_val_pheno[:, -1]
    # kshot_val_pheno = kshot_val_pheno[:, :-1]
    kshot_val_pheno = kshot_val_pheno[:, max_cod_idx] # 모델의 구조가 바뀌었으므로 max_cod_idx에 해당하는 phenotype만 남긴다. 
    
    # Load to GPU & DataLoader 
    kshot_train_dataloader = get_dataloader(kshot_train_df, kshot_train_pheno, kshot_train_df.shape[1], generator, device)
    kshot_val_dataloader = get_dataloader(kshot_val_df, kshot_val_pheno, kshot_val_df.shape[1], generator, device)
    

    # DNN k validation COD
    dnn_val_outputs = []
    model.eval() 
    with torch.no_grad():
        for inputs, targets in kshot_val_dataloader: 
            outputs = model(inputs) 
    dnn_val_outputs.append(outputs) 
    dnn_val_outputs = torch.cat(dnn_val_outputs).cpu().detach().numpy()

    pred = pd.DataFrame({'prediction': dnn_val_outputs[:, 0], 'IQ':kshot_val_iq})
    dnn_k_val_iq_cod = get_cod_score(pred)


    # Fine Tuning 진행 
    best_val_loss = float('inf')
    best_model = None
    fine_tune_count = 0
    epoch = 1 
    train_loss_list = []
    val_loss_list = []
    while epoch <= 200: 
        total_train_loss = 0
        model.train()
        
        for inputs, targets in kshot_train_dataloader: 
            optimizer.zero_grad()
            outputs = model(inputs) 
            train_loss = loss_function(outputs, targets) 
            total_train_loss += train_loss 
            train_loss.backward()
            optimizer.step() 
        scheduler.step()
        
        total_train_loss = total_train_loss / len(kshot_train_dataloader.dataset) 
        total_train_loss = torch.tensor(total_train_loss)
        train_loss_list.append(total_train_loss.to('cpu'))
        
        
        # kshot_val로 Validation Loss 계산하여 Stopping criterion
        val_preds = []
        model.eval() 
        with torch.no_grad():
            val_loss = 0 
            for inputs, targets in kshot_val_dataloader: 
                outputs = model(inputs) 
                loss = loss_function(outputs, targets) 
                val_loss += loss.item() * inputs.size(0)
                val_preds.append(outputs) 

        val_loss = val_loss / len(kshot_val_dataloader.dataset) 
        val_loss = torch.tensor(val_loss)
        val_loss_list.append(val_loss.to('cpu'))
        val_preds = torch.cat(val_preds).cpu().detach().numpy() 

        val_preds = val_preds.T
        val_gt = kshot_val_pheno.T
        val_pheno_cods = []

        if epoch % 50 == 0: 
            print(f"Epoch : {epoch} | Fine Tune Train Loss : {total_train_loss:.4f} | Validation Loss : {val_loss:.4f}")
        
        if val_loss < best_val_loss : 
            fine_tune_count = 0 
            best_val_loss = val_loss 
            best_model = model.state_dict() 
            print(f"Best Model! : Epoch {epoch} | Train Loss : {total_train_loss:.4f} | Validation Loss : {best_val_loss:.4f}")
            pred = pd.DataFrame({'prediction': val_preds[0], 'IQ':kshot_val_iq})
            k_val_iq_cod = get_cod_score(pred)

        epoch += 1

    loss_img_pth = f'D:/meta_matching_data/model_pth/plot/fine_tune_{data_file_name}/{seed}_K_{kshot_train_df.shape[0]}_fine_tuned_dnn4l_adamw_{data_file_name}.png'
    folder_pth = f"d:/meta_matching_data/model_pth/plot/fine_tune_{data_file_name}"
    if not os.path.exists(folder_pth):
        os.makedirs(folder_pth)
        print(f"{folder_pth}가 생성되었습니다!")


    save_iteration_loss_plot(train_loss_list, val_loss_list, loss_img_pth, seed)
    model.load_state_dict(best_model) 
    print(f"fine_tuned_model_pth : {fine_tuned_model_pth}")
    torch.save(model, fine_tuned_model_pth) 
    print(f"Saved Best Model!!!")
    return dnn_k_val_iq_cod, k_val_iq_cod
    
  
def test_finetuned_model(fine_tuned_model_pth, test_df, test_pheno, max_cod_idx, generator, seed ): 
    # Basic Model과 Fine-tuned 모델을 모두 비교해야 하기에 두 모델을 모두 불러오려고 했으나, 이것은 그냥, 나중에 test에 대한 성능을 측정하고 
    # 비교하여 Basic DNN과 Fine-tuned 중 고르면 될 것 같다. 
    # basic_model = torch.load(basic_model_pth)
    device='cuda'
    ft_model = torch.load(fine_tuned_model_pth)
    
    test_iq = test_pheno[:, -1]
    test_pheno = test_pheno[:, max_cod_idx:max_cod_idx+1]
    test_dataloader = get_dataloader(test_df, test_pheno, test_df.shape[0], generator, device)
    
    # Testing 
    test_outputs= []
    ft_model.eval()
    with torch.no_grad():
        for inputs, targets in test_dataloader: 
            outputs = ft_model(inputs)
            # loss = loss_function(outputs, targets)
            test_outputs.append(outputs)
            
    
    test_outputs = torch.cat(test_outputs).cpu().detach().numpy()
    test_outputs = test_outputs[:, 0].reshape(-1, 1) 
    
    pred = pd.DataFrame({'prediction': test_outputs[:, 0], 'IQ':test_iq})

    iq_corr = get_corr_score(pred)
    iq_cod = get_cod_score(pred)
    
    
    return iq_corr, iq_cod
   
    
def advanced_finetuning(df, pheno_with_iq, k_num_list, data_file_name, batch_size=128, iteration=10, only_test=False):
    '''
    Basic DNN에서 저장한 모델을 불러와서, K Sample로 어떤 Node가 가장 성능이 좋은지를 판단한다. 
    이후에, K sample을 4:1의 비율로 나눠 Training을 진행하고, Validation set으로 stopping criterion을 정한다. 
    '''
    device='cuda'
    patients = 30
    test_size=0.2
    
    # 1 ~ (iteration+1)번 모델까지 어떤 node가 가장 높은 성능을 보이는 지 저장할 리스트 
    ten_shot_idx = [-100]
    thirty_shot_idx = [-100]
    fifty_shot_idx = [-100]
    hund_shot_idx = [-100]

    
    test_iq_corr_10 = []
    test_iq_cod_10 = []
    test_iq_corr_30 = []
    test_iq_cod_30 = []
    test_iq_corr_50 = []
    test_iq_cod_50 = []
    test_iq_corr_100 = []
    test_iq_cod_100 = []
    dnn_kshot_cods = []
    adft_kshot_cods = []
    
    if not only_test:
        
        # 학습은 Basic DNN에서 저장한 모델을 사용 
        for seed in range(1, iteration+1): 
            set_random_seeds(seed) 
            generator = torch.Generator()
            generator.manual_seed(seed)
            # 학습 완료된 모델 Load 
            model_pth = f"D:/meta_matching_data/model_pth/{data_file_name}/{seed}_dnn4l_adamw_{data_file_name}.pth"
            basic_model = torch.load(model_pth)

            # Fine-Tuning 이전 K Sample을 가지고 각각 어떤 Node가 가장 높은 성능을 보이는 지 확인 및 저장
            for k_num in k_num_list: 
                print(f"==========================================K : {k_num}==========================================")
                _, _, _, _, kshot_df, kshot_pheno, test_df, test_pheno= preprocess_data(df, pheno_with_iq, test_size, k_num, seed)


                # BASIC DNN으로부터 K Samples로 max_cod_idx를 계산 
                max_cod_idx = get_kshot_idx(basic_model, kshot_df, kshot_pheno, k_num, generator, device, seed)

                print(f"MAX R2 score Node index is : {max_cod_idx}")
                if k_num == 10: 
                    ten_shot_idx.append(max_cod_idx) 
                elif k_num == 30: 
                    thirty_shot_idx.append(max_cod_idx) 
                elif k_num == 50: 
                    fifty_shot_idx.append(max_cod_idx) 
                elif k_num == 100: 
                    hund_shot_idx.append(max_cod_idx) 

            print("Best Node Selection 끝")

            # FINE_TUNING
            print("Fine Tuning Start!")
            for k_num in k_num_list: 
                _, _, _, _, kshot_df, kshot_pheno, test_df, test_pheno= preprocess_data(df, pheno_with_iq, test_size, k_num, seed)

                basic_model_pth = f'D:/meta_matching_data/model_pth/{data_file_name}/{seed}_dnn4l_adamw_{data_file_name}.pth'
                fine_tuned_model_pth = f'D:/meta_matching_data/model_pth/{data_file_name}/{seed}_k_{k_num}_fine_tuned_dnn4l_adamw_{data_file_name}.pth'
             
                if k_num == 10: 
                    max_cod_list = ten_shot_idx
                elif k_num == 30: 
                    max_cod_list = thirty_shot_idx
                elif k_num == 50: 
                    max_cod_list = fifty_shot_idx
                elif k_num == 100: 
                    max_cod_list = hund_shot_idx

                max_cod_idx = max_cod_list[seed-1]
                fine_tune_model = get_kshot_model(basic_model_pth, max_cod_idx) # Basic DNN에서 best node만 뽑아 구조를 변경한 모델
                dnn_k_shot_r2, adft_kshot_r2 = kshot_fine_tuning(fine_tune_model, kshot_df, kshot_pheno, max_cod_idx, fine_tuned_model_pth, generator, device, seed, data_file_name)

                dnn_kshot_cods.append(dnn_k_shot_r2)
                adft_kshot_cods.append(adft_kshot_r2)

                # max_cod_idx = get_kshot_idx(fine_tune_model, kshot_df, kshot_pheno, k_num, generator, device, seed) # 278번째 줄이 없어서, 코드가 제대로 돌아가지 않았었는데, 이 부분을 넣어서 그동안 잘못 돌아갔던 것.
                
                iq_corr, iq_cod = test_finetuned_model(fine_tuned_model_pth, test_df, test_pheno, max_cod_idx, generator, seed)

                print(f"Iteration {seed}, K : {k_num} - Correlation :{iq_corr:.4f}".rjust(50))
                print(f"R2 Score :{iq_cod:.4f}".rjust(50))
                print('\n\n\n')


                if k_num == 10: 
                    test_iq_corr_10.append(iq_corr) 
                    test_iq_cod_10.append(iq_cod)
                elif k_num == 30: 
                    test_iq_corr_30.append(iq_corr)
                    test_iq_cod_30.append(iq_cod)
                elif k_num ==50: 
                    test_iq_corr_50.append(iq_corr) 
                    test_iq_cod_50.append(iq_cod)
                elif k_num == 100: 
                    test_iq_corr_100.append(iq_corr) 
                    test_iq_cod_100.append(iq_cod) 
        print('\n\n\n\n')
        print("FINE TUNING 완료")
    else:
        print("FINE TUNING을 건너뜁니다.")    
    
    print(f"K=10 : Average COD : {np.mean(test_iq_cod_10):.4f}")
    print(f"K=10 : STD     COD : {np.std(test_iq_cod_10):.4f}")
    print()
    print(f"K=10 : Average Corr : {np.mean(test_iq_corr_10):.4f}")
    print(f"K=10 : STD     Corr : {np.std(test_iq_corr_10):.4f}")
    print('\n\n')
    print(f"K=30 : Average COD : {np.mean(test_iq_cod_30):.4f}")
    print(f"K=30 : STD     COD : {np.std(test_iq_cod_30):.4f}")
    print()
    print(f"K=30 : Average Corr : {np.mean(test_iq_corr_30):.4f}")
    print(f"K=30 : STD     Corr : {np.std(test_iq_corr_30):.4f}")
    print('\n\n')
    print(f"K=50 : Average COD : {np.mean(test_iq_cod_50):.4f}")
    print(f"K=50 : STD     COD : {np.std(test_iq_cod_50):.4f}")
    print()
    print(f"K=50 : Average Corr : {np.mean(test_iq_corr_50):.4f}")
    print(f"K=50 : STD     Corr : {np.std(test_iq_corr_50):.4f}")
    print('\n\n')
    print(f"K=100 : Average COD : {np.mean(test_iq_cod_100):.4f}")
    print(f"K=100 : STD     COD : {np.std(test_iq_cod_100):.4f}")
    print()
    print(f"K=100 : Average Corr : {np.mean(test_iq_corr_100):.4f}")
    print(f"K=100 : STD     Corr : {np.std(test_iq_corr_100):.4f}")   
            
    
    return test_iq_corr_10, test_iq_cod_10, test_iq_corr_30, test_iq_cod_30, test_iq_corr_50, test_iq_cod_50, test_iq_corr_100, test_iq_cod_100, ten_shot_idx[1:], thirty_shot_idx[1:], fifty_shot_idx[1:], hund_shot_idx[1:], dnn_kshot_cods, adft_kshot_cods
   
# corrs_10, cods_10, corrs_30, cods_30, corrs_50, cods_50, corrs_100, cods_100 = \
#           advanced_finetuning(df, pheno_with_iq, k_num_list=[10,30,50, 100], batch_size=128, iteration=100)


### FINE_TUNING END ### 