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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold
from CBIG_model_pytorch import dnn_4l, dnn_5l
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
# import statsmodels.api as sm
import matplotlib.patches as mpatches
import warnings
from utils import *
warnings.filterwarnings("ignore")

# 일단 K=100인 경우에 한해서 코드를 짤 계획이다. 


# Reference : https://code-angie.tistory.com/7

def get_kshot_model_preds(model, df, pheno_with_iq, generator, seed):
    device='cuda'
    # Data Loader & Model Initialization 
    _, _, _, _, kshot_df, kshot_pheno, test_df, test_pheno= preprocess_data(df, pheno_with_iq, 0.2, 100, seed)
    
    kshot_pheno = kshot_pheno[:, :-1]
    test_pheno = test_pheno[:, :-1]
    kshot_dataloader = get_dataloader(kshot_df, kshot_pheno, 100, generator, device)
    
    model = model.to(device) 
    model.eval() 
    kshot_outputs = []
    with torch.no_grad(): 
        for inputs, targets in kshot_dataloader: 
            output = model(inputs) 
            kshot_outputs.append(output)
    kshot_outputs = torch.cat(kshot_outputs).cpu().detach().numpy()
    # print(f"Model kshot prediction shape : {kshot_outputs.shape}")
    return kshot_outputs


def get_test_model_preds(model, df, pheno_with_iq, generator, seed): 
    device='cuda' 
    _, _, _, _, kshot_df, kshot_pheno, test_df, test_pheno= preprocess_data(df, pheno_with_iq, 0.2, 100, seed)
    
    test_pheno = test_pheno[:, :-1]
    test_dataloader = get_dataloader(test_df, test_pheno, 100, generator, device)

    model = model.to(device) 
    model.eval() 
    test_outputs = []
    with torch.no_grad(): 
        for inputs, targets in test_dataloader: 
            output = model(inputs) 
            test_outputs.append(output)
    test_outputs = torch.cat(test_outputs).cpu().detach().numpy()
    # print(f"Model test prediction shape : {test_outputs.shape}")
    return test_outputs



class Embedding(nn.Module): 
    def __init__(self, d_orig, d_model): 
        super().__init__()
        self.w_emb = nn.Linear(d_orig, d_model) 

    def forward(self, x): 
        return self.w_emb(x) 

    
class ScaleDotProductAttention(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.softmax = nn.Softmax(dim=-1) 

    def forward(self, q, k, v): 
        _, _, _, head_dim = q.size() 

        k_t = k.transpose(-1, -2) 

        # Q, K^T MatMul
        attention_score = torch.matmul(q, k_t) 
        # Scaling 
        attention_score /= math.sqrt(head_dim)
        # Softmax
        attention_score = self.softmax(attention_score) 

        result = torch.matmul(attention_score, v) 

        return result, attention_score

class MultiHeadAttention(nn.Module): 
    def __init__(self, d_model, head): 
        super().__init__()
        self.d_model = d_model 
        self.head = head 
        self.head_dim = d_model // head 
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attention = ScaleDotProductAttention() 

    def forward(self, q, k, v): 
        batch_size, _, _ = q.size()

        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v) 

        q = q.view(batch_size, -1, self.head, self.head_dim).transpose(1, 2) 
        k = k.view(batch_size, -1, self.head, self.head_dim).transpose(1, 2) 
        v = v.view(batch_size, -1, self.head, self.head_dim).transpose(1, 2) 

        # Scaled Dot-product Attention
        out, attention_score = self.attention(q, k, v) 

        # 분리된 head concat (Head가 1인 경우에는 딱히 필요 없다.) 
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # d_model projection 
        out = self.w_o(out)

        return out, attention_score 

class PositionWiseFCFeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.w_1 = nn.Linear(d_model,d_ff)
        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.w_2 = nn.Linear(d_ff,d_model)
    
    def forward(self, x):
        # Linear Layer1
        x = self.w_1(x)
        # ReLU
        x = self.relu(x)
        # x = self.elu(x) 
        # Linear Layer2
        x = self.w_2(x)

        return x
    

class EncoderLayer(nn.Module): 
    def __init__(self, d_orig, d_model, head, d_ff, dropout): 
        super().__init__()
        # self.emb = Embedding(d_orig, d_model) 
        self.attention = MultiHeadAttention(d_model, head) 
        self.Norm1 = nn.LayerNorm(d_model)  # 이거 BatchNorm으로 바꿔야 하나??? 
       

        self.ffn = PositionWiseFCFeedForwardNetwork(d_model, d_ff) 
        self.Norm2 = nn.LayerNorm(d_model) 
        

        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x): 
        # x = self.emb(x) 
        residual = x 

        # (Multi-Head) Attention
        x, attention_score = self.attention(q=x, k=x, v=x) 

        # Add & Norm 
        x = self.dropout(x) + residual 
        x = self.Norm1(x) 

        residual = x 

        # Feed-forward Network 
        x = self.ffn(x) 

        # Add & Norm 
        x = self.dropout(x) + residual 
        x = self.Norm2(x) 

        return x, attention_score



class Encoder(nn.Module): 
    def __init__(self, d_orig, d_model, head, d_ff, dropout, n_layers, device): 
        super().__init__() 

        # Embedding 
        self.input_emb = Embedding(d_orig, d_model) 
        self.dropout = nn.Dropout(p=dropout)

        # n개의 encoder layer를 list에 담기 
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_orig= d_orig, 
                                                          d_model=d_model, 
                                                          head = head, 
                                                          d_ff = d_ff, 
                                                          dropout=dropout)
                                            for _ in range(n_layers)])
        

    def forward(self, x): 
        # 1. 입력에 대한 input_embedding 생성 
        input_emb = self.input_emb(x) 

        # 2. Add & Dropout 
        x = self.dropout(input_emb) 

        # 3. n번 EncoderLayer 반복 
        for encoder_layer in self.encoder_layers:
            x, attention_score = encoder_layer(x) 

        return x 



class Transformer(nn.Module): 
    def __init__(self, d_orig, d_model, head, d_ff, dropout, n_layers, device): 
        super().__init__() 
        self.device = device 
        self.d_model = d_model

        # Encoder 
        self.encoder = Encoder(d_orig=d_orig, d_model=d_model, head=head, d_ff=d_ff, 
                               dropout=dropout, n_layers=n_layers, device=device)
        
        self.linear = nn.Linear(58 * d_model, 1)

    def forward(self, src): 
        memory = self.encoder(src)
        memory = memory.view(-1, 58 * self.d_model)
        output = self.linear(memory)
        output = output.squeeze()
        return output



def transformer_stacking(pheno_with_iq, f_list, d_model, option, seed):
    device='cuda' 
    set_random_seeds(seed)
    generator = torch.Generator()
    generator.manual_seed(seed) 

    model_dict = {} 
    for data_file_name in f_list: 
        model_dict[data_file_name] = torch.load(f"D:/meta_matching_data/model_pth/{data_file_name}/{seed}_dnn4l_adamw_{data_file_name}.pth")
        print(f"{data_file_name} Model Load Complete!")
    

    # Basic DNN으로부터 Prediction을 뽑아, 이것을 Transformer에 들어갈 입력으로 만드는 과정 
    kshot_dict = {} 
    test_dict = {} 
    for data_file_name in model_dict:  
        loaded_data = np.load(f"D:/meta_matching_data/INPUT_DATA/{data_file_name}.npy")
        df = pd.DataFrame(loaded_data, index=pheno_with_iq.index)
        kshot_dict[data_file_name] = get_kshot_model_preds(model_dict[data_file_name], df, pheno_with_iq, generator, seed)
        test_dict[data_file_name] = get_test_model_preds(model_dict[data_file_name], df, pheno_with_iq, generator, seed)
    
    kshot_preds = list(kshot_dict.values())
    test_preds = list(test_dict.values())
    
    kshot_preds_concat = np.stack(kshot_preds, axis=-1) 
    kshot_src = torch.tensor(kshot_preds_concat) 
    print("Base Model Prediction Complete! (KSHOT)")
    # print(f"KSHOT Source size : {kshot_src.size()}")

    test_preds_concat = np.stack(test_preds, axis=-1) 
    test_src = torch.tensor(test_preds_concat) 
    print("Base Model Prediction Complete! (TEST)")
    # print(f"TEST Source size : {test_src.size()}")

    
    # Transformer Model Config
    d_orig = kshot_src.size(-1)
    d_model = d_model
    head = 2
    d_ff = d_model
    dropout=0.1
    n_layers = 2

    transformer = Transformer(d_orig, d_model, head, d_ff, dropout, n_layers, device)

    ##### Transformer TRAINING #####
    _, _, _, _, kshot_df, kshot_pheno, test_df, test_pheno= preprocess_data(df, pheno_with_iq, 0.2, 100, seed)
    
    
    kshot_iq = kshot_pheno[:, -1]
    test_iq = test_pheno[:, -1]
    kshot_dataloader = get_dataloader(kshot_src, kshot_iq, 100, generator, device)
    test_dataloader = get_dataloader(test_src, test_iq, 100, generator, device)



    num_epochs=5000
    optimizer = optim.AdamW(transformer.parameters(), lr=1e-6, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-08)
    loss_function = nn.MSELoss()
    transformer = transformer.to(device)
    print("Transformer Model Load Complete!")



    #### Transformer Training (KSHOT) #### 
    kshot_losses = []
    print("Transformer Model Training Start!\n\n")
    for epoch in range(num_epochs):         
        transformer.train() 
        kshot_loss = 0.0 
        kshot_outputs = []
        for inputs, targets in kshot_dataloader:
            optimizer.zero_grad()
            outputs = transformer(inputs) 
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            kshot_loss += loss.item() 
            kshot_outputs.append(outputs)
            
        kshot_outputs = torch.cat(kshot_outputs).cpu().detach().numpy() 
        kshot_pred_df = pd.DataFrame({'prediction' : kshot_outputs.flatten(), 'IQ':kshot_iq.flatten()})
        kshot_corr = get_corr_score(kshot_pred_df)
        kshot_cod = get_cod_score(kshot_pred_df)
        
        scheduler.step()
        kshot_loss /= len(kshot_dataloader) 
        kshot_losses.append(kshot_loss) 
        if epoch % 100 == 0 : 
            print(f"Epoch : {epoch}      \t : Kshot Loss - {kshot_loss:.4f} | Kshot COD - {kshot_cod:.4f} | Kshot Corr - {kshot_corr:.4f}")
            
        
        
    ##### Test #####
    transformer.eval() 
    
    
    # Saving Model 
    transformer_dir = f"D:/meta_matching_data/model_pth/transformer_{option}/"
    make_dirs(transformer_dir)
    model_name = f"{seed}_transformer.pth"
    torch.save(transformer, transformer_dir+model_name)
    print("Model Saved!")
    
    
    transformer = torch.load(transformer_dir + model_name)
    test_loss = 0.0

    test_outputs = [] 
    with torch.no_grad(): 
        for inputs, targets in test_dataloader: 
            outputs = transformer(inputs) 
            loss = loss_function(outputs, targets) 
            test_loss += loss.item() 
            test_outputs.append(outputs) 
    
    test_outputs = torch.cat(test_outputs).cpu().detach().numpy() 
    pred_df = pd.DataFrame({'prediction' : test_outputs.flatten(), 'IQ':test_iq.flatten()})
    test_cod = get_cod_score(pred_df) 
    test_corr = get_corr_score(pred_df) 

    test_loss /= len(test_dataloader) 
    print(f"Test Loss - {test_loss:.4f}\n\n")
    print(f"Test Corr - {test_corr:.4f}\n\n")
    print(f"Test COD - {test_cod:.4f}\n\n")


    return test_corr, test_cod





def advanced_transformer(pheno_with_iq, f_list, d_model, option, iteration=50): 
    
    test_dict = {'corr' : [], 'cod':[]} 
    
    for seed in range(1, iteration + 1): 
        print(f'==========================================Iter : {seed}==========================================')
        test_corr, test_cod = transformer_stacking(pheno_with_iq, f_list, d_model, option, seed)
        test_dict['corr'].append(test_corr) 
        test_dict['cod'].append(test_cod) 
        
        
    test_df = pd.DataFrame(test_dict)
    csv_dir = 'D:/meta_matching_data/transformer_stacking_csv/'
    make_dirs(csv_dir)
    test_df.to_csv(f'D:/meta_matching_data/transformer_stacking_csv/{option}.csv')
    print(f'D:/meta_matching_data/transformer_stacking_csv/{option}.csv를 생성하였습니다.')
    
    return test_df
        
        
    