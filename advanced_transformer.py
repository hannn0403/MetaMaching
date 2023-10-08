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
    print(f"Model kshot prediction shape : {kshot_outputs.shape}")
    return kshot_outputs


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
        # out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # d_model projection 
        out = self.w_o(out)

        return out, attention_score 

class PositionWiseFCFeedForwardNetwork(nn.Module):
    def __init__(self,d_model,d_ff):
        super().__init__()
        self.w_1 = nn.Linear(d_model,d_ff)
        self.relu = nn.ReLU()
        self.w_2 = nn.Linear(d_ff,d_model)
    
    def forward(self, x):
        # Linear Layer1
        x = self.w_1(x)
        # ReLU
        x = self.relu(x)
        # Linear Layer2
        x = self.w_2(x)

        return x
    

class EncoderLayer(nn.Module): 
    def __init__(self, d_orig, d_model, head, d_ff, dropout): 
        super().__init__()
        # self.emb = Embedding(d_orig, d_model) 
        self.attention = MultiHeadAttention(d_model, head) 
        self.layerNorm1 = nn.LayerNorm(d_model)  # 이거 BatchNorm으로 바꿔야 하나??? 

        self.ffn = PositionWiseFCFeedForwardNetwork(d_model, d_ff) 
        self.layerNorm2 = nn.LayerNorm(d_model) 

        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x): 
        # x = self.emb(x) 
        residual = x 

        # (Multi-Head) Attention
        x, attention_score = self.attentio0n(q=x, k=x, v=x) 

        # Add & Norm 
        x = self.dropout(x) + residual 
        x = self.layerNorm1(x) 

        residual = x 

        # Feed-forward Network 
        x = self.ffn(x) 

        # Add & Norm 
        x = self.dropout(x) + residual 
        x = self.layerNorm2(x) 

        return x, attention_score



class Encoder(nn.Module): 
    def __init__(self, d_orig, d_model, head, d_ff, dropout, n_layers, device): 
        super().__init__() 

        # Embedding 
        self.input_emb = Embedding(d_orig, d_model) 
        self.dropout = nn.Dropout(p=dropout)

        # n개의 encoder layer를 list에 담기 
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model=d_model, 
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

        # Encoder 
        self.encoder = Encoder(d_orig=d_orig, d_model=d_model, head=head, d_ff=d_ff, 
                               dropout=dropout, n_layers=n_layers, device=device)
        
        self.linear = nn.Linear(d_model, 1)

    def forward(self, src): 
        memory = self.encoder(src)
        output = self.linear(memory)
        return output



def transformer_stacking(pheno_with_iq, k_num=100, seed=1):
    device='cuda' 
    set_random_seeds(seed)
    generator = torch.Generator()
    generator.manual_seed(seed) 
    
    # Load Model 
    model_gamb = torch.load("C:/Users/HanJH/meta_matching_data/model_pth/functional_dnn_model/1_dnn4l_adamw_functional_gamb.pth")
    model_lang = torch.load("C:/Users/HanJH/meta_matching_data/model_pth/functional_dnn_model/1_dnn4l_adamw_functional_lang.pth")
    model_wm = torch.load("C:/Users/HanJH/meta_matching_data/model_pth/functional_dnn_model/1_dnn4l_adamw_functional_wm.pth")
    model_social = torch.load("C:/Users/HanJH/meta_matching_data/model_pth/functional_dnn_model/1_dnn4l_adamw_functional_social.pth")
    model_motor = torch.load("C:/Users/HanJH/meta_matching_data/model_pth/functional_dnn_model/1_dnn4l_adamw_functional_motor.pth")
    model_rs = torch.load("C:/Users/HanJH/meta_matching_data/model_pth/functional_dnn_model/1_dnn4l_adamw_functional_restingstate.pth")

    f_list = ['functional_restingstate', 'functional_gamb', 'functional_lang','functional_wm', 'functional_social','functional_motor']
    model_list = [model_rs, model_gamb, model_lang, model_wm, model_social, model_motor]



    # Basic DNN으로부터 Prediction을 뽑아, 이것을 Transformer에 들어갈 입력으로 만드는 과정 
    preds_dict = {} 
    for f_data, model in zip(f_list, model_list): 
        data_file_name = f_data 
        loaded_data = np.load(f"C:/Users/HanJH/meta_matching_data/INPUT_DATA/{data_file_name}.npy")
        df = pd.DataFrame(loaded_data, index=pheno_with_iq.index)
        preds_dict[f_data] = get_kshot_model_preds(model, df, pheno_with_iq, generator, seed)

    preds = list(preds_dict.values())
    concat_preds = np.stack(preds, axis=-1) 
    src = torch.tensor(concat_preds) 
    
    
    # Transformer Model Config
    d_orig = src.size(-1)
    d_model = 8
    head = 1
    d_ff = 8
    dropout=0.1 
    n_layers = 1 

    transformer = Transformer(d_orig, d_model, head, d_ff, dropout, n_layers, device)

    ##### Transformer TRAINING #####
    _, _, _, _, kshot_df, kshot_pheno, test_df, test_pheno= preprocess_data(df, pheno_with_iq, 0.2, 100, seed)
    
    kshot_pheno = kshot_pheno[:, :-1]
    kshot_iq = kshot_pheno[:, -1]
    test_pheno = test_pheno[:, :-1]
    test_iq = test_pheno[:, -1]
    # kshot_dataloader = get_dataloader(kshot_df, kshot_pheno, 100, generator, device)
    kshot_dataloader = get_dataloader(kshot_df, kshot_iq, 100, generator, device)
    test_dataloader = get_dataloader(test_df, test_iq, 100, generator, device)

    num_epochs=3
    optimizer = optim.AdamW(model.parameters(), lr=1e-06, weight_decay=0.001)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-08)
    loss_function = nn.MSELoss()
    transformer = transformer.to(device)


    kshot_losses = []
    test_losses = []
    for epoch in range(num_epochs): 
        #### Transformer Training (KSHOT) #### 
        model.train() 
        kshot_loss = 0.0 

        for inputs, targets in kshot_dataloader:
            optimizer.zero_grad()
            outputs = transformer(inputs) 
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            kshot_loss += loss.item() 
        
        scheduler.step()
        kshot_loss /= len(kshot_dataloader) 
        kshot_losses.append(kshot_loss) 

        ##### Test #####
        model.eval()
        test_loss = 0.0 
        

        with torch.no_grad(): 
            for inputs, targets in test_dataloader: 
                outputs = transformer(inputs) 
                loss = loss_function(outputs, targets) 
                test_loss += loss.item() 

        test_loss /= len(test_dataloader) 
        test_losses.append(test_loss)

        if best_loss > test_loss: 
            best_loss = test_loss 
            # torch.save(model, model_pth) 
            print(f"Epoch : {epoch}   Best Model! \t : Kshot Loss - {kshot_loss:.4f} | Test Loss - {test_loss:.4f}")
        
        else: 
            print(f"Epoch : {epoch}               \t : Kshot Loss - {kshot_loss:.4f} | Test Loss - {test_loss:.4f}")


    return kshot_losses, test_losses 




