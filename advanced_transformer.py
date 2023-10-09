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
        # self.relu = nn.ReLU()
        self.elu = nn.ELU()
        self.w_2 = nn.Linear(d_ff,d_model)
    
    def forward(self, x):
        # Linear Layer1
        x = self.w_1(x)
        # ReLU
        # x = self.relu(x)
        x = self.elu(x) 
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



def transformer_stacking(pheno_with_iq, k_num=100, seed=1):
    device='cuda' 
    set_random_seeds(seed)
    generator = torch.Generator()
    generator.manual_seed(seed) 
    
    # Load Model 
    model_gamb = torch.load("D:/meta_matching_data/model_pth/functional_gamb/1_dnn4l_adamw_functional_gamb.pth")
    model_lang = torch.load("D:/meta_matching_data/model_pth/functional_lang/1_dnn4l_adamw_functional_lang.pth")
    model_wm = torch.load("D:/meta_matching_data/model_pth/functional_wm/1_dnn4l_adamw_functional_wm.pth")
    model_social = torch.load("D:/meta_matching_data/model_pth/functional_social/1_dnn4l_adamw_functional_social.pth")
    model_motor = torch.load("D:/meta_matching_data/model_pth/functional_motor/1_dnn4l_adamw_functional_motor.pth")
    model_rs = torch.load("D:/meta_matching_data/model_pth/functional_restingstate/1_dnn4l_adamw_functional_restingstate.pth")
    
    # sMRI
    model_area= torch.load("D:/meta_matching_data/model_pth/structural_area/1_dnn4l_adamw_structural_area.pth")
    model_thick= torch.load("D:/meta_matching_data/model_pth/structural_thick/1_dnn4l_adamw_structural_thick.pth")
    model_vol= torch.load("D:/meta_matching_data/model_pth/structural_vol/1_dnn4l_adamw_structural_vol.pth")
    model_mind= torch.load("D:/meta_matching_data/model_pth/structural_mind/1_dnn4l_adamw_structural_mind.pth")
    model_msn= torch.load("D:/meta_matching_data/model_pth/structural_msn/1_dnn4l_adamw_structural_msn.pth")
    
    # TBSS 
    model_tbss_ad = torch.load("D:/meta_matching_data/model_pth/tbss_ad/1_dnn4l_adamw_tbss_ad.pth")
    model_tbss_fa = torch.load("D:/meta_matching_data/model_pth/tbss_fa/1_dnn4l_adamw_tbss_fa.pth")
    model_tbss_icvf = torch.load("D:/meta_matching_data/model_pth/tbss_icvf/1_dnn4l_adamw_tbss_icvf.pth")
    model_tbss_isovf = torch.load("D:/meta_matching_data/model_pth/tbss_isovf/1_dnn4l_adamw_tbss_isovf.pth")
    model_tbss_md = torch.load("D:/meta_matching_data/model_pth/tbss_md/1_dnn4l_adamw_tbss_md.pth")
    model_tbss_od = torch.load("D:/meta_matching_data/model_pth/tbss_od/1_dnn4l_adamw_tbss_od.pth")
    model_tbss_rd = torch.load("D:/meta_matching_data/model_pth/tbss_rd/1_dnn4l_adamw_tbss_rd.pth")
    
    model_tracto_ad = torch.load("D:/meta_matching_data/model_pth/tracto_ad/1_dnn4l_adamw_tracto_ad.pth")
    model_tracto_fa = torch.load("D:/meta_matching_data/model_pth/tracto_fa/1_dnn4l_adamw_tracto_fa.pth")
    model_tracto_fss = torch.load("D:/meta_matching_data/model_pth/tracto_fss/1_dnn4l_adamw_tracto_fss.pth")
    model_tracto_fssl = torch.load("D:/meta_matching_data/model_pth/tracto_fssl/1_dnn4l_adamw_tracto_fssl.pth")
    model_tracto_icvf = torch.load("D:/meta_matching_data/model_pth/tracto_icvf/1_dnn4l_adamw_tracto_icvf.pth")
    model_tracto_isovf = torch.load("D:/meta_matching_data/model_pth/tracto_isovf/1_dnn4l_adamw_tracto_isovf.pth")
    model_tracto_md = torch.load("D:/meta_matching_data/model_pth/tracto_md/1_dnn4l_adamw_tracto_md.pth")
    model_tracto_od = torch.load("D:/meta_matching_data/model_pth/tracto_od/1_dnn4l_adamw_tracto_od.pth")
    model_tracto_rd = torch.load("D:/meta_matching_data/model_pth/tracto_rd/1_dnn4l_adamw_tracto_rd.pth")
    # f_list = ['functional_restingstate', 'functional_gamb', 'functional_lang','functional_wm', 'functional_social','functional_motor']
    # model_list = [model_rs, model_gamb, model_lang, model_wm, model_social, model_motor]
    # f_list = ['functional_wm', 'structural_area', 'tbss_ad','tracto_ad', 'functional_restingstate']
    # model_list = [model_wm, model_area, model_tbss_ad, model_tracto_ad, model_rs]
    
    f_list = ['functional_restingstate', 'functional_gamb', 'functional_lang','functional_wm', 'functional_social','functional_motor',
             'structural_area', 'structural_thick','structural_vol','structural_mind','structural_msn', 
             'tbss_ad','tbss_fa','tbss_icvf','tbss_isovf','tbss_md','tbss_od','tbss_rd',
             'tracto_ad','tracto_fa','tracto_fss','tracto_fssl','tracto_icvf','tracto_isovf','tracto_md','tracto_od','tracto_rd']
    model_list = [model_rs, model_gamb, model_lang, model_wm, model_social, model_motor, 
                 model_area, model_thick, model_vol, model_mind, model_msn, 
                 model_tbss_ad, model_tbss_fa, model_tbss_icvf, model_tbss_isovf, model_tbss_md, model_tbss_od, model_tbss_rd,
                 model_tracto_ad, model_tracto_fa, model_tracto_fss, model_tracto_fssl, model_tracto_icvf, model_tracto_isovf, 
                 model_tracto_md, model_tracto_od, model_tracto_rd]

    # Basic DNN으로부터 Prediction을 뽑아, 이것을 Transformer에 들어갈 입력으로 만드는 과정 
    kshot_dict = {} 
    test_dict = {} 
    for f_data, model in zip(f_list, model_list): 
        data_file_name = f_data 
        loaded_data = np.load(f"D:/meta_matching_data/INPUT_DATA/{data_file_name}.npy")
        df = pd.DataFrame(loaded_data, index=pheno_with_iq.index)
        kshot_dict[f_data] = get_kshot_model_preds(model, df, pheno_with_iq, generator, seed)
        test_dict[f_data] = get_test_model_preds(model, df, pheno_with_iq, generator, seed)
    
    kshot_preds = list(kshot_dict.values())
    test_preds = list(test_dict.values())
    
    kshot_preds_concat = np.stack(kshot_preds, axis=-1) 
    kshot_src = torch.tensor(kshot_preds_concat) 
    print("Base Model Prediction Complete! (KSHOT)")
    print(f"KSHOT Source size : {kshot_src.size()}")

    test_preds_concat = np.stack(test_preds, axis=-1) 
    test_src = torch.tensor(test_preds_concat) 
    print("Base Model Prediction Complete! (TEST)")
    print(f"TEST Source size : {test_src.size()}")
    
    # Transformer Model Config
    d_orig = kshot_src.size(-1)
    d_model = 64
    head = 2
    d_ff = 64
    dropout=0.1
    n_layers = 2

    transformer = Transformer(d_orig, d_model, head, d_ff, dropout, n_layers, device)

    ##### Transformer TRAINING #####
    _, _, _, _, kshot_df, kshot_pheno, test_df, test_pheno= preprocess_data(df, pheno_with_iq, 0.2, 100, seed)
    
    # kshot_pheno = kshot_pheno[:, :-1]
    kshot_iq = kshot_pheno[:, -1]
    # test_pheno = test_pheno[:, :-1]
    test_iq = test_pheno[:, -1]
    # kshot_dataloader = get_dataloader(kshot_df, kshot_pheno, 100, generator, device)
    kshot_dataloader = get_dataloader(kshot_src, kshot_iq, 100, generator, device)
    test_dataloader = get_dataloader(test_src, test_iq, 100, generator, device)

    num_epochs=10000
    optimizer = optim.AdamW(transformer.parameters(), lr=1e-6, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-08)
    loss_function = nn.MSELoss()
    transformer = transformer.to(device)
    print("Transformer Model Load Complete!")


    kshot_losses = []
    test_losses = []
    print("Transformer Model Training Start!")
    best_loss = float("inf")
    for epoch in range(num_epochs): 
        #### Transformer Training (KSHOT) #### 
        transformer.train() 
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
        transformer.eval()
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
        # print(f"test_outputs: {test_outputs[:5]}" )


        test_loss /= len(test_dataloader) 
        test_losses.append(test_loss)

        if best_loss > test_loss: 
            best_loss = test_loss 
            test_cod = get_cod_score(pred_df)
            test_corr = get_corr_score(pred_df)
            # torch.save(model, model_pth) 
            print(f"Epoch : {epoch}   Best Model! \t : Kshot Loss - {kshot_loss:.4f} | Test Loss - {test_loss:.4f} ====> COD : {test_cod:.4f} & Corr : {test_corr:.4f}")

        elif epoch % 100 == 0 : 
            print(f"Epoch : {epoch}               \t : Kshot Loss - {kshot_loss:.4f} | Test Loss - {test_loss:.4f} ====> COD : {test_cod:.4f} & Corr : {test_corr:.4f}")
        

    return kshot_losses, test_losses 




