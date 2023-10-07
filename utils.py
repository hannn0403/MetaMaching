# initialization and random seed set

import os
import scipy
import scipy.io
import seaborn as sns
import torch
import math
import random
import sklearn
import numpy as np
import pandas as pd
import torch.nn as nn
from collections import Counter
from sklearn.metrics import r2_score
from scipy.spatial import distance 
from scipy.sparse import coo_matrix, csr 
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




def get_upper_tri(directory_path, length):
    file_list = []
    idx = []
    for filename in os.listdir(directory_path):
        subject = scipy.io.loadmat(directory_path + filename) 
        a = subject['connectivity'][np.triu_indices(length, k=1)]
        file_list.append(a) 
        idx.append(filename[:6])
    result = np.vstack(file_list) 
    return idx, result


def get_common_elements(list1, list2): 
    counter1 = Counter(list1) 
    counter2 = Counter(list2) 
    
    # 교집합 
    intersection = counter1 & counter2 
    common_values = list(intersection.elements())
    
    return common_values


def get_additional_elements(small_list, big_list):
    small_set = set(small_list)
    big_set = set(big_list) 
    
    result_set = big_set - small_set 
    result = list(result_set)
    return result 


def set_random_seeds(seed): 
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
def preprocess_data(df, pheno_with_iq, test_size, k_num, seed): 
    '''
    Shuffling을 진행한 뒤에 Meta-Train set과 Meta-Test set으로 분리한다. 
    이후에 Meta-Train set을 각각 Train / Validation으로 나눠 Z-score normalization을 진행하고 
    Meta-Test set을 각각 Kshot / Test으로 나눠 Z-Score normalization을 진행한다. 
    '''
    # Shuffling (이후에는 shuffling 하지 않음) 
    merged_df = pd.merge(df, pheno_with_iq, on='Subject')
    shuffled_df = merged_df.sample(frac=1, random_state=seed) 
    
    pheno = shuffled_df[pheno_with_iq.columns].to_numpy()
    df = shuffled_df[df.columns].to_numpy()
    
    
    # Meta-Train set (Train & Validation) / Meta-Test set (K-shot & Test)
    meta_train_df, meta_test_df, meta_train_pheno, meta_test_pheno = \
            train_test_split(df, pheno, test_size=test_size, random_state=seed, shuffle=False)

    # Train & Validation Split 
    train_df, val_df, train_pheno, val_pheno = \
            train_test_split(meta_train_df, meta_train_pheno, test_size=test_size, random_state=seed, shuffle=False)
    # K-Shot & Test split 
    kshot_df, test_df, kshot_pheno, test_pheno = \
            train_test_split(meta_test_df, meta_test_pheno,  train_size=k_num, random_state=seed, shuffle=False) 
    
    # Phenotype Scaling 
    meta_train_scaler = StandardScaler()
    meta_test_scaler = StandardScaler()
    
    meta_train_scaler.fit(train_pheno)
    train_pheno = meta_train_scaler.transform(train_pheno)
    val_pheno = meta_train_scaler.transform(val_pheno)
    
    meta_test_scaler.fit(kshot_pheno)
    kshot_pheno = meta_test_scaler.transform(kshot_pheno)
    test_pheno = meta_test_scaler.transform(test_pheno)
    
    return train_df, train_pheno, val_df, val_pheno, kshot_df, kshot_pheno, test_df, test_pheno


def get_dataloader(df, pheno, batch_size, generator, device): 
    df, pheno = (
        torch.Tensor(df).to(device), 
        torch.Tensor(pheno).to(device)
    )
    dataset= torch.utils.data.TensorDataset(df, pheno) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, generator=generator)

    return dataloader


def get_cod_score(predict_and_gt):
    X = predict_and_gt['IQ']
    y = predict_and_gt['prediction']
    # 절편(intercept)을 추가합니다.
    X = sm.add_constant(X)
    # OLS 모델을 만들고 fitting 합니다.
    model = sm.OLS(y, X).fit()
    # R-squared 값을 가져옵니다.
    r_squared = model.rsquared
    return r_squared


def get_corr_score(predict_and_gt):
    correlation = predict_and_gt['prediction'].corr(predict_and_gt['IQ'], method='pearson')
    return correlation


def save_iteration_loss_plot(train_loss_list, val_loss_list, loss_img_pth, seed):
    
    epochs = range(1, len(train_loss_list)+1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss_list, label='Train Loss', marker='o', linestyle='-')
    plt.plot(epochs, val_loss_list, label='Validation Loss', marker='o', linestyle='-')

    # 그래프에 레이블, 제목, 범례 등 추가
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(0, 1.5)
    plt.title(f'{seed}th Iteration : Train and Validation Loss Over Epochs')
    plt.legend()
    
    plt.savefig(loss_img_pth, dpi=300)
    

def mean_and_std(lst):
    absolute_values = [abs(x) for x in lst]
    mean = sum(absolute_values) / len(absolute_values)
    variance = sum((x - mean) ** 2 for x in absolute_values) / len(absolute_values)
    std_dev = math.sqrt(variance)
    return mean, std_dev


def correlation_viz(data_file_name, revised = False):
    dnn_df = pd.read_csv(f"D:/meta_matching_data/results/csv/{data_file_name}_dnn.csv", index_col=0)
    gcn_df = pd.read_csv(f"D:/meta_matching_data/results/csv/{data_file_name}_gcn.csv", index_col=0)
    dgcnn_df = pd.read_csv(f"D:/meta_matching_data/results/csv/{data_file_name}_dgcnn.csv", index_col=0)
    adst_df = pd.read_csv(f"D:/meta_matching_data/results/csv/{data_file_name}_adst.csv", index_col=0)
    if revised: 
        adft_df = pd.read_csv(f"D:/meta_matching_data/results/csv/{data_file_name}_adft_revised.csv", index_col=0)
    else: 
        adft_df = pd.read_csv(f"D:/meta_matching_data/results/csv/{data_file_name}_adft.csv", index_col=0)
    
    # cod_col_list = ['cods_k10','cods_k30','cods_k50','cods_k100']
    corr_col_list = ['corrs_k10','corrs_k30','corrs_k50','corrs_k100']

    
    dnn_corr = dnn_df.loc[:, corr_col_list]
    gcn_corr = gcn_df.loc[:, corr_col_list]
    dgcnn_corr = dgcnn_df.loc[:, corr_col_list]
    adft_corr = adft_df.loc[:, corr_col_list]
    adst_corr = adst_df.loc[:, corr_col_list]

    fig = plt.figure(figsize =(15, 8))
    plt.boxplot(dnn_corr, positions=[1,6,11,16], patch_artist = True,boxprops = dict(facecolor = "lightsteelblue"))
    plt.boxplot(gcn_corr, positions=[2,7,12,17],patch_artist = True,boxprops = dict(facecolor = "cornflowerblue"))
    plt.boxplot(dgcnn_corr, positions=[3,8,13,18], patch_artist = True, boxprops = dict(facecolor = "royalblue"))
    plt.boxplot(adft_corr, positions=[4,9,14,19],patch_artist = True,boxprops = dict(facecolor = "limegreen"))
    plt.boxplot(adst_corr, positions=[5,10,15,20], patch_artist = True, boxprops = dict(facecolor = "goldenrod"))

    # x축 라벨 설정
    plt.xticks([3,8,13,18], [10, 30, 50, 100], fontsize = 12)

    label_1 = mpatches.Patch(color='lightsteelblue', label='Basic DNN')
    label_2 = mpatches.Patch(color='cornflowerblue', label='Basic GCN')
    label_3 = mpatches.Patch(color='royalblue', label='Basic DGCNN')
    label_4 = mpatches.Patch(color='limegreen', label='Advanced Fine-tuning')
    label_5 = mpatches.Patch(color='goldenrod', label='Advanced Stacking')

    plt.legend(handles=[label_1, label_2, label_3, label_4, label_5], fontsize = 12, frameon = False)
    
    title = f"{data_file_name.split('_')[0].upper()} : {data_file_name.split('_')[1].upper()}"
    plt.title(title, fontsize = 20)
    plt.xlabel('Number of participants(K-shot)',fontsize=14)
    plt.ylabel('Prediction performance(correlation)', fontsize=14)
    plt.yticks(fontsize=12)
    if revised:
        plt.savefig(f'D:/meta_matching_data/results/plot/corr/{data_file_name}_revised_with_GNN.png')
    else:
        plt.savefig(f'D:/meta_matching_data/results/plot/corr/{data_file_name}_with_GNN.png')


def cod_viz(data_file_name, revised = False):
    dnn_df = pd.read_csv(f"D:/meta_matching_data/results/csv/{data_file_name}_dnn.csv", index_col=0)
    gcn_df = pd.read_csv(f"D:/meta_matching_data/results/csv/{data_file_name}_gcn.csv", index_col=0)
    dgcnn_df = pd.read_csv(f"D:/meta_matching_data/results/csv/{data_file_name}_dgcnn.csv", index_col=0)
    adst_df = pd.read_csv(f"D:/meta_matching_data/results/csv/{data_file_name}_adst.csv", index_col=0)
    if revised: 
        adft_df = pd.read_csv(f"D:/meta_matching_data/results/csv/{data_file_name}_adft_revised.csv", index_col=0)
    else: 
        adft_df = pd.read_csv(f"D:/meta_matching_data/results/csv/{data_file_name}_adft.csv", index_col=0)
    
    cod_col_list = ['cods_k10','cods_k30','cods_k50','cods_k100']

    dnn_cods = dnn_df.loc[:, cod_col_list]
    gcn_cods = gcn_df.loc[:, cod_col_list]
    dgcnn_cods = dgcnn_df.loc[:, cod_col_list]
    adft_cods = adft_df.loc[:, cod_col_list]
    adst_cods = adst_df.loc[:, cod_col_list]

    fig = plt.figure(figsize =(15, 8))
    plt.boxplot(dnn_cods, positions=[1,6,11,16], patch_artist = True,boxprops = dict(facecolor = "lightsteelblue"))
    plt.boxplot(gcn_cods, positions=[2,7,12,17],patch_artist = True,boxprops = dict(facecolor = "cornflowerblue"))
    plt.boxplot(dgcnn_cods, positions=[3,8,13,18], patch_artist = True, boxprops = dict(facecolor = "royalblue"))
    plt.boxplot(adft_cods, positions=[4,9,14,19],patch_artist = True,boxprops = dict(facecolor = "limegreen"))
    plt.boxplot(adst_cods, positions=[5,10,15,20], patch_artist = True, boxprops = dict(facecolor = "goldenrod"))

    # x축 라벨 설정
    plt.xticks([3,9,15,20], [10, 30, 50, 100], fontsize = 12)


    label_1 = mpatches.Patch(color='lightsteelblue', label='Basic DNN')
    label_2 = mpatches.Patch(color='cornflowerblue', label='Basic GCN')
    label_3 = mpatches.Patch(color='royalblue', label='Basic DGCNN')
    label_4 = mpatches.Patch(color='limegreen', label='Advanced Fine-tuning')
    label_5 = mpatches.Patch(color='goldenrod', label='Advanced Stacking')

    plt.legend(handles=[label_1, label_2, label_3, label_4, label_5], fontsize = 12, frameon = False)
    
    title = f"{data_file_name.split('_')[0].upper()} : {data_file_name.split('_')[1].upper()}"
    plt.title(title, fontsize = 20)
    plt.xlabel('Number of participants(K-shot)',fontsize=14)
    plt.ylabel('Prediction performance(COD)', fontsize=14)
    plt.yticks(fontsize=12)
    if revised:
        plt.savefig(f'D:/meta_matching_data/results/plot/cod/{data_file_name}_revised_with_GNN.png')
    else :
        plt.savefig(f'D:/meta_matching_data/results/plot/cod/{data_file_name}_with_GNN.png')


def viz_node_corr(data_file_name): 
    node_corr_df = pd.read_csv(f'D:/meta_matching_data/node_corr/{data_file_name}_node_corr.csv', index_col=0)
    plot_dir = f"D:/meta_matching_data/node_corr/plot/{data_file_name}/"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
        print(f"{plot_dir}을 생성하였습니다")
    
    col_list = node_corr_df.columns.tolist() 

    col_10 = []
    col_30 = []
    col_50 = []
    col_100 = []

    for column in col_list: 
        if column.split('_')[-1] == '10': 
            col_10.append(column)
        elif column.split('_')[-1] == '30': 
            col_30.append(column)
        elif column.split('_')[-1] == '50': 
            col_50.append(column)
        elif column.split('_')[-1] == '100': 
            col_100.append(column)
            
            
    # '10', '30', '50', '100'에 따라서 DataFrame을 나눕니다.
    df_10 = node_corr_df.loc[:, col_10]
    df_30 = node_corr_df.loc[:, col_30]
    df_50 = node_corr_df.loc[:, col_50]
    df_100 = node_corr_df.loc[:, col_100]


    df_10 = df_10.T.mean().values.reshape(1, -1) 
    # Heatmap을 생성합니다.
    plt.figure(figsize=(40, 1))  # 그림의 크기를 조절합니다.
    sns.heatmap(df_10, annot=True, fmt='.2f', cmap='Reds')
    plt.title(f"{data_file_name} K : 10 Node Correlation", y=4.03, fontsize=26, fontweight='bold')

    # 그래프를 표시합니다.
    plt.savefig(f"D:meta_matching_data/node_corr/plot/{data_file_name}/{data_file_name}_k10.png", dpi=300, bbox_inches='tight', pad_inches=0)
    
    df_30 = df_30.T.mean().values.reshape(1, -1) 
    # Heatmap을 생성합니다.
    plt.figure(figsize=(40, 1))  # 그림의 크기를 조절합니다.
    sns.heatmap(df_30, annot=True, fmt='.2f', cmap='Reds')
    plt.title(f"{data_file_name} K : 30 Node Correlation", y=4.03, fontsize=26, fontweight='bold')

    # 그래프를 표시합니다.
    plt.savefig(f"D:meta_matching_data/node_corr/plot/{data_file_name}/{data_file_name}_k30.png", dpi=300, bbox_inches='tight', pad_inches=0)
    
    
    df_50 = df_50.T.mean().values.reshape(1, -1) 
    # Heatmap을 생성합니다.
    plt.figure(figsize=(40, 1))  # 그림의 크기를 조절합니다.
    sns.heatmap(df_50, annot=True, fmt='.2f', cmap='Reds')
    plt.title(f"{data_file_name} K : 50 Node Correlation", y=4.03, fontsize=26, fontweight='bold')

    # 그래프를 표시합니다.
    plt.savefig(f"D:meta_matching_data/node_corr/plot/{data_file_name}/{data_file_name}_k50.png", dpi=300, bbox_inches='tight', pad_inches=0)
    
    
    df_100 = df_100.T.mean().values.reshape(1, -1) 
    # Heatmap을 생성합니다.
    plt.figure(figsize=(40, 1))  # 그림의 크기를 조절합니다.
    sns.heatmap(df_100, annot=True, fmt='.2f', cmap='Reds')
    plt.title(f"{data_file_name} K : 100 Node Correlation", y=4.03, fontsize=26, fontweight='bold')

    # 그래프를 표시합니다.
    plt.savefig(f"D:meta_matching_data/node_corr/plot/{data_file_name}/{data_file_name}_k100.png", dpi=300, bbox_inches='tight', pad_inches=0)
    
    


def save_results(cods, corrs, data_file_name, model_name): 
    #dnn
    df1 = pd.DataFrame(cods).T
    df2 = pd.DataFrame(corrs).T

    columns = ['cods_k10', 'cods_k30', 'cods_k50', 'cods_k100','corrs_k10', 'corrs_k30', 'corrs_k50', 'corrs_k100']
    df = pd.concat([df1, df2], axis=1)
    df.columns = columns
    df.to_csv(f'D:/meta_matching_data/results/csv/{data_file_name}_{model_name}.csv')
    print("Save Results Complete!")


def save_best_nodes(gcn_best_nodes, dgcnn_best_nodes, adft_best_nodes, adst_best_nodes, data_file_name):
    # dnn best node나 adft best node 같아서 dnn best node에 관한 변수 추가 하지 않음
    df1 = pd.DataFrame(gcn_best_nodes).T
    df2 = pd.DataFrame(dgcnn_best_nodes).T
    df3 = pd.DataFrame(adft_best_nodes).T
    df4 = pd.DataFrame(adst_best_nodes).T

    columns = ['gcn_k10', 'gcn_k30', 'gcn_k50','gcn_k100', 'dgcnn_k10', 'dgcnn_k30', 'dgcnn_k50','dgcnn_k100', 'adft_k10','adft_k30','adft_k50','adft_k100', 'adst_k10','adst_k30','adst_k50','adst_k100']
    best_nodes = pd.concat([df1, df2, df3, df4], axis=1)
    best_nodes.columns = columns
    best_nodes.to_csv(f'D:/meta_matching_data/results/csv/{data_file_name}_bestnodes.csv')
    print("Save Best Nodes Complete!")


def save_best_nodes_with_krr(krr_best_nodes, data_file_name):
    krr_node_df = pd.DataFrame(np.array(krr_best_nodes).T, columns=['krr_k10', 'krr_k30', 'krr_k50', 'krr_k100'])
    krr_node_df.to_csv(f'D:/meta_matching_data/results/csv/{data_file_name}_bestnodes_2.csv')
    print("Save Best Nodes ver 2 Complete!")


def save_k_val_cods(dnn_kshot_r2s, adft_kshot_r2s, data_file_name):
    dnn_k10, dnn_k30, dnn_k50, dnn_k100 = [], [], [], []
    adft_k10, adft_k30, adft_k50, adft_k100 = [], [], [], []
    for i in range(len(dnn_kshot_r2s)//4):
        dnn_k10.append(dnn_kshot_r2s[i*4+0])
        dnn_k30.append(dnn_kshot_r2s[i*4+1])
        dnn_k50.append(dnn_kshot_r2s[i*4+2]) 
        dnn_k100.append(dnn_kshot_r2s[i*4+3])
        adft_k10.append(adft_kshot_r2s[i*4+0])
        adft_k30.append(adft_kshot_r2s[i*4+1])
        adft_k50.append(adft_kshot_r2s[i*4+2]) 
        adft_k100.append(adft_kshot_r2s[i*4+3])
    dnn_kshot = [dnn_k10, dnn_k30, dnn_k50, dnn_k100]
    adft_kshot = [adft_k10, adft_k30, adft_k50, adft_k100]
       
    df1 = pd.DataFrame(dnn_kshot).T
    df2 = pd.DataFrame(adft_kshot).T

    columns = ['dnn_k10','dnn_k30','dnn_k50','dnn_k100', 'adft_k10','adft_k30','adft_k50','adft_k100']
    best_nodes = pd.concat([df1, df2], axis=1)
    best_nodes.columns = columns
    best_nodes.to_csv(f'D:/meta_matching_data/results/csv/{data_file_name}_validation_cod.csv')
    print("Save Validation Cods Complete!")


def save_revised_results(data_file_name):
    cod_df = pd.read_csv(f"D:/meta_matching_data/results/csv/{data_file_name}_validation_cod.csv", index_col = 0)
    dnn_df = pd.read_csv(f"D:/meta_matching_data/results/csv/{data_file_name}_dnn.csv", index_col = 0)
    adft_df = pd.read_csv(f"D:/meta_matching_data/results/csv/{data_file_name}_adft.csv", index_col = 0)

    cods_k30 = cod_df[cod_df['dnn_k30'] > cod_df['adft_k30']]
    for index, row in cods_k30.iterrows():
        adft_df.at[index, 'cods_k30'] = dnn_df.at[index, 'cods_k30']
        adft_df.at[index, 'corrs_k30'] = dnn_df.at[index, 'corrs_k30']

    cods_k50 = cod_df[cod_df['dnn_k50'] > cod_df['adft_k50']]
    for index, row in cods_k50.iterrows():
        adft_df.at[index, 'cods_k50'] = dnn_df.at[index, 'cods_k50']
        adft_df.at[index, 'corrs_k50'] = dnn_df.at[index, 'corrs_k50']

    cods_k100 = cod_df[cod_df['dnn_k100'] > cod_df['adft_k100']]
    for index, row in cods_k100.iterrows():
        adft_df.at[index, 'cods_k100'] = dnn_df.at[index, 'cods_k100']
        adft_df.at[index, 'corrs_k100'] = dnn_df.at[index, 'corrs_k100']

    adft_df.to_csv(f'D:/meta_matching_data/results/csv/{data_file_name}_adft_revised.csv')
    revised_adft_corrs = [list(adft_df['corrs_k10']), list(adft_df['corrs_k30']), list(adft_df['corrs_k50']), list(adft_df['corrs_k100'])]
    revised_adft_cods = [list(adft_df['cods_k10']), list(adft_df['cods_k30']), list(adft_df['cods_k50']), list(adft_df['cods_k100'])]

    return revised_adft_cods, revised_adft_corrs


def compute_KNN_graph(matrix, k_degree=10, metric='euclidean'): 
    """
    Calculate the adjacency matrix from the connectivity matrix.
    """

    dist = distance.pdist(matrix, metric) 
    dist = distance.squareform(dist)


    idx = np.argsort(dist)[:, 1:k_degree+1]
    dist.sort() 
    dist = dist[:, 1:k_degree+1]

    A = adjacency(dist, idx).astype(np.float32) 

    return A 


def adjacency(dist, idx): 
    
    m, k = dist.shape 
    assert m, k == idx.shape 
    assert dist.min() >= 0 

    # weights 
    sigma2 = np.mean(dist[:, -1])**2 
    dist = np.exp(-dist ** 2 / sigma2) 

    # weight matrix 
    I = np.arange(0, m).repeat(k) 
    J = idx.reshape(m * k) 
    V = dist.reshape(m * k) 
    W = coo_matrix((V, (I, J)), shape=(m, m))


    # No self-connections 
    W.setdiag(0)

    # Non-directed graph 
    bigger = W.T > W 
    W = W - W.multiply(bigger) + W.T.multiply(bigger) 

    assert W.nnz % 2 == 0 
    assert np.abs(W - W.T).mean() < 1e-10
    assert type(W) is csr.csr_matrix
    return W.todense()