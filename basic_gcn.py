import numpy as np 
import pandas as pd 
import os
import torch 
from torch import nn 
import torch.nn.functional as func 
from utils import *
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset# , DataLoader 
from torch_geometric.utils import dense_to_sparse 
from torch_geometric.utils import remove_self_loops
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import Conv1d, MaxPool1d, Linear, Dropout 
from torch_geometric.nn import GCNConv, global_mean_pool, SortAggregation 



class ConnectivityData(Dataset): 
    """Dataset for the connectivity data"""

    def __init__(self, connectivity, labels): 
        # labels는 최종적으로 맞히고자 하는 Feature는 이미 제외된 상태로 Dataset에 입력되었다고 가정 
        # connectivity = np.load("./data/functional_restingstate.npy")
        # labels = np.genfromtxt("./data/pheno_with_iq.csv", delimiter=',', skip_header=1) 
        # labels = labels[:, 1:-1] # -> IQ를 제외 
        # labels = np.delete(labels, 4, axis=1)

        self.data_list = []
        N = 400 

        for i, y in enumerate(labels): 
            y = torch.tensor([y]).float() 
            c = np.zeros((N, N)) 
            c[np.triu_indices(N, k=1)] = connectivity[i]
            c[np.tril_indices(N, k=-1)] = connectivity[i]


            # with np.errstate(divide='ignore', invalid='ignore'): 
                # c = np.arctanh(c) 

            x = torch.from_numpy(c).float()

            adj = compute_KNN_graph(c) 
            adj = torch.from_numpy(adj).float() 
            edge_index, edge_attr = dense_to_sparse(adj) 

            self.data_list.append(Data(x = x, edge_index=edge_index, edge_attr=edge_attr, y=y)) 

    def __len__(self): 
        return len(self.data_list) 
    
    def __getitem__(self, idx):
        return self.data_list[idx]


class GCN(torch.nn.Module): 
    def __init__(
            self, 
            num_features, 
            num_classes, 
            dropout=.3):
        super(GCN, self).__init__() 

        self.p = dropout 

        self.conv1 = GCNConv(int(num_features), 64) 
        self.conv2 = GCNConv(64, 128)
        self.conv3 = GCNConv(128, 128)

        self.lin1 = torch.nn.Linear(128,64)
        self.lin2 = torch.nn.Linear(64, int(num_classes))
        
        
    def forward(self, data): 
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr 
        batch = data.batch 

        x = func.relu(self.conv1(x, edge_index, edge_attr))
        x = func.dropout(x, p=self.p, training=self.training) 
        x = func.relu(self.conv2(x, edge_index, edge_attr))
        x = func.dropout(x, p=self.p, training=self.training) 
        x = func.relu(self.conv3(x, edge_index, edge_attr)) 


        x = global_mean_pool(x, batch)
        x = func.relu(self.lin1(x))
        x = self.lin2(x)
        return x 
    

class DGCNN(torch.nn.Module): 
    def __init__(self, 
                 num_features, 
                 num_classes): 
        super(DGCNN, self).__init__() 


        self.conv1 = GCNConv(num_features, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32) 
        self.conv4 = GCNConv(32, 1) 

        self.sort_pool = SortAggregation(k=30)
        self.conv5 = Conv1d(1,16,97,97)
        self.conv6 = Conv1d(16,32,5,1)
        self.pool = MaxPool1d(2,2)
        self.classifier_1 = Linear(352,256)
        self.drop_out = Dropout(0.5)
        self.classifier_2 = Linear(256, num_classes)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, data): 
        x, edge_index, batch = data.x, data.edge_index, data.batch 
        edge_index, _ = remove_self_loops(edge_index) 


        x_1 = torch.tanh(self.conv1(x, edge_index))
        x_2 = torch.tanh(self.conv2(x_1, edge_index))
        x_3 = torch.tanh(self.conv3(x_2, edge_index))
        x_4 = torch.tanh(self.conv4(x_3, edge_index))
        x = torch.cat([x_1, x_2, x_3, x_4], dim=-1)
        x = self.sort_pool(x, batch) 
        x = x.view(x.size(0), 1, x.size(-1)) 
        x = self.relu(self.conv5(x)) 
        x = self.pool(x) 
        x = self.relu(self.conv6(x)) 
        x = x.view(x.size(0), -1) 
        out= self.relu(self.classifier_1(x))
        out = self.drop_out(out) 
        classes = self.classifier_2(out)

        return classes





def GCN_train(model, dataset, loader, optimizer, device): 
    model.train()

    train_loss_all = 0 
    for data in loader: 
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        train_loss = func.mse_loss(output.to(torch.float32), data.y.to(torch.float32)) 
        train_loss.backward() 
        train_loss_all += data.num_graphs * train_loss.item() 
        optimizer.step() 
    # scheduler.step()

    return train_loss_all / len(dataset) 


def GCN_val(model, dataset, loader, device) : 
    model.eval() 
    val_loss_all = 0 
    for data in loader: 
        data = data.to(device) 
        output = model(data) 
        val_loss = func.mse_loss(output.to(torch.float32), data.y.to(torch.float32)) 
        val_loss_all += data.num_graphs * val_loss.item() 


    return val_loss_all / len(dataset) 


def GCN_test(model, test_loader, test_iq, max_cod_idx, device): 
    model.eval() 
    test_loss_all = 0 

    test_outputs = []
    with torch.no_grad():
        for data in test_loader: 
            data = data.to(device)
            output = model(data) 
            test_outputs.append(output)
            # test_loss = func.mse_loss(output.to(torch.float32), data.y.to(torch.float32))
            # test_loss_all += data.num_graphs * test_loss.item() 

    test_outputs = torch.cat(test_outputs).cpu().detach().numpy()
    test_outputs = test_outputs[:, max_cod_idx].reshape(-1, 1) 
    pred_df = pd.DataFrame({'prediction': test_outputs.flatten(), 'IQ':test_iq.flatten()})

    cod = get_cod_score(pred_df) 
    corr = get_corr_score(pred_df) 



    return corr, cod 


def get_gcn_kshot_idx(model, kshot_loader, kshot_iq): 
    model.eval() 
    kshot_outputs = []

    with torch.no_grad(): 
        for data in kshot_loader: 
            data = data.to('cuda')
            output = model(data)
            kshot_outputs.append(output) 

    kshot_outputs = torch.cat(kshot_outputs).cpu().detach().numpy().T
    kshot_iq_cods = []

    for i in range(len(kshot_outputs)): 
        pred = pd.DataFrame({'prediction':kshot_outputs[i], 'IQ':kshot_iq})
        kshot_iq_cods.append(get_cod_score(pred))


    max_cod_idx = kshot_iq_cods.index(max(kshot_iq_cods))
    return max_cod_idx


def basic_gcn(df, pheno_with_iq, k_num_list, data_file_name, iteration, only_test=False): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_size=0.2 
    k_num=10 # 임의로 이 값을 지정해준 것 (실질적으로 사용 안됨)
    num_features = 400 
    num_classes = 58 
    
    if not only_test: 
        for seed in range(1, iteration+1): 
            model_pth = f"D:/meta_matching_data/model_pth/{data_file_name}/{seed}_gcn_adamw_{data_file_name}.pth"

            # Training 
            print(f'==========================================Iter{seed}==========================================')
            # Random Seed Setting 
            set_random_seeds(seed) 
            generator = torch.Generator() 
            generator.manual_seed(seed) 

            model = GCN(num_features, num_classes).to(device) 
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-03, weight_decay=0.4)
            scheduler = CosineAnnealingLR(optimizer, T_max=20)

            train_df, train_pheno, val_df, val_pheno, _, _, _, _ = \
                                        preprocess_data(df, pheno_with_iq, test_size, k_num, seed)
            train_pheno = train_pheno[:, :-1]
            val_pheno = val_pheno[:, :-1]

            train_dataset = ConnectivityData(train_df, train_pheno)
            val_dataset = ConnectivityData(val_df, val_pheno)

            train_loader = DataLoader(train_dataset, batch_size=64)
            val_loader = DataLoader(val_dataset, batch_size=64)


            train_losses = []
            val_losses = []
            min_v_loss = np.inf 

            for epoch in range(200):
                train_loss = GCN_train(model, train_dataset, train_loader, optimizer, device)
                val_loss = GCN_val(model, val_dataset, val_loader, device)
                train_losses.append(train_loss) 
                val_losses.append(val_losses)
                scheduler.step()

                if min_v_loss > val_loss: 
                    min_v_loss = val_loss 
                    torch.save(model.state_dict(), model_pth)

                print("Epoch : {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f}".format(epoch+1, train_loss, val_loss))

        print('==========================================학습을 완료하였습니다.==========================================')
        print('\n\n')
    else: 
        print("학습을 건너뜁니다.")

    
    # 예측 성능을 기록할 Dictionary Intialization 
    corr_dict = {'10':[], '30':[], '50':[], '100':[]} 
    cod_dict = {'10':[], '30':[], '50':[], '100':[]} 
    best_node_dict = {'10':[], '30':[], '50':[], '100':[]} 
    
    # K-shot / Test 성능 측정 
    for seed in range(1, iteration+1): 
        set_random_seeds(seed)
        generator = torch.Generator() 
        generator.manual_seed(seed) 

        # 학습 완료된 모델 Load 
        model_pth = f"D:/meta_matching_data/model_pth/{data_file_name}/{seed}_gcn_adamw_{data_file_name}.pth"
        model = GCN(num_features, num_classes)
        state_dict = torch.load(model_pth)
        model.load_state_dict(state_dict)
        model.to(device)


        for k_num in k_num_list: 
            print(f"==========================================K : {k_num}==========================================")
            _, _, _, _, kshot_df, kshot_pheno, test_df, test_pheno= preprocess_data(df, pheno_with_iq, test_size, k_num, seed)

            test_iq = test_pheno[:, -1]
            test_pheno = test_pheno[:, :-1]
            kshot_iq = kshot_pheno[:, -1]
            kshot_pheno = kshot_pheno[:, :-1]

            test_dataset = ConnectivityData(test_df, test_pheno)
            kshot_dataset = ConnectivityData(kshot_df, kshot_pheno)

            test_loader = DataLoader(test_dataset, batch_size=test_pheno.shape[0])
            kshot_loader = DataLoader(kshot_dataset, batch_size=k_num)

            # max_cod_idx를 K Samples로 계산 
            max_cod_idx = get_gcn_kshot_idx(model, kshot_loader, kshot_iq)
            test_corr, test_cod = GCN_test(model, test_loader, test_iq, max_cod_idx, device)

            corr_dict[str(k_num)].append(test_corr)
            cod_dict[str(k_num)].append(test_cod)
            best_node_dict[str(k_num)].append(max_cod_idx)
            print(f"Iteration {seed} | K = {k_num} : Corr - {test_corr:.4f} & R2 - {test_cod:.4f}")
        print('\n\n')


    for key in k_num_list: 
        if len(cod_dict[str(key)]) != 0: 
            print(f"K={str(key)} : Average COD : {np.mean(cod_dict[str(key)])}")
            print(f"K={str(key)} : STD     COD : {np.std(cod_dict[str(key)])}")
            print()
            print(f"K={str(key)} : Average Corr : {np.mean(corr_dict[str(key)])}")
            print(f"K={str(key)} : STD     Corr : {np.std(corr_dict[str(key)])}")
            print('\n\n')    
        
    return corr_dict, cod_dict, best_node_dict
    
    
def basic_dgcnn(df, pheno_with_iq, k_num_list, data_file_name, iteration, only_test=False): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_size=0.2 
    k_num=10 # 임의로 이 값을 지정해준 것 (실질적으로 사용 안됨)
    num_features = 400 
    num_classes = 58 
    
    if not only_test: 
        for seed in range(1, iteration+1): 
            model_pth = f"D:/meta_matching_data/model_pth/{data_file_name}/{seed}_dgcnn_adamw_{data_file_name}.pth"

            # Training 
            print(f'==========================================Iter{seed}==========================================')
            # Random Seed Setting 
            set_random_seeds(seed) 
            generator = torch.Generator() 
            generator.manual_seed(seed) 

            model = DGCNN(num_features, num_classes).to(device) 
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-03, weight_decay=0.4)
            scheduler = CosineAnnealingLR(optimizer, T_max=20)

            train_df, train_pheno, val_df, val_pheno, _, _, _, _ = \
                                        preprocess_data(df, pheno_with_iq, test_size, k_num, seed)
            train_pheno = train_pheno[:, :-1]
            val_pheno = val_pheno[:, :-1]

            train_dataset = ConnectivityData(train_df, train_pheno)
            val_dataset = ConnectivityData(val_df, val_pheno)

            train_loader = DataLoader(train_dataset, batch_size=64)
            val_loader = DataLoader(val_dataset, batch_size=64)


            train_losses = []
            val_losses = []
            min_v_loss = np.inf 

            for epoch in range(200):
                train_loss = GCN_train(model, train_dataset, train_loader, optimizer, device)
                val_loss = GCN_val(model, val_dataset, val_loader, device)
                train_losses.append(train_loss) 
                val_losses.append(val_losses)
                scheduler.step()

                if min_v_loss > val_loss: 
                    min_v_loss = val_loss 
                    torch.save(model.state_dict(), model_pth)

                print("Epoch : {:03d}, Train Loss: {:.5f}, Val Loss: {:.5f}".format(epoch+1, train_loss, val_loss))

        print('==========================================학습을 완료하였습니다.==========================================')
        print('\n\n')
    else: 
        print("학습을 건너뜁니다.")


    # 예측 성능을 기록할 Dictionary Intialization 
    corr_dict = {'10':[], '30':[], '50':[], '100':[]} 
    cod_dict = {'10':[], '30':[], '50':[], '100':[]} 
    best_node_dict = {'10':[], '30':[], '50':[], '100':[]} 
    
    # K-shot / Test 성능 측정 
    for seed in range(1, iteration+1): 
        set_random_seeds(seed)
        generator = torch.Generator() 
        generator.manual_seed(seed) 

        # 학습 완료된 모델 Load 
        model_pth = f"D:/meta_matching_data/model_pth/{data_file_name}/{seed}_dgcnn_adamw_{data_file_name}.pth"
        model = DGCNN(num_features, num_classes)
        state_dict = torch.load(model_pth)
        model.load_state_dict(state_dict)
        model.to(device)

        for k_num in k_num_list: 
            print(f"==========================================K : {k_num}==========================================")
            _, _, _, _, kshot_df, kshot_pheno, test_df, test_pheno= preprocess_data(df, pheno_with_iq, test_size, k_num, seed)

            test_iq = test_pheno[:, -1]
            test_pheno = test_pheno[:, :-1]
            kshot_iq = kshot_pheno[:, -1]
            kshot_pheno = kshot_pheno[:, :-1]

            test_dataset = ConnectivityData(test_df, test_pheno)
            kshot_dataset = ConnectivityData(kshot_df, kshot_pheno)

            test_loader = DataLoader(test_dataset, batch_size=test_pheno.shape[0])
            kshot_loader = DataLoader(kshot_dataset, batch_size=k_num)

            # max_cod_idx를 K Samples로 계산 
            max_cod_idx = get_gcn_kshot_idx(model, kshot_loader, kshot_iq)
            test_corr, test_cod = GCN_test(model, test_loader, test_iq, max_cod_idx, device)


            corr_dict[str(k_num)].append(test_corr)
            cod_dict[str(k_num)].append(test_cod)
            best_node_dict[str(k_num)].append(max_cod_idx)     
            print(f"Iteration {seed} | K = {k_num} : Corr - {test_corr:.4f} & R2 - {test_cod:.4f}")
        print('\n\n')

    for key in k_num_list: 
        if len(cod_dict[str(key)]) != 0: 
            print(f"K={str(key)} : Average COD : {np.mean(cod_dict[str(key)])}")
            print(f"K={str(key)} : STD     COD : {np.std(cod_dict[str(key)])}")
            print()
            print(f"K={str(key)} : Average Corr : {np.mean(corr_dict[str(key)])}")
            print(f"K={str(key)} : STD     Corr : {np.std(corr_dict[str(key)])}")
            print('\n\n')   
            
        
        
    return corr_dict, cod_dict, best_node_dict
    

    