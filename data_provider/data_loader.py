import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, Optional, Union
import os
import pandas as pd

class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
        features='S', data_path='Debutanizer_data.txt', target='y',
        scale=True):

        self.args = args

        if size == None:
            self.seq_len = 1
            self.label_len = 1
            self.pred_len = 1
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        
        self.root_path = root_path
        self.data_path = data_path

        self.feature_columns = getattr(args, 'feature_columns', None)
        if self.feature_columns is not None:
            if isinstance(self.feature_columns, (int, str)):
                self.feature_columns = [int(self.feature_columns)]
            else:
                self.feature_columns = [int(col) for col in self.feature_columns]
   
        self.target_columns = getattr(args, 'target_columns', None)
        if self.target_columns is None:
            self.target_columns = [-1]
        elif isinstance(self.target_columns, (int, str)):
            self.target_columns = [int(self.target_columns)]
        else:
            self.target_columns = [int(col) for col in self.target_columns]
        
             
        self.__read_data__()
    
    def __read_data__(self):
        
        self.scaler = StandardScaler()

        local_fp = os.path.join(self.root_path, self.data_path)

        df_raw = pd.read_csv(local_fp, sep='\s+')
        dl = len(df_raw)
        num_train = int(dl * 0.7)
        num_test = int(dl * 0.2)
        num_vali = dl - num_train - num_test


        border1s = [0, num_train-self.seq_len, dl-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, dl]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # if self.features == 'S':
        #     return
        df_data = df_raw
        
        if self.scale:
            train_data = df_data[border1s[0]:border2s[1]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
        
        # features and targets
        num_cols = data.shape[1]
        target_indices = [col if col >= 0 else num_cols + col for col in self.target_columns]
        
        # features
        if self.feature_columns is not None:
            feature_indices = [col if col >= 0 else num_cols + col for col in self.feature_columns]
            self.data_x = data[border1:border2, feature_indices]
        else:
            all_indices = list(range(num_cols))
            feature_indices = [idx for idx in all_indices if idx not in target_indices]
            self.data_x = data[border1:border2, feature_indices]
 
        # targets
        if len(target_indices) == 1:
            self.data_y = data[border1:border2, target_indices[0]]
            if self.data_y.ndim == 1:
                self.data_y = self.data_y.reshape(-1, 1)
        else:
            self.data_y = data[border1:border2, target_indices]
               
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        
        r_begin = s_end - 1
        r_end = r_begin + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]  # [seq_len, input_dim]
        seq_y = self.data_y[r_begin:r_end]   # [pred_len, num_targets] 或 [pred_len]
        
        if seq_y.ndim == 1:
            seq_y = seq_y.reshape(-1, 1)

        return seq_x, seq_y
    
    def __len__(self):
        
        return max(0, len(self.data_x) - self.seq_len - self.pred_len + 2)
        
