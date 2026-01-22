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

        self.data_x = data[border1:border2, :-1]
        self.data_y = data[border1:border2, -1]
        
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        
        r_begin = s_end - 1
        r_end = r_begin + self.pred_len
        

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]

        return seq_x, seq_y
    
    def __len__(self):
        # 需要同时满足 seq_x 和 seq_y 的边界约束
        # seq_x 需要: index + seq_len <= len(data_x)
        # seq_y 需要: index + seq_len - 1 + pred_len <= len(data_y)
        return max(0, len(self.data_x) - self.seq_len - self.pred_len + 2)
        
