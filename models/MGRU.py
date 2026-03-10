import torch.nn as nn
import torch
import math

# 多层GRU

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.input_dim = configs.input_dim
        self.hidden_dim = configs.hidden_dim
        self.pred_len = configs.pred_len
        self.num_targets = getattr(configs, 'num_targets', 1)

        self.encoder_gru_l1 = nn.GRU(self.input_dim, 
                                     self.hidden_dim,
                                     num_layers=1)
        self.encoder_gru_l2 = nn.GRU(self.hidden_dim,
                                     self.hidden_dim,
                                     num_layers=1)
        self.prediction_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.pred_len * self.num_targets)
        )
    
    def forward(self, x):
        # b: batch_size 
        # i: input_dim
        # s: seq_len

        x_gruf, _ = self.encoder_gru_l1(x)
        
        x_grus, _ = self.encoder_gru_l2(x_gruf)

        x = x_grus[:, -1, :]

        batch_size = x.shape[0]
        y_pred = self.prediction_layer(x)

        if self.num_targets > 1:
            y_pred = y_pred.reshape(batch_size, self.pred_len, self.num_targets)
        else:
            y_pred = y_pred.reshape(batch_size, self.pred_len, 1)
        return y_pred
    
    def predict(self, x):
        y = self.forward(x)
        return y