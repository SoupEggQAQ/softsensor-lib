import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.hidden_dim = configs.hidden_dim
        self.seq_len = configs.seq_len
        self.input_dim = configs.input_dim
        self.pred_len = configs.pred_len
        self.num_targets = getattr(configs, 'num_targets', 1)


        self.lstm = nn.LSTM(
            self.input_dim,
            self.hidden_dim,
            num_layers=1
        )

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.pred_len * self.num_targets)
        )
    
    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        lstm_out, (h_n, c_n) = self.lstm(x)
        x_out_last = lstm_out[:, -1, :]
        y_pred = self.fc(x_out_last)

        if self.num_targets > 1:
            y_pred = y_pred.reshape(batch_size, self.pred_len, self.num_targets)
        else:
            y_pred = y_pred.reshape(batch_size, self.pred_len, -1)

        return y_pred
    
    def predict(self, x):
        y_pred = self.forward(x)
        return y_pred