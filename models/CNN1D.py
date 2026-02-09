import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.hidden_dim = configs.hidden_dim
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.input_dim = configs.input_dim
        self.num_targets = getattr(configs, 'num_targets', 1)

        self.conv1 = nn.Conv1d(
            in_channels=self.seq_len,
            out_channels=self.hidden_dim,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.hidden_dim * 2,
            kernel_size=3,
            padding=1
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.pred_len * self.num_targets)
        )
    
    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        x = x.transpose(1, 2)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = self.global_pool(x)
        x = x.squeeze(-1)

        y_pred = self.fc(x)

        if self.num_targets > 1:
            y_pred = y_pred.reshape(batch_size, self.pred_len, self.num_targets)
        else:
            y_pred = y_pred.reshape(batch_size, self.pred_len)
            
        return y_pred
    
    def predict(self, x):
        return self.forward(x)
