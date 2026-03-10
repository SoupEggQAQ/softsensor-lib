import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()

        self.hidden_dim = configs.hidden_dim
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.input_dim = configs.input_dim
        self.num_targets = getattr(configs, 'num_targets', 1)

        self.bidirectional = configs.bidirectional
        self.dir_mult = configs.dir_mult
        self.attention_type = configs.attention_type

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1
        )

        if self.attention_type == 'scaled_dot':
            self.scale = math.sqrt(self.hidden_dim * self.dir_mult)
        else:
            self.attn = nn.Sequential(
                nn.Linear(self.hidden_dim * self.dir_mult * 2, self.hidden_dim),
                nn.Tanh(),
                nn.Linear(self.hidden_dim, 1)
            )

        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim * self.dir_mult, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.pred_len * self.num_targets)
        )

    def forward(self, x):

        batch_size, seq_len, _ = x.size()
        
        lstm_out, (h_n, c_n) = self.lstm(x)

        if self.attention_type == 'scaled_dot':
            scores = torch.bmm(lstm_out, lstm_out.transpose(1, 2)) / self.scale
            attn_weights = F.softmax(scores, dim=-1)
            context = torch.bmm(attn_weights, lstm_out)
        
        elif self.attention_type == 'additive':
            seq_len = lstm_out.size(1)
            attn_weights = []
            for t in range(seq_len):
                query = h_n[-1].unsqueeze(1), # (batch_size, 1, hidden_dim)
                key = lstm_out[:, t, :].unsqueeze(2) # (batch_size, hidden_dim, 1)
                energy = self.attn(torch.cat([query, key], dim=1))
                attn_weights.append(energy)
            
            attn_weights = F.softmax(torch.cat(attn_weights, dim=1), dim=1)
            context = torch.sum(lstm_out * attn_weights, dim=1)

        output = self.fc(context)

        y_pred = output[:, -1, :]
        if self.num_targets > 1:
            y_pred = y_pred.reshape(batch_size, self.pred_len, self.num_targets)
        else:
            y_pred = y_pred.reshape(batch_size, self.pred_len, 1)
            
        return y_pred
    
    def predict(self, x):
        y_pred = self.forward(x)
        return y_pred
        