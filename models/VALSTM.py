import torch.nn as nn
import torch
import math
import sklearn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.input_dim = configs.input_dim
        self.hidden_dim = configs.hidden_dim
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.num_targets = getattr(configs, 'num_targets', 1)

        # LSTMCell
        self.lstm_cell = nn.LSTMCell(self.input_dim, self.hidden_dim)

        # 注意力机制参数
        self.Wa = nn.Parameter(torch.Tensor(self.input_dim, self.input_dim), requires_grad=True)
        self.Ua = nn.Parameter(torch.Tensor(self.hidden_dim * 2, self.input_dim), requires_grad=True)
        self.ba = nn.Parameter(torch.Tensor(self.input_dim), requires_grad=True)
        self.Va = nn.Parameter(torch.Tensor(self.input_dim, self.input_dim), requires_grad=True)
        self.Softmax = nn.Softmax(dim=1)

        # 全连接层做预测
        self.fc = nn.Linear(self.hidden_dim, self.pred_len * self.num_targets, bias=True)

        self.init_weights()
    
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_dim)
        
        for weight in [self.Wa, self.Ua, self.Va]:
            weight.data.uniform_(-stdv, stdv)
        self.ba.data.zero_()
        
    
    def forward(self, x, init_states=None):
        # b: batch_size 
        # i: input_dim
        # s: seq_len
        batch_size, seq_len, input_dim = x.size()
        
        # 初始化隐藏状态和cell状态
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
            c_t = torch.zeros(batch_size, self.hidden_dim, device=x.device, dtype=x.dtype)
        else:
            h_t, c_t = init_states
        
        
        hidden_seq = []
        for t in range(seq_len):
            
            x_t = x[:, t, :]
            
            a_t = torch.tanh(x_t @ self.Wa + torch.cat((h_t, c_t), dim=1) @ self.Ua + self.ba) @ self.Va
            
            alpha_t = self.Softmax(a_t)
            
            x_t_attended = alpha_t * x_t
            
            h_t, c_t = self.lstm_cell(x_t_attended, (h_t, c_t))
            
            hidden_seq.append(h_t.unsqueeze(1))
        
        
        hidden_seq = torch.cat(hidden_seq, dim=1)
        
        final_feature = hidden_seq[:, -1, :]  # [batch_size, hidden_dim]
        
        y_pred = self.fc(final_feature)  # [batch_size, pred_len]
        
        if self.num_targets > 1:
            y_pred = y_pred.reshape(batch_size, self.pred_len, self.num_targets)
        else:
            y_pred = y_pred.reshape(batch_size, self.pred_len, 1)

        
        return y_pred
    
    def predict(self, x):
        
        y_pred = self.forward(x)
        return y_pred
    

