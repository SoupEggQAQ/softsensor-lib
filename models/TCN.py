import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.hidden_dim = configs.hidden_dim
        self.seq_len = configs.seq_len
        self.input_dim = configs.input_dim
        self.pred_len = configs.pred_len

        self.num_targets = getattr(configs, 'num_targets', 1)
        self.num_layers = getattr(configs, 'num_layers', 3)
        self.kernel_size = getattr(configs, 'kernel_size', 3)
        # configs.dropout 可能来自命令行解析为 str；Dropout 需要 float
        self.dropout = float(getattr(configs, 'dropout', 0.1))

        # 为了让后续 fc 的输入维度固定为 hidden_dim * seq_len，
        # 这里的 input_conv 必须保持时间长度不变（使用 padding + 裁剪实现因果卷积）
        self.input_padding = self.kernel_size - 1
        self.input_conv = nn.Conv1d(
            self.input_dim, 
            self.hidden_dim,
            self.kernel_size,
            padding=self.input_padding
        )

        self.tcn_layers = nn.ModuleList()
        dilation = 1
        for i in range(self.num_layers):
            self.tcn_layers.append(
                TemporalBlock(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim,
                    kernel_size=self.kernel_size,
                    dilation=dilation,
                    dropout=self.dropout
                )
            )
            dilation *= 2  # 指数级增加空洞率，扩大感受野
        
        # 输出层
        self.output_conv = nn.Conv1d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=1
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_dim * self.seq_len, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim * 2, self.pred_len * self.num_targets)
        )
    
    def forward(self, x):
        # 输入: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()
        
        # 转置以适应Conv1d: (batch_size, input_dim, seq_len)
        x = x.transpose(1, 2)
        
        # 输入卷积
        x = self.input_conv(x)
        x = x[:, :, :-self.input_padding] if self.input_padding > 0 else x
        
        # TCN层
        for tcn_layer in self.tcn_layers:
            x = tcn_layer(x)
        
        # 输出卷积
        x = self.output_conv(x)
        
        # 展平
        x = x.view(batch_size, -1)
        
        # 全连接层
        y_pred = self.fc(x)
        
        # 调整输出形状
        if self.num_targets > 1:
            y_pred = y_pred.reshape(batch_size, self.pred_len, self.num_targets)
        else:
            y_pred = y_pred.reshape(batch_size, self.pred_len, 1)
            
        return y_pred
    
    def predict(self, x):
        return self.forward(x)



class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super(TemporalBlock, self).__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.bn2 = nn.BatchNorm1d(out_channels)

        # 下采样层（如果输入输出通道数不同）
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        
         # dropout
        self.dropout = nn.Dropout(float(dropout))
        
        self.padding = padding

    def forward(self, x):
        identity = x
        
        # 第一层
        out = self.conv1(x)
        out = out[:, :, :-self.padding] if self.padding > 0 else out  # 裁剪多余的填充
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        # 第二层
        out = self.conv2(out)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.bn2(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        # 残差连接
        if self.downsample is not None:
            identity = self.downsample(identity)
        
        out += identity
        
        return out