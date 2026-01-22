from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
# from utils.tools import EarlyStopping, adjust_learning_rate, visual
# from utils.losses import mape_loss, mase_loss, smape_loss
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np


warnings.filters('ignore')

# 预测当前step

class Exp_Short_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Short_Term_Forecast).__init__(args)
    
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self):
        pass
    



    