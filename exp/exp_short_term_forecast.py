import data_provider
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# short_term_forecast

class Exp_Short_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Short_Term_Forecast, self).__init__(args)
    
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
    
    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)
                
                if outputs.dim() > batch_y.dim():
                    if hasattr(self.args, 'pred_len') and self.args.pred_len > 1:
                        outputs = outputs[:, -self.args.pred_len:]
                    else:
                        outputs = outputs.squeeze()
                
                if batch_y.dim() == 1:
                    batch_y = batch_y.unsqueeze(-1) if outputs.dim() > 1 else batch_y
                elif hasattr(self.args, 'pred_len') and self.args.pred_len > 1:
                    batch_y = batch_y[:, -self.args.pred_len:]
                    # 单目标多步预测时，标签通常形状为 [batch, pred_len, 1]，压缩最后一维以匹配输出
                    if batch_y.dim() == 3 and batch_y.shape[-1] == 1 and outputs.dim() == 2:
                        batch_y = batch_y.squeeze(-1)

                pred = outputs.detach()
                true = batch_y.detach()

                loss = criterion(pred, true)
                total_loss.append(loss.item())
        
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss
    
    
    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)
                
                if outputs.dim() > batch_y.dim():
                    if hasattr(self.args, 'pred_len') and self.args.pred_len > 1:
                        outputs = outputs[:, -self.args.pred_len:]
                    else:
                        outputs = outputs.squeeze()
                
                if batch_y.dim() == 1:
                    batch_y = batch_y.unsqueeze(-1) if outputs.dim() > 1 else batch_y
                elif hasattr(self.args, 'pred_len') and self.args.pred_len > 1:
                    batch_y = batch_y[:, -self.args.pred_len:]
                    # 单目标多步预测：将 [batch, pred_len, 1] 压缩为 [batch, pred_len] 与输出对齐
                    if batch_y.dim() == 3 and batch_y.shape[-1] == 1 and outputs.dim() == 2:
                        batch_y = batch_y.squeeze(-1)

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                if (i + 1) % 10 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model   
    

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)
                
                # 确保输出和目标维度匹配
                if outputs.dim() > batch_y.dim():
                    if hasattr(self.args, 'pred_len') and self.args.pred_len > 1:
                        outputs = outputs[:, -self.args.pred_len:]
                    else:
                        outputs = outputs.squeeze()
                
                if batch_y.dim() == 1:
                    batch_y = batch_y.unsqueeze(-1) if outputs.dim() > 1 else batch_y
                elif hasattr(self.args, 'pred_len') and self.args.pred_len > 1:
                    batch_y = batch_y[:, -self.args.pred_len:]

                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                
                # 数据反标准化
                if hasattr(test_data, 'scale') and test_data.scale and hasattr(self.args, 'inverse') and self.args.inverse:
                    if hasattr(test_data, 'inverse_transform'):
                        shape = batch_y.shape
                        # 处理维度不匹配的情况
                        if outputs.shape[-1] != batch_y.shape[-1]:
                            outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                        outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                
                # 可视化 暂无
                
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape before reshape:', preds.shape, trues.shape)
        
        # 
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
            trues = trues.reshape(-1, 1)
        elif preds.ndim == 2:
            
            pass
        elif preds.ndim >= 3:
            
            preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
            trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        
        print('test shape after reshape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        #

        # 
        preds_flat = preds.flatten()
        trues_flat = trues.flatten()

        if self.args.pred_len == 1:
            r2, mae, mse, rmse = metric(preds_flat, trues_flat, self.args.pred_len)
            print('r2:{:.6f}, mse:{:.6f}, mae:{:.6f}, rmse:{:.6f}'.format(r2, mse, mae, rmse))

        else:
            mae, mse, rmse = metric(preds_flat, trues_flat, self.args.pred_len)
            print('mse:{:.6f}, mae:{:.6f}, rmse:{:.6f}'.format(mse, mae, rmse))
        
        # 
        result_file = "result_softsensor_forecast.txt"
        f = open(result_file, 'a')
        f.write(setting + "  \n")

        if self.args.pred_len == 1:
            f.write('r2:{:.6f}, mse:{:.6f}, mae:{:.6f}, rmse:{:.6f}\n'.format(r2, mse, mae, rmse))
        else:
            f.write('mse:{:.6f}, mae:{:.6f}, rmse:{:.6f}\n'.format(mse, mae, rmse))
        f.write('\n')
        f.close()

        # 
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return