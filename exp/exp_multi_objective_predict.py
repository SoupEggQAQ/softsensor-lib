import os
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping
from utils.metrics import metric, MAE, MSE, RMSE, R2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')

# multi_objective_predict

class Exp_Softsensor_Multi_Objective(Exp_Basic):
    def __init__(self, args):
        super(Exp_Softsensor_Multi_Objective, self).__init__(args)
        self.num_targets = getattr(args, 'num_targets', 1)
        self.target_weights = getattr(args, 'target_weights', [1.0] * self.num_targets)
        if len(self.target_weights) != self.num_targets:
            self.target_weights = [1.0] * self.num_targets
        self.loss_type = getattr(args, 'multi_target_loss_type', 'weighted_sum')
    
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
        
        if self.loss_type == 'independent':
            
            return [nn.MSELoss() for _ in range(self.num_targets)]
        else:
            
            return nn.MSELoss(reduction='none')  
    
    def _compute_multi_target_loss(self, outputs, batch_y, criterion):
        """
        Args:
            outputs: [batch, pred_len, num_targets] 或 [batch, num_targets]
            batch_y: [batch, pred_len, num_targets] 或 [batch, num_targets]
            criterion: 损失函数
        Returns:
            total_loss: 总损失
            individual_losses: 每个目标的损失列表
        """
        
        if outputs.dim() == 2 and batch_y.dim() == 2:
            
            if outputs.shape[-1] == self.num_targets:
                
                outputs = outputs.unsqueeze(1) if outputs.dim() == 2 else outputs  
                batch_y = batch_y.unsqueeze(1) if batch_y.dim() == 2 else batch_y
            else:
                # [batch, pred_len] -> [batch, pred_len, 1]
                outputs = outputs.unsqueeze(-1)
                batch_y = batch_y.unsqueeze(-1)
        
        
        if outputs.dim() == 2:
            outputs = outputs.unsqueeze(1)
        if batch_y.dim() == 2:
            batch_y = batch_y.unsqueeze(1)
        
        
        if outputs.shape[1] > 1:
            outputs = outputs.mean(dim=1)  
            batch_y = batch_y.mean(dim=1)  
        else:
            outputs = outputs.squeeze(1)  
            batch_y = batch_y.squeeze(1)  
        
        if self.loss_type == 'independent':
            
            individual_losses = []
            for i in range(self.num_targets):
                loss = criterion[i](outputs[:, i], batch_y[:, i])
                individual_losses.append(loss)
            
            total_loss = sum(w * loss for w, loss in zip(self.target_weights, individual_losses))
        else:
            
            # outputs: [batch, num_targets], batch_y: [batch, num_targets]
            loss_per_target = criterion(outputs, batch_y)  # [batch, num_targets]
            
            individual_losses = [loss_per_target[:, i].mean() for i in range(self.num_targets)]
            total_loss = sum(w * loss for w, loss in zip(self.target_weights, individual_losses))
        
        return total_loss, individual_losses

    def vali(self, vali_data, vali_loader, criterion):
        
        total_loss = []
        individual_losses_list = [[] for _ in range(self.num_targets)]
        
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
                
                loss, individual_losses = self._compute_multi_target_loss(outputs, batch_y, criterion)
                total_loss.append(loss.item())
                for j, ind_loss in enumerate(individual_losses):
                    individual_losses_list[j].append(ind_loss.item() if isinstance(ind_loss, torch.Tensor) else ind_loss)
        
        avg_total_loss = np.average(total_loss)
        avg_individual_losses = [np.average(losses) for losses in individual_losses_list]
        
        self.model.train()
        return avg_total_loss, avg_individual_losses

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
            train_individual_losses = [[] for _ in range(self.num_targets)]

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
                        loss, individual_losses = self._compute_multi_target_loss(outputs, batch_y, criterion)
                else:
                    outputs = self.model(batch_x)
                    loss, individual_losses = self._compute_multi_target_loss(outputs, batch_y, criterion)
                
                train_loss.append(loss.item())
                for j, ind_loss in enumerate(individual_losses):
                    train_individual_losses[j].append(ind_loss.item() if isinstance(ind_loss, torch.Tensor) else ind_loss)

                if (i + 1) % 10 == 0:
                    loss_str = ', '.join([f'target_{j}: {np.mean(train_individual_losses[j][-10:]):.7f}' 
                                         for j in range(self.num_targets)])
                    print("\titers: {0}, epoch: {1} | total_loss: {2:.7f} | {3}".format(
                        i + 1, epoch + 1, loss.item(), loss_str))
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
            train_individual = [np.average(losses) for losses in train_individual_losses]
            
            vali_loss, vali_individual = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_individual = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            print("  Individual Train Losses: {}".format([f'{l:.7f}' for l in train_individual]))
            print("  Individual Vali Losses: {}".format([f'{l:.7f}' for l in vali_individual]))
            
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
                
                
                if outputs.dim() == 2:
                    if outputs.shape[-1] == self.num_targets:
                        # [batch, num_targets]
                        outputs = outputs.unsqueeze(1)  # [batch, 1, num_targets]
                    else:
                        # [batch, pred_len] -> [batch, pred_len, 1]
                        outputs = outputs.unsqueeze(-1)
                
                if batch_y.dim() == 2:
                    if batch_y.shape[-1] == self.num_targets:
                        batch_y = batch_y.unsqueeze(1)
                    else:
                        batch_y = batch_y.unsqueeze(-1)
                
                
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                
                # 反标准化
                if hasattr(test_data, 'scale') and test_data.scale and hasattr(self.args, 'inverse') and self.args.inverse:
                    if hasattr(test_data, 'inverse_transform'):
                        shape = batch_y.shape
                        outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                preds.append(outputs)
                trues.append(batch_y)
        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        
        # [samples, pred_len, num_targets]
        if preds.ndim == 2:
            if preds.shape[-1] == self.num_targets:
                preds = preds.reshape(-1, 1, self.num_targets)
                trues = trues.reshape(-1, 1, self.num_targets)
            else:
                preds = preds.reshape(-1, preds.shape[-1], 1)
                trues = trues.reshape(-1, trues.shape[-1], 1)
        
        print('test shape:', preds.shape, trues.shape)
        
        # 为每个目标计算指标
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # 计算每个目标的指标
        all_metrics = {}
        for target_idx in range(self.num_targets):
            pred_target = preds[:, :, target_idx].flatten()
            true_target = trues[:, :, target_idx].flatten()
            
            mae = MAE(pred_target, true_target)
            mse = MSE(pred_target, true_target)
            rmse = RMSE(pred_target, true_target)
            
            if self.args.pred_len == 1:
                r2 = R2(pred_target, true_target)
                all_metrics[target_idx] = {'r2': r2, 'mae': mae, 'mse': mse, 'rmse': rmse}
                print('Target {0}: r2:{1:.6f}, mse:{2:.6f}, mae:{3:.6f}, rmse:{4:.6f}'.format(
                    target_idx, r2, mse, mae, rmse))
            else:
                all_metrics[target_idx] = {'mae': mae, 'mse': mse, 'rmse': rmse}
                print('Target {0}: mse:{1:.6f}, mae:{2:.6f}, rmse:{3:.6f}'.format(
                    target_idx, mse, mae, rmse))
        
        # 计算平均指标
        avg_metrics = {}
        for metric_name in ['mae', 'mse', 'rmse']:
            avg_metrics[metric_name] = np.mean([m[metric_name] for m in all_metrics.values()])
        if self.args.pred_len == 1:
            avg_metrics['r2'] = np.mean([m['r2'] for m in all_metrics.values()])
            print('Average: r2:{0:.6f}, mse:{1:.6f}, mae:{2:.6f}, rmse:{3:.6f}'.format(
                avg_metrics['r2'], avg_metrics['mse'], avg_metrics['mae'], avg_metrics['rmse']))
        else:
            print('Average: mse:{0:.6f}, mae:{1:.6f}, rmse:{2:.6f}'.format(
                avg_metrics['mse'], avg_metrics['mae'], avg_metrics['rmse']))
        
        # 保存结果到文件
        result_file = "result_multi_objective_forecast.txt"
        f = open(result_file, 'a')
        f.write(setting + "  \n")
        for target_idx, metrics in all_metrics.items():
            if self.args.pred_len == 1:
                f.write('Target {0}: r2:{1:.6f}, mse:{2:.6f}, mae:{3:.6f}, rmse:{4:.6f}\n'.format(
                    target_idx, metrics['r2'], metrics['mse'], metrics['mae'], metrics['rmse']))
            else:
                f.write('Target {0}: mse:{1:.6f}, mae:{2:.6f}, rmse:{3:.6f}\n'.format(
                    target_idx, metrics['mse'], metrics['mae'], metrics['rmse']))
        f.write('\n')
        f.close()
        
        # 保存结果
        np.save(folder_path + 'metrics.npy', all_metrics)
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return