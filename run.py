import argparse
import os
from sympy.polys.polytools import options
import torch
import torch.backends
from exp.exp_softsensor_predict import Exp_Softsensor_Predict
import random
import numpy as np

if __name__ == '__main__':
    fix_seed = 1024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='realtime_prediction',
                        help='task name, options:[realtime_prediction, short_term_forecast, imputation, generate_virtual_samples]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='VA-LSTM', 
                        help='model name, options: [VALSTM, LSTM, RNN, GRU]')
    
    # data loader
    parser.add_argument('--data', type=str, required=True, default='SRU', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='SRU.csv', help='data file')
    parser.add_argument('--features', type=str, default='M') # useless
    parser.add_argument('--target', type=str, default='y', help='prediction target')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # task of realtime_prediction 
    parser.add_argument('--seq_len', type=int, default=4, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=2, help='...') # useless
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')


    # model define

    ## VA-LSTM
    parser.add_argument('--hidden_dim', type=int, default=60, help='hidden_dim for VALSTM')

    # optimization
    parser.add_argument('--num_workers', type=int, default=1, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='exp times')
    parser.add_argument('--train_epochs', type=int, default=200, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_arugment('--gpu_type', type=str, default='cuda', help='gpu type')
    parser.add_argument('--use_multi_gpu', action='store_true', help='')

