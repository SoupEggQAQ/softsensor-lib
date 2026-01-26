import argparse
import os
from sympy.polys.polytools import options
import torch
from torch._dynamo.mutation_guard import install_generation_tagging_init
import torch.backends
from torch.cpu import is_available
from exp.exp_softsensor_predict import Exp_Softsensor_Realtime_Value
from utils.print_args import print_args
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
    parser.add_argument('--input_dim', type=int, default=6, help='channel size')
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
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multi gpu', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multi gpus')

    # Argmentation
    parser.add_argument('--seed', type=int, default=1024, help='Randomization seed')

    args = parser.parse_args()
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')
    
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    print('Args in experiment:')
    print_args(args)

    if args.task_name in ['realtime_prediction', 'short_term_forecast']:
        Exp = Exp_Softsensor_Realtime_Value
    else:
        pass # waiting...

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            setting = '{}_{}_{}_{}_{}_{}_{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.target,
                args.seq_len,
                args.pred_len
            )
            print('>>>>>>>>>>>>>start training: {}>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    else:
        pass



