from data_loader import Dataset_Custom
from torch.utils.data import DataLoader

data_dict = {
    'Debutanizer': Dataset_Custom,
    'SRU': Dataset_Custom
}

def data_provider(args, flag):
    
    Data = data_dict[args.data]
    
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    batch_size = args.batch_size
    
    # use_graph = args.use_graph
    # if use_graph:
    # ...

    data_set = Data(
        args = args,
        root_path = args.root_path,
        data_path = args.data_path,
        flag = flag,
        size = [args.seq_len, args.label_len, args.pred_len],
        features = args.features,
        target = args.target,
        scale = getattr(args, 'scale', True)
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size = batch_size,
        shuffle = shuffle_flag,
        num_workers=args.num_workers
    )
    return data_set, data_loader


















    

if __name__ == '__main__':
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Test data_factory')
    parser.add_argument('--data', type=str, default='Debutanizer', help='Dataset name')
    parser.add_argument('--seq_len', type=int, default=4, help='Sequence length')
    parser.add_argument('--label_len', type=int, default=1, help='Label length')
    parser.add_argument('--pred_len', type=int, default=1, help='Prediction length')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers')
    parser.add_argument('--features', type=str, default='S', help='Features type')
    parser.add_argument('--target', type=str, default='y', help='Target column')
    parser.add_argument('--scale', type=bool, default=True, help='Whether to scale data')
    
    # 获取当前脚本所在目录，然后构建dataset路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(os.path.dirname(current_dir), 'dataset')
    
    parser.add_argument('--root_path', type=str, default=dataset_dir, help='Root path of data')
    parser.add_argument('--data_path', type=str, default='Debutanizer_Data.txt', help='Data file name')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Testing data_factory.py")
    print("=" * 60)
    print(f"Dataset: {args.data}")
    print(f"Root path: {args.root_path}")
    print(f"Data path: {args.data_path}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Label length: {args.label_len}")
    print(f"Prediction length: {args.pred_len}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)
    
    # 测试训练集
    print("\n[1] Testing training set...")
    train_set, train_loader = data_provider(args, 'train')
    print(f"Training set size: {len(train_set)}")
    if len(train_set) > 0:
        x, y = train_set[0]
        print(f"  First sample - X shape: {x.shape}, Y shape: {y.shape}")
        # 测试 DataLoader
        batch_x, batch_y = next(iter(train_loader))
        print(f"  First batch - X shape: {batch_x.shape}, Y shape: {batch_y.shape}")
    
    # 测试验证集
    print("\n[2] Testing validation set...")
    val_set, val_loader = data_provider(args, 'val')
    for i, (batch_x, batch_y) in enumerate(val_loader):
        print(i, batch_x.shape, batch_y.shape)
    print(f"Validation set size: {len(val_set)}")
    if len(val_set) > 0:
        x, y = val_set[0]
        print(f"  First sample - X shape: {x.shape}, Y shape: {y.shape}")
    
    # 测试测试集
    print("\n[3] Testing test set...")
    test_set, test_loader = data_provider(args, 'test')
    print(f"Test set size: {len(test_set)}")
    if len(test_set) > 0:
        x, y = test_set[0]
        print(f"  First sample - X shape: {x.shape}, Y shape: {y.shape}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60) 