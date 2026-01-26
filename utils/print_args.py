# print args
def print_args(args):
    
    print("\033[1m" + "Basic Config" + "\033[0m")
    print(f' {"Task Name:":<20}{args.task_name:<20}{"Is Training:":<20}{args.is_training:<20}')
    print(f' {"Model ID:":<20}{args.model_id:<20}{"Model:":<20}{args.model:<20}')
    print()

    print("\033[1m" + "Data Loader" + "\033[0m")
    print(f' {"Data:":<20}{args.data:<20}{"Root Path:":<20}{args.root_path:<20}')
    print(f' {"Data Path:":<20}{args.data_path:<20}{"Features:":<20}{args.features:<20}')
    print(f' {"Target:":<20}{args.target:<20}{"Checkpoints:":<20}{args.checkpoints:<20}')
    print()

    if args.task_name in ['realtime_prediction', 'short_term_forecast']:
        print(f' {"Seq Len:":<20}{args.seq_len:<20}')
        print(f' {"Pred Len:":<20}{args.pred_len:<20}')
        print()

    elif args.task_name in ['imputation']:
        pass

    elif args.task_name in ['generate_virtual_samples']:
        pass
    
    print("\033[1m" + "Model Parameters" + "\033[0m")
    # VALSTM
    if args.model in ['VALSTM']:
        print(f' {"Hidden Dim:":<20}{args.hidden_dim:<20}')

    print("\033[1m" + "Run Parameters" + "\033[0m")
    print(f'  {"Num Workers:":<20}{args.num_workers:<20}{"Itr:":<20}{args.itr:<20}')
    print(f'  {"Train Epochs:":<20}{args.train_epochs:<20}{"Batch Size:":<20}{args.batch_size:<20}')
    print(f'  {"Patience:":<20}{args.patience:<20}{"Learning Rate:":<20}{args.learning_rate:<20}')
    print(f'  {"Des:":<20}{args.des:<20}{"Loss:":<20}{args.loss:<20}')
    print(f'  {"Use Amp:":<20}{args.use_amp:<20}')
    print()

    print("\033[1m" + "GPU" + "\033[0m")
    print(f'  {"Use GPU:":<20}{args.use_gpu:<20}{"GPU:":<20}{args.gpu:<20}')
    print(f'  {"Use Multi GPU:":<20}{args.use_multi_gpu:<20}{"Devices:":<20}{args.devices:<20}')
    print()
