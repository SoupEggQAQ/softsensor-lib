export CUDA_VIBLE_DEVICES=0

model_name=AttentionLSTM

task_name=multi_objective_prediction
seq_len=20
pred_len=1
hidden_dim=60

python -u ./run.py \
    --task_name $task_name \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path SRU_data.txt \
    --model_id SRU_$seq_len'_'$pred_len \
    --model $model_name \
    --data SRU \
    --features M \
    --input_dim 5 \
    --target y \
    --target_columns -1 -2 \
    --feature_columns 0 1 2 3 4  \
    --num_targets 2 \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --hidden_dim $hidden_dim \
    --bidirectional 0 \
    --dir_mult 1 \
    --attention_type scaled_dot \
    --learning_rate 0.0004 \
    --itr 1 \
    --train_epoch 100 \
    --batch_size 64 \
    --patience 7 \