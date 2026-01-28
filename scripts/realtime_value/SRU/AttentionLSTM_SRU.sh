export CUDA_VIBLE_DEVICES=0

model_name=AttentionLSTM

seq_len=20
pred_len=1
hidden_dim=60

python -u ./softsensor-lib/run.py \
    --task_name realtime_prediction \
    --is_training 1 \
    --root_path ./softsensor-lib/dataset/ \
    --data_path SRU_data.txt \
    --model_id SRU_$seq_len'_'$pred_len \
    --model $model_name \
    --data SRU \
    --features M \
    --input_dim 6 \
    --target y \
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