export CUDA_VIBLE_DEVICES=0

model_name=CNN1D

task_name=realtime_prediction
seq_len=20
pred_len=1
hidden_dim=30

python -u ./run.py \
    --task_name $task_name \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path Debutanizer_Data.txt \
    --model_id Debutanizer_$seq_len'_'$pred_len \
    --model $model_name \
    --data Debutanizer \
    --features M \
    --num_targets 1 \
    --input_dim 7 \
    --target y \
    --target_columns -1 \
    --feature_columns 0 1 2 3 4 5 6 \
    --num_targets 1 \
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