export CUDA_VIBLE_DEVICES=0

model_name=MGRU

task_name=short_term_forecast
seq_len=20
pred_len=2
hidden_dim=60


python -u ./run.py \
    --task_name $task_name \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path Debutanizer_Data.txt \
    --model_id Debutanizer_$seq_len'_'$pred_len \
    --model $model_name \
    --data Debutanizer \
    --features M \
    --input_dim 7\
    --target y \
    --target_columns -1 \
    --feature_columns 0 1 2 3 4 5 6 \
    --num_targets 1 \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --hidden_dim $hidden_dim \
    --learning_rate 0.0004 \
    --itr 1 \
    --train_epoch 100 \
    --batch_size 64 \
    --patience 7 \