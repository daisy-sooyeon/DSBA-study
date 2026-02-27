#!/bin/bash

seq_len=96
label_len=48
data_path=./dataset/ETTm2.csv
data_name=ETTm2
model_name=InformerStack
batch_size=32

# 1. pred_len = 24
accelerate launch --num_processes 1 main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.seq_len $seq_len \
    DATASET.label_len $label_len \
    DATASET.pred_len 24 \
    DATASET.time_embedding "[1, 'min']" \
    DATAINFO.datadir $data_path \
    DEFAULT.exp_name forecasting_${data_name}_${model_name}_${seq_len}_24 \
    TRAIN.batch_size $batch_size \
    MODELSETTING.d_model 512 \
    MODELSETTING.n_heads 8  \
    MODELSETTING.freq 'min'

# 2. pred_len = 48
accelerate launch --num_processes 1 main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.seq_len $seq_len \
    DATASET.label_len $label_len \
    DATASET.pred_len 48 \
    DATASET.time_embedding "[1, 'min']" \
    DATAINFO.datadir $data_path \
    DEFAULT.exp_name forecasting_${data_name}_${model_name}_${seq_len}_48 \
    TRAIN.batch_size $batch_size \
    MODELSETTING.d_model 512 \
    MODELSETTING.n_heads 8  \
    MODELSETTING.freq 'min'

# 3. pred_len = 168
accelerate launch --num_processes 1 main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.seq_len $seq_len \
    DATASET.label_len $label_len \
    DATASET.pred_len 168 \
    DATASET.time_embedding "[1, 'min']" \
    DATAINFO.datadir $data_path \
    DEFAULT.exp_name forecasting_${data_name}_${model_name}_${seq_len}_168 \
    TRAIN.batch_size $batch_size \
    MODELSETTING.d_model 512 \
    MODELSETTING.n_heads 8  \
    MODELSETTING.freq 'min'

# 4. pred_len = 336
accelerate launch --num_processes 1 main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.seq_len $seq_len \
    DATASET.label_len $label_len \
    DATASET.pred_len 336 \
    DATASET.time_embedding "[1, 'min']" \
    DATAINFO.datadir $data_path \
    DEFAULT.exp_name forecasting_${data_name}_${model_name}_${seq_len}_336 \
    TRAIN.batch_size $batch_size \
    MODELSETTING.d_model 512 \
    MODELSETTING.n_heads 8  \
    MODELSETTING.freq 'min'

# 5. pred_len = 720
accelerate launch --num_processes 1 main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.seq_len $seq_len \
    DATASET.label_len $label_len \
    DATASET.pred_len 720 \
    DATASET.time_embedding "[1, 'min']" \
    DATAINFO.datadir $data_path \
    DEFAULT.exp_name forecasting_${data_name}_${model_name}_${seq_len}_720 \
    TRAIN.batch_size $batch_size \
    MODELSETTING.d_model 512 \
    MODELSETTING.n_heads 8  \
    MODELSETTING.freq 'min'

# 6. pred_len = 960
accelerate launch --num_processes 1 main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.seq_len $seq_len \
    DATASET.label_len $label_len \
    DATASET.pred_len 960 \
    DATASET.time_embedding "[1, 'min']" \
    DATAINFO.datadir $data_path \
    DEFAULT.exp_name forecasting_${data_name}_${model_name}_${seq_len}_960 \
    TRAIN.batch_size $batch_size \
    MODELSETTING.d_model 512 \
    MODELSETTING.n_heads 8  \
    MODELSETTING.freq 'min'



# 다변량
# 1. pred_len = 24
accelerate launch --num_processes 1 main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.seq_len $seq_len \
    DATASET.label_len $label_len \
    DATASET.pred_len 24 \
    DATASET.features "M" \
    DATASET.time_embedding "[1, 'min']" \
    DATAINFO.datadir $data_path \
    DEFAULT.exp_name forecasting_M_${data_name}_${model_name}_${seq_len}_24 \
    TRAIN.batch_size $batch_size \
    MODELSETTING.enc_in 7 \
    MODELSETTING.dec_in 7 \
    MODELSETTING.c_out 7 \
    MODELSETTING.d_model 512 \
    MODELSETTING.n_heads 8  \
    MODELSETTING.freq 'min'

# 2. pred_len = 48
accelerate launch --num_processes 1 main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.seq_len $seq_len \
    DATASET.label_len $label_len \
    DATASET.pred_len 48 \
    DATASET.features "M" \
    DATASET.time_embedding "[1, 'min']" \
    DATAINFO.datadir $data_path \
    DEFAULT.exp_name forecasting_M_${data_name}_${model_name}_${seq_len}_48 \
    TRAIN.batch_size $batch_size \
    MODELSETTING.enc_in 7 \
    MODELSETTING.dec_in 7 \
    MODELSETTING.c_out 7 \
    MODELSETTING.d_model 512 \
    MODELSETTING.n_heads 8  \
    MODELSETTING.freq 'min'

# 3. pred_len = 168
accelerate launch --num_processes 1 main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.seq_len $seq_len \
    DATASET.label_len $label_len \
    DATASET.pred_len 168 \
    DATASET.features "M" \
    DATASET.time_embedding "[1, 'min']" \
    DATAINFO.datadir $data_path \
    DEFAULT.exp_name forecasting_M_${data_name}_${model_name}_${seq_len}_168 \
    TRAIN.batch_size $batch_size \
    MODELSETTING.enc_in 7 \
    MODELSETTING.dec_in 7 \
    MODELSETTING.c_out 7 \
    MODELSETTING.d_model 512 \
    MODELSETTING.n_heads 8  \
    MODELSETTING.freq 'min'

# 4. pred_len = 336
accelerate launch --num_processes 1 main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.seq_len $seq_len \
    DATASET.label_len $label_len \
    DATASET.pred_len 336 \
    DATASET.features "M" \
    DATASET.time_embedding "[1, 'min']" \
    DATAINFO.datadir $data_path \
    DEFAULT.exp_name forecasting_M_${data_name}_${model_name}_${seq_len}_336 \
    TRAIN.batch_size $batch_size \
    MODELSETTING.enc_in 7 \
    MODELSETTING.dec_in 7 \
    MODELSETTING.c_out 7 \
    MODELSETTING.d_model 512 \
    MODELSETTING.n_heads 8  \
    MODELSETTING.freq 'min'

# 5. pred_len = 720
accelerate launch --num_processes 1 main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.seq_len $seq_len \
    DATASET.label_len $label_len \
    DATASET.pred_len 720 \
    DATASET.features "M" \
    DATASET.time_embedding "[1, 'min']" \
    DATAINFO.datadir $data_path \
    DEFAULT.exp_name forecasting_M_${data_name}_${model_name}_${seq_len}_720 \
    TRAIN.batch_size $batch_size \
    MODELSETTING.enc_in 7 \
    MODELSETTING.dec_in 7 \
    MODELSETTING.c_out 7 \
    MODELSETTING.d_model 512 \
    MODELSETTING.n_heads 8  \
    MODELSETTING.freq 'min'

# 6. pred_len = 960
accelerate launch --num_processes 1 main.py \
    --model_name $model_name \
    --default_cfg ./configs/default_setting.yaml \
    --model_cfg ./configs/model_setting.yaml \
    DATASET.seq_len $seq_len \
    DATASET.label_len $label_len \
    DATASET.pred_len 960 \
    DATASET.features "M" \
    DATASET.time_embedding "[1, 'min']" \
    DATAINFO.datadir $data_path \
    DEFAULT.exp_name forecasting_M_${data_name}_${model_name}_${seq_len}_960 \
    TRAIN.batch_size $batch_size \
    MODELSETTING.enc_in 7 \
    MODELSETTING.dec_in 7 \
    MODELSETTING.c_out 7 \
    MODELSETTING.d_model 512 \
    MODELSETTING.n_heads 8  \
    MODELSETTING.freq 'min'