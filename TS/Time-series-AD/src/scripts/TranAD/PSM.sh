#!/bin/bash

model_name=TranAD
data_name=PSM

accelerate launch --num_processes 1 src/main.py \
    --model_name $model_name \
    --default_cfg src/configs/default_setting.yaml \
    --model_cfg src/configs/model_setting.yaml \
    --opts DEFAULT.exp_name anomaly_detection_${data_name}_${model_name}