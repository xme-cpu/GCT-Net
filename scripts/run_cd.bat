@echo off
REM 训练脚本：适用于Windows系统的批处理文件
cd ..

set gpus=0
set checkpoint_root=checkpoints
set data_name=LEVIR

set img_size=256
set batch_size=8
set lr=0.01
set max_epochs=200
set net_G=base_transformer_pos_s4_dd8
REM base_resnet18
REM base_transformer_pos_s4_dd8
REM base_transformer_pos_s4_dd8_dedim8
set lr_policy=linear

set split=trainval
set split_val=test
set project_name=CD_%net_G%_%data_name%_b%batch_size%_lr%lr%_%split%_%split_val%_%max_epochs%_%lr_policy%

python main_cd.py ^
    --img_size %img_size% ^
    --checkpoint_root %checkpoint_root% ^
    --lr_policy %lr_policy% ^
    --split %split% ^
    --split_val %split_val% ^
    --net_G %net_G% ^
    --gpu_ids %gpus% ^
    --max_epochs %max_epochs% ^
    --project_name %project_name% ^
    --batch_size %batch_size% ^
    --data_name %data_name% ^
    --lr %lr%
