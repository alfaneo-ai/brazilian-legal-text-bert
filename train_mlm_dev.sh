#!/bin/bash

python train_mlm.py \
    --config_name /home/cviegas/Workspace/mestrado/brazilian-legal-text-bert/tokenizer/config.json \
    --tokenizer_name /home/cviegas/Workspace/mestrado/brazilian-legal-text-bert/tokenizer \
    --train_file resources/corpus_train.txt \
    --validation_file resources/corpus_dev.txt \
    --output_dir output \
    --overwrite_output_dir false \
    --do_train \
    --do_eval \
    --do_predict \
    --line_by_line \
    --fp16 \
    --load_best_model_at_end \
    --save_steps 500 \
    --eval_steps 500 \
    --evaluation_strategy steps \
    --num_train_epochs 5 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 2 \
    --preprocessing_num_workers 4 \
    --max_seq_length 384 \
    --gradient_accumulation_steps 64 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --adam_epsilon 0.000001 \
    --learning_rate 0.0001 \
    --weight_decay 0.01 \
    --warmup_steps 10000 \
    --max_train_samples 1000 \
    --max_eval_samples 100

