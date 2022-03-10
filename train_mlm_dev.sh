#!/bin/bash

python train_mlm.py \
    --model_name_or_path neuralmind/bert-base-portuguese-cased \
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
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --preprocessing_num_workers 4 \
    --max_seq_length 384 \
    --gradient_checkpointing \
    --gradient_accumulation_steps 64 \
    --max_train_samples 1000 \
    --max_eval_samples 100
