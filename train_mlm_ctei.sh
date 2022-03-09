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
    --save_steps 5000 \
    --eval_steps 5000 \
    --evaluation_strategy steps \
    --num_train_epochs 5 \
    --per_device_train_batch_size 5 \
    --per_device_eval_batch_size 5 \
    --preprocessing_num_workers 8 \
    --max_seq_length 384

