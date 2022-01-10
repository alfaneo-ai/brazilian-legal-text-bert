#!/bin/bash

python mlm_trainning.py \
    --model_name_or_path neuralmind/bert-base-portuguese-cased \
    --train_file corpus.txt \
    --output_dir output \
    --overwrite_output_dir true \
    --do_train \
    --line_by_line \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --max_seq_length 384 \
    --max_train_samples 200
