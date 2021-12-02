#!/bin/bash

python run_mlm.py \
    --model_name_or_path neuralmind/bert-base-portuguese-cased \
    --train_file corpus.txt \
    --output_dir output \
    --overwrite_output_dir true \
    --do_train \
    --line_by_line \
    --num_train_epochs 5.0 \
    --per_device_train_batch_size 2 \
    --max_seq_length 512 \
    --max_train_samples 200
