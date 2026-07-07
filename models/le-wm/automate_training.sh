#!/bin/bash
set -e

SIZES=(10 30 100 300 500 1000)

for SIZE in "${SIZES[@]}"
do
    echo
    echo "===== TRAIN SIZE: ${SIZE} ====="

    python train.py \
        data=ogb \
        trainer.max_epochs=100 \
        train_num_episodes=${SIZE}
done
