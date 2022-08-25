#!/bin/bash

description="Experiment 0.2: SGD baseline with pruning max 0.8"
echo $description

python train.py --run-description "${description}" --batch-size 1024 --optimiser SGD --epochs 10 --run-classic-pruning --pruning_min 0.5 --pruning_max 0.8

description="Experiment 0.2: SGD baseline with pruning max 0.9"
echo $description

python train.py --run-description "${description}" --batch-size 1024 --optimiser SGD --epochs 10 --run-classic-pruning --pruning_min 0.5 --pruning_max 0.9