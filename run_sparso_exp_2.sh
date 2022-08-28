#!/bin/bash

description="Experiment 2.0: replace improve varying swap ratio"
echo $description

python train.py --run-description "${description}, swap proportion = 0.2" --batch-size 1024 --optimiser spaRSO --opt-iters 5 --swap-proportion 0.2 --initial-density 0.2 --maximum-density 1.0 --batch-mode every_weight --norm-type batch --const-norm-weights --phases replace improve
python train.py --run-description "${description}, swap proportion = 0.3" --batch-size 1024 --optimiser spaRSO --opt-iters 5 --swap-proportion 0.3 --initial-density 0.2 --maximum-density 1.0 --batch-mode every_weight --norm-type batch --const-norm-weights --phases replace improve
python train.py --run-description "${description}, swap proportion = 0.4" --batch-size 1024 --optimiser spaRSO --opt-iters 5 --swap-proportion 0.4 --initial-density 0.2 --maximum-density 1.0 --batch-mode every_weight --norm-type batch --const-norm-weights --phases replace improve
python train.py --run-description "${description}, swap proportion = 0.5" --batch-size 1024 --optimiser spaRSO --opt-iters 5 --swap-proportion 0.5 --initial-density 0.2 --maximum-density 1.0 --batch-mode every_weight --norm-type batch --const-norm-weights --phases replace improve
python train.py --run-description "${description}, swap proportion = 0.6" --batch-size 1024 --optimiser spaRSO --opt-iters 5 --swap-proportion 0.6 --initial-density 0.2 --maximum-density 1.0 --batch-mode every_weight --norm-type batch --const-norm-weights --phases replace improve
