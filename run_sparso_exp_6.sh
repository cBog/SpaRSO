#!/bin/bash

description="Experiment 6.0: warm up replace"
echo $description

python train.py --run-description "${description}, 1 warm up phase" --batch-size 1024 --optimiser spaRSO --opt-iters 5 --swap-proportion 0.3 --initial-density 0.2 --maximum-density 1.0 --batch-mode every_weight --norm-type batch --const-norm-weights --warm-up-replace-phases 1 --phases replace improve
python train.py --run-description "${description}, 2 warm up phase" --batch-size 1024 --optimiser spaRSO --opt-iters 5 --swap-proportion 0.3 --initial-density 0.2 --maximum-density 1.0 --batch-mode every_weight --norm-type batch --const-norm-weights --warm-up-replace-phases 2 --phases replace improve
python train.py --run-description "${description}, 3 warm up phase" --batch-size 1024 --optimiser spaRSO --opt-iters 5 --swap-proportion 0.3 --initial-density 0.2 --maximum-density 1.0 --batch-mode every_weight --norm-type batch --const-norm-weights --warm-up-replace-phases 3 --phases replace improve

