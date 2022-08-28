#!/bin/bash

description="Experiment 4.0: others"
echo $description

python train.py --run-description "${description}, zero improve no prune" --batch-size 1024 --optimiser spaRSO --opt-iters 5 --initial-density 0.2 --maximum-density 0.3 --batch-mode every_weight --norm-type batch --const-norm-weights --consider-zero-improve --phases regrow improve
python train.py --run-description "${description}, zero improve is prune" --batch-size 1024 --optimiser spaRSO --opt-iters 5 --initial-density 0.2 --maximum-density 0.3 --initial-prune-factor 0.1 --batch-mode every_weight --norm-type batch --const-norm-weights --consider-zero-improve --phases prune regrow improve
python train.py --run-description "${description}, batch mode phase" --batch-size 1024 --optimiser spaRSO --opt-iters 5 --swap-proportion 0.3 --initial-density 0.2 --maximum-density 1.0 --batch-mode every_phase --norm-type batch --const-norm-weights --phases replace improve
python train.py --run-description "${description}, batch mode phase" --batch-size 1024 --optimiser spaRSO --opt-iters 5 --swap-proportion 0.3 --initial-density 0.2 --maximum-density 1.0 --batch-mode every_iteration --norm-type batch --const-norm-weights --phases replace improve
