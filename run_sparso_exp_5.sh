#!/bin/bash

description="Experiment 5.0: combined phases"
echo $description

python train.py --run-description "${description}, combined phases" --batch-size 1024 --optimiser spaRSO --opt-iters 5 --swap-proportion 0.3 --initial-density 0.2 --maximum-density 0.3 --initial-prune-factor 0.1 --batch-mode every_weight --norm-type batch --const-norm-weights --phases prune regrow improve replace improve
