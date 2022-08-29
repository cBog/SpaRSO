#!/bin/bash

description="Final run all phases"
echo $description

python train.py --run-description "${description}" --batch-size 1024 --optimiser spaRSO --opt-iters 20 --swap-proportion 0.3 --initial-density 0.2 --maximum-density 0.3 --initial-prune-factor 0.3  --batch-mode every_phase --norm-type batch --const-norm-weights --consider-zero-improve --phases prune regrow improve replace improve
