#!/bin/bash

description="Final run replace only different config"
echo $description

python train.py --run-description "${description}" --batch-size 512 --optimiser spaRSO --opt-iters 100 --swap-proportion 0.3 --initial-density 0.25 --maximum-density 1.0 --batch-mode every_weight --norm-type batch --const-norm-weights --phases replace improve
