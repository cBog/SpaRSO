#!/bin/bash

description="Final run replace only"
echo $description

python train.py --run-description "${description}" --batch-size 1024 --optimiser spaRSO --opt-iters 50 --swap-proportion 0.3 --initial-density 0.2 --maximum-density 1.0 --batch-mode every_phase --norm-type batch --const-norm-weights --phases replace improve
