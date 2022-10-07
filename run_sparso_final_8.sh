#!/bin/bash

description="Final run 8 replace only with norm weights enabled and bigger batch size"
echo $description

python train.py --run-description "${description}" --batch-size 2048 --optimiser spaRSO --opt-iters 100 --swap-proportion 0.3 --initial-density 0.2 --maximum-density 0.2 --batch-mode every_weight --norm-type batch --phases replace improve
