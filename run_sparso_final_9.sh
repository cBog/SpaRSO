#!/bin/bash

description="Final run 9 replace only with norm weights enabled and same batch size"
echo $description

python train.py --run-description "${description}" --batch-size 1024 --optimiser spaRSO --opt-iters 100 --swap-proportion 0.3 --initial-density 0.2 --maximum-density 0.2 --batch-mode every_weight --norm-type batch --phases replace improve
