#!/bin/bash

description="Final run replace only with norm weights enabled"
echo $description

python train.py --run-description "${description}" --batch-size 1024 --optimiser spaRSO --opt-iters 60 --swap-proportion 0.6 --initial-density 0.2 --maximum-density 0.21 --batch-mode every_weight --norm-type batch --consider-zero-improve --phases replace improve regrow
