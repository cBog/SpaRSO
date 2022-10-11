#!/bin/bash

description="Final run 10 replace only with norm weights enabled and giant batch size (fewer iters)"
echo $description

python train.py --run-description "${description}" --batch-size 4096 --optimiser spaRSO --opt-iters 50 --swap-proportion 0.4 --initial-density 0.2 --maximum-density 0.2 --batch-mode every_weight --norm-type batch --phases replace improve
