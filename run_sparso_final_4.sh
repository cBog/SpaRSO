#!/bin/bash

description="Final run with bs64, replace and grow with zero to improve and 10 warm up phases and 100 iters"
echo $description

python train.py --run-description "${description}" --batch-size 64 --optimiser spaRSO --opt-iters 100 --swap-proportion 0.3 --initial-density 0.2 --maximum-density 0.4 --batch-mode every_phase --norm-type batch --const-norm-weights --consider-zero-improve --warm-up-replace-phases 10 --phases replace regrow improve
