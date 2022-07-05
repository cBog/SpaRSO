#!/bin/bash

python train.py --run-description ${1} --batch-size 1024 --optimiser spaRSO --opt-iters 50 --initial-density 0.05 --maximum-density 0.25 --swap-proportion 0.2 --consider-zero-improve --batch-mode every_iteration

