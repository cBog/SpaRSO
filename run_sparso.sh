#!/bin/bash

description="${@:1}"
echo $description

python train.py --run-description "${description}" --batch-size 1024 --optimiser spaRSO --opt-iters 50 --initial-density 0.05 --maximum-density 0.25 --swap-proportion 0.2 --consider-zero-improve --batch-mode every_iteration

