#!/bin/bash

description="${@:1}"
echo $description

python train.py --run-description "${description}" --batch-size 1024 --optimiser spaRSO --opt-iters 50 --initial-density 0.05 --maximum-density 0.25 --initial-prune-factor 0.2 --swap-proportion 0.2 --consider-zero-improve --batch-mode every_iteration --norm-type batch --const-norm-weights --phases improve prune regrow replace

