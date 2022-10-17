#!/bin/bash

description="Final run replace only with 10 seeds"
echo $description

python train.py --run-description "${description}" --batch-size 1024 --optimiser spaRSO --opt-iters 50 --swap-proportion 0.3 --initial-density 0.2 --maximum-density 1.0 --batch-mode every_weight --norm-type batch --phases replace improve --seed 1
python train.py --run-description "${description}" --batch-size 1024 --optimiser spaRSO --opt-iters 50 --swap-proportion 0.3 --initial-density 0.2 --maximum-density 1.0 --batch-mode every_weight --norm-type batch --phases replace improve --seed 2
python train.py --run-description "${description}" --batch-size 1024 --optimiser spaRSO --opt-iters 50 --swap-proportion 0.3 --initial-density 0.2 --maximum-density 1.0 --batch-mode every_weight --norm-type batch --phases replace improve --seed 3
python train.py --run-description "${description}" --batch-size 1024 --optimiser spaRSO --opt-iters 50 --swap-proportion 0.3 --initial-density 0.2 --maximum-density 1.0 --batch-mode every_weight --norm-type batch --phases replace improve --seed 4
python train.py --run-description "${description}" --batch-size 1024 --optimiser spaRSO --opt-iters 50 --swap-proportion 0.3 --initial-density 0.2 --maximum-density 1.0 --batch-mode every_weight --norm-type batch --phases replace improve --seed 5
python train.py --run-description "${description}" --batch-size 1024 --optimiser spaRSO --opt-iters 50 --swap-proportion 0.3 --initial-density 0.2 --maximum-density 1.0 --batch-mode every_weight --norm-type batch --phases replace improve --seed 6
python train.py --run-description "${description}" --batch-size 1024 --optimiser spaRSO --opt-iters 50 --swap-proportion 0.3 --initial-density 0.2 --maximum-density 1.0 --batch-mode every_weight --norm-type batch --phases replace improve --seed 7
python train.py --run-description "${description}" --batch-size 1024 --optimiser spaRSO --opt-iters 50 --swap-proportion 0.3 --initial-density 0.2 --maximum-density 1.0 --batch-mode every_weight --norm-type batch --phases replace improve --seed 8
python train.py --run-description "${description}" --batch-size 1024 --optimiser spaRSO --opt-iters 50 --swap-proportion 0.3 --initial-density 0.2 --maximum-density 1.0 --batch-mode every_weight --norm-type batch --phases replace improve --seed 9
python train.py --run-description "${description}" --batch-size 1024 --optimiser spaRSO --opt-iters 50 --swap-proportion 0.3 --initial-density 0.2 --maximum-density 1.0 --batch-mode every_weight --norm-type batch --phases replace improve --seed 10
