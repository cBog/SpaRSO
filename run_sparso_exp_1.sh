#!/bin/bash

description="Experiment 1.0: improve only"
echo $description

python train.py --run-description "${description} initial density = 0.7" --batch-size 1024 --optimiser spaRSO --opt-iters 10 --initial-density 0.7 --maximum-density 1.0 --batch-mode every_weight --norm-type batch --const-norm-weights --phases improve
# python train.py --run-description "${description} initial density = 0.6" --batch-size 1024 --optimiser spaRSO --opt-iters 10 --initial-density 0.6 --maximum-density 1.0 --batch-mode every_weight --norm-type batch --const-norm-weights --phases improve
# python train.py --run-description "${description} initial density = 0.5" --batch-size 1024 --optimiser spaRSO --opt-iters 10 --initial-density 0.5 --maximum-density 1.0 --batch-mode every_weight --norm-type batch --const-norm-weights --phases improve
# python train.py --run-description "${description} initial density = 0.4" --batch-size 1024 --optimiser spaRSO --opt-iters 10 --initial-density 0.4 --maximum-density 1.0 --batch-mode every_weight --norm-type batch --const-norm-weights --phases improve
# python train.py --run-description "${description} initial density = 0.3" --batch-size 1024 --optimiser spaRSO --opt-iters 10 --initial-density 0.3 --maximum-density 1.0 --batch-mode every_weight --norm-type batch --const-norm-weights --phases improve
# python train.py --run-description "${description} initial density = 0.2" --batch-size 1024 --optimiser spaRSO --opt-iters 10 --initial-density 0.2 --maximum-density 1.0 --batch-mode every_weight --norm-type batch --const-norm-weights --phases improve
# python train.py --run-description "${description} initial density = 0.1" --batch-size 1024 --optimiser spaRSO --opt-iters 10 --initial-density 0.1 --maximum-density 1.0 --batch-mode every_weight --norm-type batch --const-norm-weights --phases improve