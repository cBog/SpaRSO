#!/bin/bash

description="Experiment 3.0: prune, regrow, improve"
echo $description

python train.py --run-description "${description}, max density = 0.3, initial prune factor = 0.1" --batch-size 1024 --optimiser spaRSO --opt-iters 10 --initial-density 0.2 --maximum-density 0.3 --initial-prune-factor 0.1 --batch-mode every_weight --norm-type batch --const-norm-weights --phases prune regrow improve
python train.py --run-description "${description}, max density = 0.4, initial prune factor = 0.1" --batch-size 1024 --optimiser spaRSO --opt-iters 10 --initial-density 0.2 --maximum-density 0.4 --initial-prune-factor 0.1 --batch-mode every_weight --norm-type batch --const-norm-weights --phases prune regrow improve
python train.py --run-description "${description}, max density = 0.5, initial prune factor = 0.1" --batch-size 1024 --optimiser spaRSO --opt-iters 10 --initial-density 0.2 --maximum-density 0.5 --initial-prune-factor 0.1 --batch-mode every_weight --norm-type batch --const-norm-weights --phases prune regrow improve
python train.py --run-description "${description}, max density = 0.3, initial prune factor = 0.2" --batch-size 1024 --optimiser spaRSO --opt-iters 10 --initial-density 0.2 --maximum-density 0.3 --initial-prune-factor 0.2 --batch-mode every_weight --norm-type batch --const-norm-weights --phases prune regrow improve
python train.py --run-description "${description}, max density = 0.3, initial prune factor = 0.3" --batch-size 1024 --optimiser spaRSO --opt-iters 10 --initial-density 0.2 --maximum-density 0.3 --initial-prune-factor 0.3 --batch-mode every_weight --norm-type batch --const-norm-weights --phases prune regrow improve
