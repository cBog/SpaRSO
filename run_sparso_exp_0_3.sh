#!/bin/bash

description="Experiment 0.3: RSO baseline"
echo $description

python train.py --run-description "${description}" --batch-size 1024 --optimiser WPB_RSO --opt-iters 50

