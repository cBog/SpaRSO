#!/bin/bash

description="Experiment 0.1: SGD baseline"
echo $description

python train.py --run-description "${description}" --batch-size 128 --optimiser SGD --epochs 10

