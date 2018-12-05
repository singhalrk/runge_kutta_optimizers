#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH -t36:00:00


python main.py -a resnet18 --lr 0.1 -data
# python main.py -a resnet18 --lr 0.5

# python main.py -a resnet18 --optimizer RK2 --epoch_step 20
# python main.py -a resnet18 --optimizer RK2 --epoch_step 30

# python main.py -a resnet18 --optimizer RK2 --epoch_step 20 --lr 0.5
# python main.py -a resnet18 --optimizer RK2 --epoch_step 30 --lr 0.5

# python main.py -a resnet18 --optimizer RK2_heun --epoch_step 20
# python main.py -a resnet18 --optimizer RK2_heun --epoch_step 30

# python main.py -a resnet18 --optimizer RK2_heun --epoch_step 20 --lr 0.5
# python main.py -a resnet18 --optimizer RK2_heun --epoch_step 30 --lr 0.5

