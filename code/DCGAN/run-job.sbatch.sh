#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH -t36:00:00

# python main.py --optimizer RK2_heun --lr 0.1 --niter 50 --cuda
# python main.py --optimizer RK2_heun --lr 0.5 --niter 50 --cuda
# python main.py --optimizer RK2_heun --lr 0.01 --niter 50 --cuda
# python main.py --optimizer RK2_heun --lr 0.05 --niter 50 --cuda

# python main.py --optimizer RK2 --lr 0.1 --niter 50 --cuda
# python main.py --optimizer RK2 --lr 0.5 --niter 50 --cuda
# python main.py --optimizer RK2 --lr 0.01 --niter 50 --cuda
# python main.py --optimizer RK2 --lr 0.05 --niter 50 --cuda


python main_adam.py --cuda --dataroot '.'
