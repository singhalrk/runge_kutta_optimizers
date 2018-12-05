#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mem=4000
#SBATCH -t24:00:00

#python main.py --optimizer RK2 --lr 0.05 --wd 5e-3
#python main.py --optimizer RK2_momentum --lr 0.05 --wd 5e-3
#python main.py --optimizer RK2_heun --lr 0.05 --wd 5e-3

#python main.py --optimizer SGD --lr 0.1 --wd 5e-4  --momentum 0.9
#python main.py --optimizer SGD_momentum --lr 0.1 --wd 5e-4  --momentum 0.9
#python main.py --optimizer SGD_nesterov --lr 0.1 --wd 5e-4  --nesterov True --momentum 0.9
#python main.py --optimizer Adagrad --lr 0.1 --wd 5e-4  --momentum 0.9

python main.py --optimizer RK2 --lr 0.1 --wd 5e-3
#python main.py --optimizer RK2_momentum --lr 0.1 --wd 5e-3
#python main.py --optimizer RK2_heun --lr 0.1 --wd 5e-3

python main.py --optimizer RK2 --lr 0.5 --wd 5e-3
#python main.py --optimizer RK2_momentum --lr 0.5 --wd 5e-3
#python main.py --optimizer RK2_heun --lr 0.5 --wd 5e-3
