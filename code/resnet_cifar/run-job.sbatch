#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH -t36:00:00

# python main.py --optimizer RK2 --lr 0.1 --wd 5e-4 --epoch_step '[80,120]' --epochs 350 --lr_decay 0.1

# python main.py --optimizer RK2 --lr 0.5 --wd 5e-4 --epoch_step '[40,80]' --lr_decay 0.5 --epochs 350

# python main.py --optimizer RK2 --lr 0.05 --wd 5e-4 --epochs 350

# python main.py --optimizer RK2 --lr 0.5 --wd 5e-4 --epoch_step '[40,80]' --lr_decay 0.1 --epochs 350

python main.py --optimizer RK2 --lr 0.1 --wd 5e-4 --epoch_step '[150, 250]' --lr_decay 0.1 --epochs 350

#python main.py --optimizer RK2_heun --lr 0.5 --wd 5e-4 --epoch_step '[100,200]' --lr_decay 0.1 --epochs 350

# python main.py --optimizer RK2_heun --lr 0.5 --wd 5e-4 --epoch_step '[-10,-50]' --lr_decay 0.5 --epochs 350

# python main.py --optimizer RK2_heun --lr 0.1 --wd 5e-4 --epoch_step '[150, 250]' --lr_decay 0.1 --epochs 350


# python main.py --optimizer SGD_momentum --lr 0.1 --wd 5e-4 --momentum 0.9 --epoch_step '[150,250]' --epochs 350

#python main.py --optimizer SGD --lr 0.1 --wd 5e-4  --momentum 0.9 --epoch_step '[150,250]' --epochs 350

#python main.py --optimizer SGD_momentum --lr 0.1 --wd 5e-4  --momentum 0.9 --epoch_step '[150,250]' --epochs 350

#python main.py --optimizer SGD_nesterov --lr 0.1 --wd 5e-4  --nesterov True --momentum 0.9 '[150,250]' --epochs 350


#python main.py --optimizer Adagrad --lr 0.1 --wd 5e-4  --momentum 0.9 --epoch_step '[50,100]' --epochs 350



