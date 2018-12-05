#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mem=8000
#SBATCH -t36:00:00

# python main.py --optimizer RK2 --lr 0.1
# python main.py --optimizer RK2 --lr 0.2
# python main.py --optimizer RK2 --lr 0.5
# python main.py --optimizer RK2 --lr 0.01
# python main.py --optimizer RK2 --lr 0.05
# python main.py --optimizer RK2 --lr 0.005

# python main.py --optimizer RK2 --lr 0.1 --wd 5e-6
# python main.py --optimizer RK2 --lr 0.2 --wd 5e-6
# python main.py --optimizer RK2 --lr 0.5 --wd 5e-6
# python main.py --optimizer RK2 --lr 0.01 --wd 5e-6
# python main.py --optimizer RK2 --lr 0.05 --wd 5e-6
# python main.py --optimizer RK2 --lr 0.001 --wd 5e-6


# python main.py --optimizer SGD --lr 0.1
# python main.py --optimizer SGD --lr 0.2
# python main.py --optimizer SGD --lr 0.5
# python main.py --optimizer SGD --lr 0.01
# python main.py --optimizer SGD --lr 0.05
# python main.py --optimizer SGD --lr 0.005

# python main.py --optimizer RK2_heun --lr 0.1
# python main.py --optimizer RK2_heun --lr 0.2
# python main.py --optimizer RK2_heun --lr 0.5
# python main.py --optimizer RK2_heun --lr 0.01
# python main.py --optimizer RK2_heun --lr 0.05
# python main.py --optimizer RK2_heun --lr 0.005

# python main.py --optimizer SGD_nesterov --lr 0.1
# python main.py --optimizer SGD_nesterov --lr 0.2
# python main.py --optimizer SGD_nesterov --lr 0.5
# python main.py --optimizer SGD_nesterov --lr 0.01
# python main.py --optimizer SGD_nesterov --lr 0.05
# python main.py --optimizer SGD_nesterov --lr 0.005

# python main.py --optimizer RK2 --lr 0.1 --wd 5e-5 --epoch_step '[30,60,90]' --epochs 100
# python main.py --optimizer RK2 --lr 0.2 --wd 5e-5 --epoch_step '[30,60,90]' --epochs 100
# python main.py --optimizer RK2 --lr 0.5 --wd 5e-5 --epoch_step '[30,60,90]' --epochs 100
# python main.py --optimizer RK2 --lr 0.01 --wd 5e-5 --epoch_step '[30,60,90]' --epochs 100
# python main.py --optimizer RK2 --lr 0.05 --wd 5e-5 --epoch_step '[30,60,90]' --epochs 100
# python main.py --optimizer RK2 --lr 0.001 --wd 5e-5 --epoch_step '[30,60,90]' --epochs 100


# python main.py --optimizer Adam --lr 0.1
# python main.py --optimizer Adam --lr 0.2
# python main.py --optimizer Adam --lr 0.5
# python main.py --optimizer Adam --lr 0.01
# python main.py --optimizer Adam --lr 0.05
# python main.py --optimizer Adam --lr 0.005


python main.py --optimizer RK2 --lr 1 --wd 5e-5 --epoch_step '[30,60,90]' --epochs 100 --lr_decay 0.5
python main.py --optimizer RK2 --lr 0.5 --wd 5e-5 --epoch_step '[30,60,90]' --epochs 100 --lr_decay 0.5
python main.py --optimizer RK2 --lr 1 --wd 5e-5 --epoch_step '[30,60,90]' --epochs 100 --lr_decay 0.5
python main.py --optimizer RK2 --lr 2 --wd 5e-7 --epoch_step '[30,60,90]' --epochs 100 --lr_decay 0.5
python main.py --optimizer RK2 --lr 0.6 --wd 5e-3 --epoch_step '[30,60,90]' --epochs 100 --lr_decay 0.5
python main.py --optimizer RK2 --lr 0.9 --wd 5e-5 --epoch_step '[30,60,90]' --epochs 100 --lr_decay 0.5

