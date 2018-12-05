#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --mem=15000
#SBATCH -t48:00:00

# python main.py --cuda --lr 0.5  --save ./logs/resnet_$RANDOM$RANDOM  --optimizer RK2 --depth     16 --width 10 --ngpu 2

# python main.py --cuda --lr 0.5  --save ./logs/resnet_$RANDOM$RANDOM --optimizer SGD_momentum  --depth 16 --width 10 --ngpu 1

# python main.py --cuda  --epoch_step '[100,150]' --lr_decay_ratio 0.1 --lr 0.5 --epochs 250  --save ./logs/resnet_$RANDOM$RANDOM  --optimizer RK2 --depth     16 --width 10 --ngpu 1

# python main.py --cuda  --epoch_step '[50,100,150]' --lr_decay_ratio 0.1 --lr 0.5  --save ./logs/resnet_$RANDOM$RANDOM  --optimizer RK2_heun --depth     16 --width 10 --ngpu 1

# python main.py --cuda  --save ./logs/resnet_$RANDOM$RANDOM --optimizer SGD  --depth 16 --width 10 --ngpu 1 --lr 0.1

# python main.py --cuda  --epoch_step '[50,100,150]' --lr_decay_ratio 0.5 --lr 0.1  --save ./logs/resnet_$RANDOM$RANDOM  --optimizer RK2_heun --depth     16 --width 10 --ngpu 1

# python main.py --cuda  --save ./logs/resnet_$RANDOM$RANDOM --optimizer SGD_nesterov  --depth 16 --width 10 --ngpu 1 --lr 0.1 --epoch_step '[60,120,160]' --lr_decay_ratio 0.2 --weightDecay 0.0001

# python main.py --cuda  --save ./logs/resnet_$RANDOM$RANDOM --optimizer SGD_nesterov  --depth 16 --width 10 --ngpu 1 --lr 0.1 --epoch_step '[60,120,160]' --lr_decay_ratio 0.2 --weightDecay 0.0005


# python main.py --cuda  --save ./logs/resnet_$RANDOM$RANDOM --optimizer RK2  --depth 16 --width 10 --ngpu 1 --lr 0.1 --epoch_step '[30,60,90,120]' --lr_decay_ratio 0.5 --weightDecay 0.0005

# python main.py --cuda  --save ./logs/resnet_$RANDOM$RANDOM --optimizer RK2  --depth 16 --width 10 --ngpu 1 --lr 0.1 --epoch_step '[30,60,90,120]' --lr_decay_ratio 0.5 --weightDecay 0.0001

# python main.py --cuda  --save ./logs/resnet_$RANDOM$RANDOM --optimizer RK2  --depth 16 --width 10 --ngpu 1 --lr 0.1 --epoch_step '[30,60,90,120]' --lr_decay_ratio 0.1 --weightDecay 0.0001


# python main.py --cuda  --save ./logs/resnet_$RANDOM$RANDOM --optimizer RK2  --depth 16 --width 10 --ngpu 1 --lr 0.5 --epoch_step '[30,60,90,120]' --lr_decay_ratio 0.5 --weightDecay 0.0005

# python main.py --cuda  --save ./logs/resnet_$RANDOM$RANDOM --optimizer RK2  --depth 16 --width 10 --ngpu 1 --lr 0.5 --epoch_step '[30,60,90,120]' --lr_decay_ratio 0.1 --weightDecay 0.0001


# python main.py --cuda  --save ./logs/resnet_$RANDOM$RANDOM --optimizer RK2_heun  --depth 16 --width 10 --ngpu 1 --lr 0.1 --epoch_step '[30,60,90,120]' --lr_decay_ratio 0.5 --weightDecay 0.0005

# python main.py --cuda  --save ./logs/resnet_$RANDOM$RANDOM --optimizer RK2_heun  --depth 16 --width 10 --ngpu 1 --lr 0.1 --epoch_step '[30,60,90,120]' --lr_decay_ratio 0.5 --weightDecay 0.0001

# python main.py --cuda  --save ./logs/resnet_$RANDOM$RANDOM --optimizer RK2_heun  --depth 16 --width 10 --ngpu 1 --lr 0.1 --epoch_step '[30,60,90,120]' --lr_decay_ratio 0.1 --weightDecay 0.0001


# python main.py --cuda  --save ./logs/resnet_$RANDOM$RANDOM --optimizer RK2_heun  --depth 16 --width 10 --ngpu 1 --lr 0.1 --epoch_step '[30,60,90,120]' --lr_decay_ratio 0.5 --weightDecay 0.0005

# python main.py --cuda  --save ./logs/resnet_$RANDOM$RANDOM --optimizer RK2_heun --depth 16 --width 10 --ngpu 1 --lr 0.5 --epoch_step '[30,60,90,120]' --lr_decay_ratio 0.1 --weightDecay 0.0001

# python main.py --cuda  --save ./logs/resnet_$RANDOM$RANDOM --optimizer RK2_heun --depth 16 --width 10 --ngpu 1 --lr 0.5 --epoch_step '[100,150]' --lr_decay_ratio 0.1 --weightDecay 0.0001

# python main.py --cuda  --save ./logs/resnet_$RANDOM$RANDOM --optimizer RK2 --depth 16 --width 10 --ngpu 1 --lr 0.5 --epoch_step '[100,150]' --lr_decay_ratio 0.1 --weightDecay 0.0001


python main.py --cuda  --save ./logs/resnet_$RANDOM$RANDOM --optimizer RK2 --depth 16 --width 10 --ngpu 1 --lr 0.1 --epoch_step '[60,120,160]' --lr_decay_ratio 0.2 --weightDecay 0.0001

