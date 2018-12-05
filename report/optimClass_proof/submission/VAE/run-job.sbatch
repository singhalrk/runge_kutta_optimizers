#!/bin/bash


#SBATCH --gres=gpu:1   
#SBATCH --mem=4000
#SBATCH -t24:00:00
 
#python main.py --optimizer RK2 --lr 0.05 --epoch 25
#python main.py --optimizer RK2 --lr 0.5 --epoch 25
#python main.py --optimizer RK2 --lr 0.005 --epoch 25
#python main.py --optimizer RK2 --lr 1e-3 --epoch 25
 
python main.py --optimizer SGD --lr 0.05 --epoch 25
python main.py --optimizer SGD --lr 0.5 --epoch 25
python main.py --optimizer SGD --lr 0.005 --epoch 25
python main.py --optimizer SGD --lr 1e-3 --epoch 25

#python main.py --optimizer Adam --lr 0.05 --epoch 25 
#python main.py --optimizer Adam --lr 1e-3 --epoch 25
#python main.py --optimizer Adam --lr 1e-2 --epoch 25
