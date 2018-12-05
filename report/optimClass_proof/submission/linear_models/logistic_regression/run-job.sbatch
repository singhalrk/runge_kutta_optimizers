#!/bin/bash
#SBATCH --nodes=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000
#SBATCH -t24:00:00

python main.py --optimizer RK2 --lr 0.1 --model Logistic_Regression
python main.py --optimizer RK2 --lr 0.01  --model Logistic_Regression
python main.py --optimizer RK2 --lr 0.05  --model Logistic_Regression

python main.py --optimizer SGD --lr 0.1 --model Logistic_Regression 
python main.py --optimizer SGD --lr 0.01 --model Logistic_Regression
python main.py --optimizer SGD --lr 0.05 --model Logistic_Regression 

python main.py --optimizer RK4 --lr 0.1 --model Logistic_Regression
python main.py --optimizer RK4 --lr 0.01 --model Logistic_Regression
python main.py --optimizer RK4 --lr 0.05 --model Logistic_Regression
