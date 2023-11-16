#!/bin/bash

cd ~/deeplearning/panceras

python train_test.py --fold 0 --model mainmodel --datatype Pancreas
python train_test.py --fold 1 --model mainmodel --datatype Pancreas
python train_test.py --fold 2 --model mainmodel --datatype Pancreas
python train_test.py --fold 3 --model mainmodel --datatype Pancreas

python train_test.py --fold 0 --model mainmodel --datatype LA
python train_test.py --fold 1 --model mainmodel --datatype LA
python train_test.py --fold 2 --model mainmodel --datatype LA
python train_test.py --fold 3 --model mainmodel --datatype LA
python train_test.py --fold 4 --model mainmodel --datatype LA