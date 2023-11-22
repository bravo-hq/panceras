#!/bin/bash

cd ~/deeplearning/panceras

python train_test.py --fold 0 --model vnet --datatype Pancreas
python train_test.py --fold 1 --model vnet --datatype Pancreas
python train_test.py --fold 2 --model vnet --datatype Pancreas
python train_test.py --fold 3 --model vnet --datatype Pancreas

python train_test.py --fold 0 --model vnet --datatype LA
python train_test.py --fold 1 --model vnet --datatype LA
python train_test.py --fold 2 --model vnet --datatype LA
python train_test.py --fold 3 --model vnet --datatype LA
python train_test.py --fold 4 --model vnet --datatype LA