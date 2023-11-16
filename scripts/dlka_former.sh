#!/bin/bash

cd ~/deeplearning/panceras

python train_test.py --fold 0 --model dlka-former --datatype Pancreas
python train_test.py --fold 1 --model dlka-former --datatype Pancreas
python train_test.py --fold 2 --model dlka-former --datatype Pancreas
python train_test.py --fold 3 --model dlka-former --datatype Pancreas

python train_test.py --fold 0 --model dlka-former --datatype LA
python train_test.py --fold 1 --model dlka-former --datatype LA
python train_test.py --fold 2 --model dlka-former --datatype LA
python train_test.py --fold 3 --model dlka-former --datatype LA
python train_test.py --fold 4 --model dlka-former --datatype LA