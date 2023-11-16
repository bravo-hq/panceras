#!/bin/bash

cd ~/deeplearning/panceras

python train_test.py --fold 0 --model unetr3d --datatype Pancreas
python train_test.py --fold 1 --model unetr3d --datatype Pancreas
python train_test.py --fold 2 --model unetr3d --datatype Pancreas
python train_test.py --fold 3 --model unetr3d --datatype Pancreas

python train_test.py --fold 0 --model unetr3d --datatype LA
python train_test.py --fold 1 --model unetr3d --datatype LA
python train_test.py --fold 2 --model unetr3d --datatype LA
python train_test.py --fold 3 --model unetr3d --datatype LA
python train_test.py --fold 4 --model unetr3d --datatype LA