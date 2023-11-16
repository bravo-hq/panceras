#!/bin/bash

cd ~/deeplearning/panceras

python train_test.py --fold 0 --model swinunetr3d --datatype Pancreas
python train_test.py --fold 1 --model swinunetr3d --datatype Pancreas
python train_test.py --fold 2 --model swinunetr3d --datatype Pancreas
python train_test.py --fold 3 --model swinunetr3d --datatype Pancreas

python train_test.py --fold 0 --model swinunetr3d --datatype LA
python train_test.py --fold 1 --model swinunetr3d --datatype LA
python train_test.py --fold 2 --model swinunetr3d --datatype LA
python train_test.py --fold 3 --model swinunetr3d --datatype LA
python train_test.py --fold 4 --model swinunetr3d --datatype LA