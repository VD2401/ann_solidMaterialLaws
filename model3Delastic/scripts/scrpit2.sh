#!/bin/bash

N=(8 16 32 64 128 256 384 512 640 768 896 1024)
for n in ${N[@]}
do 
echo "


————————————————————————————————
RUNNING SIMULATION FOR n_samples = $n
————————————————————————————————
"
python3 -u main.py --n_samples $n --n_epochs 10 --stress_number 0 --data_path data/data_files/ | tee -a output.txt
done