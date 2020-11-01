#!/bin/bash
#SBATCH -o prepare_deltas.log                                                                                         #SBATCH -e prepare_deltas.err  


echo 'Preparing deltas'
hostname
module load anaconda3
time
python $MY_SCRIPTS/prepare_deltas.py
time
