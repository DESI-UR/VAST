#!/bin/bash
#SBATCH -o $MY_SCRIPTS/prepare_deltas.log                                                                                                                                                             
#SBATCH -e $MY_SCRIPTS/prepare_deltas.log  


echo 'Preparing deltas'
hostname
module load anaconda3
time
python $MY_SCRIPTS/prepare_deltas.py
time
