#!/bin/bash
#SBATCH -o $MY_OUTPUTS/very_simple.log                                                                                                                                                             
#SBATCH -e $MY_OUTPUTS/very_simple.err  


echo 'It is very easy:)'
hostname
module load anaconda3
python $MY_SCRIPTS/very_simple.py

