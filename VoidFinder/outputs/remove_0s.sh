#!/bin/bash
#SBATCH --mem=40G
#SBATCH --job-name=test
#SBATCH --time=02:00:00
#SBATCH --mail-user=email
#SBATCH -o /scratch/ierez/IGMCosmo/VoidFinder/outputs/remove_output.log                                                                                                                                  
#SBATCH -e /scratch/ierez/IGMCosmo/VoidFinder/outputs/remove_error.err  


echo 'Removing 0 entries.'
hostname
module load anaconda3
time
python /scratch/ierez/IGMCosmo/VoidFinder/data/DR16S82_H/reconstructed/ready_data_reconstructed.py
time
