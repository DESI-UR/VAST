#!/bin/bash
#SBATCH --mem=300G
#SBATCH --job-name=test
#SBATCH --time=06:00:00
#SBATCH --mail-user=email
#SBATCH -o /scratch/ierez/IGMCosmo/VoidFinder/outputs/VF_DR16r_output.log                                                                                                                                  
#SBATCH -e /scratch/ierez/IGMCosmo/VoidFinder/outputs/VF_DR16r_error.err  


echo 'Running the reconstructed maps on VoidFinder with 300G and 2h.'
hostname
module load anaconda3
time
python /scratch/ierez/IGMCosmo/VoidFinder/python/scripts/VoidFinder_DR16_reconstructed_changed.py
time
