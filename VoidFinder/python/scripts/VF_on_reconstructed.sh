#!/bin/bash
#SBATCH --mem=350G
#SBATCH --job-name=test
#SBATCH --time=5-00:00:00
#SBATCH --mail-type=ALL
#SBATCH -o /scratch/ierez/IGMCosmo/VoidFinder/outputs/VF_DR16r_without0s.log
#SBATCH -e /scratch/ierez/IGMCosmo/VoidFinder/outputs/VF_DR16r_without0s.err

echo 'Running VoidFinder on DR16 reconstructed maps'
hostname
module load anaconda3
python /scratch/ierez/IGMCosmo/VoidFinder/python/scripts/VoidFinder_DR16_reconstructed_fits.py
echo 'Done :)'
