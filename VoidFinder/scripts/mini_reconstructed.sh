#!/bin/bash                                                                                                                                                                                                                                  
#SBATCH --mem=200G                                                                                                                                                                                                                           
#SBATCH --job-name=mini_data                                                                                                                                                                                                                
#SBATCH --time=01:00:00                                                                                                                                                                                                                  
#SBATCH --mail-type=ALL                                                                                                                                                                                                                     
#SBATCH -o /scratch/ierez/IGMCosmo/VoidFinder/scripts/mini_reconstructed.log                                                                                                                                                  
#SBATCH -e /scratch/ierez/IGMCosmo/VoidFinder/scripts/mini_reconstructed.err                                                                                                                                                   

echo 'map_reconstructed.fitsn ---> map_reconstructed_mini.fits'
echo '200g 1d'
if [ X"$SLURM_STEP_ID" = "X" -a X"$SLURM_PROCID" = "X"0 ]
then
  echo "print =========================================="
  echo "print SLURM_JOB_ID = $SLURM_JOB_ID"
  echo "print SLURM_JOB_NODELIST = $SLURM_JOB_NODELIST"
  echo "print =========================================="
fi
hostname
now=$(date)
echo "Starting date: $now"
module load anaconda3
python /scratch/ierez/IGMCosmo/VoidFinder/scripts/mini_reconstructed.py
echo 'Done :)'
now=$(date)
echo "Ending date: $now"
