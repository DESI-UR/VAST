#!/bin/bash                                                                                                                   
#SBATCH --mem=350G                                                                                                            
#SBATCH --job-name=test                                                                                                       
#SBATCH --time=01-00:00:00                                                                                                    
#SBATCH --mail-type=ALL                                                                                                       
#SBATCH -o /scratch/ierez/IGMCosmo/VoidFinder/outputs/recons_runs/before_names.log                                
#SBATCH -e /scratch/ierez/IGMCosmo/VoidFinder/outputs/recons_runs/before_names.err                                
                                                                                                                              

echo 'Run VoidFinder on reconstructed maps with removed 0s  without filter and before name changes.'
echo '350G 5d'
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
python /scratch/ierez/IGMCosmo/VoidFinder/python/scripts/VoidFinder_DR16_reconstructed_fits_beforenames.py
echo 'Done :)'
now=$(date)
echo "Ending date: $now"
