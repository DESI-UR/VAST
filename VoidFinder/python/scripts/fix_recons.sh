#!/bin/bash                                                                                                                                                                                                
#SBATCH --mem=350G                                                                                                                                                                                         
#SBATCH --job-name=fix_recons                                                                                                                                                                              
#SBATCH --time=02-00:00:00                                                                                                                                                                                 
#SBATCH --mail-type=ALL                                                                                                                                                                                    
#SBATCH -o fix_recons.log                                                                                                                                                                                  
#SBATCH -e fix_recons.err                 

echo 'Prepare reconstructed maps with removed 0s  without filter and before name changes.\
'
echo '350G 2d'
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
python /scratch/ierez/IGMCosmo/VoidFinder/python/scripts/prepare_recons_shiftback.py
echo 'Done :)'
now=$(date)
echo "Ending date: $now"
