#!/bin/bash                                                                                                                                                                                                
#SBATCH --mem=200G                                                                                                                                                                                         
#SBATCH --job-name=deltafields_mergecheck_single                                                                                                                                                                 
#SBATCH --time=01:00:00                                                                                                                                                                                  
#SBATCH --mail-type=ALL                                                                                                                                                                                    
#SBATCH -o /scratch/ierez/IGMCosmo/VoidFinder/outputs/delta_runs/deltas_single/deltas_single.log                                                                                                                          
#SBATCH -e /scratch/ierez/IGMCosmo/VoidFinder/outputs/delta_runs/deltas_single/deltas_single.err                                
                                                                                                                              

echo 'Run VoidFinder on delta fields without filter for merge check.'
echo '200g 1h'
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
python /scratch/ierez/IGMCosmo/VoidFinder/scripts/VoidFinder_DR16S82_deltafields.py  
echo 'Done :)'
now=$(date)
echo "Ending date: $now"
