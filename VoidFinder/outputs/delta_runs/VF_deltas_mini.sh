#!/bin/bash                                                                                                                                                                                         
#SBATCH --job-name=deltas_mini                                                                                                                                                 
#SBATCH --mail-type=ALL                                                                                                       
#SBATCH -o /scratch/ierez/IGMCosmo/VoidFinder/outputs/delta_runs/before_names_mini.log                                
#SBATCH -e /scratch/ierez/IGMCosmo/VoidFinder/outputs/delta_runs/before_names_mini.err                                
                                                                                                                              

echo 'Run VoidFinder on shifted by 90 delta fields without filter and before name changes.'
echo '200G 1d'
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
python /scratch/ierez/IGMCosmo/VoidFinder/python/scripts/VoidFinder_DR16_deltafields_fits.py
echo 'Done :)'
now=$(date)
echo "Ending date: $now"
