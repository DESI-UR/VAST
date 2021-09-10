#!/bin/bash                                                                                                                   
#SBATCH --mem=400G                                                                                                            
#SBATCH --job-name=recons                                                                                                       
#SBATCH --time=05-00:00:00 

#SBATCH -c 17                                                                                                   
#SBATCH --mail-type=ALL                                                                                                       
#SBATCH -o /scratch/ierez/IGMCosmo/VoidFinder/outputs/recons_runs/before_names_continue.log                                
#SBATCH -e /scratch/ierez/IGMCosmo/VoidFinder/outputs/recons_runs/before_names_continue.err                                
                                                                                                                              

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
python /scratch/ierez/IGMCosmo/VoidFinder/python/scripts/VF_DR16_continue.py
echo 'Done :)'
now=$(date)
echo "Ending date: $now"
