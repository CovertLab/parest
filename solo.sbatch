#!/bin/bash

#SBATCH --job-name="parest"

#SBATCH -p owners

#SBATCH --output="slurm/slurm-%A_%a.out"

#SBATCH --mail-user="jmason42@stanford.edu"
#SBATCH --mail-type="END,FAIL"

srun python main.py --seed=$SLURM_ARRAY_TASK_ID --problem=$1
