#!/bin/bash

#SBATCH --job-name="validate"

#SBATCH -p owners

#SBATCH --array=0-30

#SBATCH --output="slurm/slurm-%A_%a.out"

#SBATCH --time=24:00:00

srun python validate_model.py $SLURM_ARRAY_TASK_ID
