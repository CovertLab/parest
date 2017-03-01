#!/bin/bash

#SBATCH --job-name="parest"

#SBATCH -p owners

#SBATCH --output="slurm/slurm-%A_%a.out"

#SBATCH --time=24:00:00

let seed=SLURM_ARRAY_TASK_ID*$2

srun python main.py --problem=$1 --seed=$seed --n=$2
