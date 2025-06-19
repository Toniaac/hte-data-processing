#!/bin/bash
#SBATCH --account=def-j3goals
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=257000M
#SBATCH --output=llm-%j.out
#SBATCH --error=llm-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=hongchen.wang@mail.utoronto.ca

set -eu

# load the environment
env_name="llm-al-env"
module load python/3.11.5  scipy-stack
source $env_name/bin/activate 


python narval_job.py

