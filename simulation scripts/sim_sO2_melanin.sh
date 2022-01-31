#!/bin/bash
#SBATCH --job-name="sO2-mel"
#SBATCH --array=0-4999
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:gtx1080ti:1
#SBATCH --mem-per-cpu=6G
cd $HOME/projects
ID=${SLURM_ARRAY_TASK_ID}
rm -r SETS/MI-LSD/MI-LSD_sO2_melanin_skin_"$ID"_*
python sO2_melanin_sim_array.py ${SLURM_ARRAY_TASK_ID}

