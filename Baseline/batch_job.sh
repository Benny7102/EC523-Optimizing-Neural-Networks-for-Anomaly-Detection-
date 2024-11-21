#!/bin/bash -l

# Specify project
#$ -P ec523kb

# Request appropriate time (default 12 hours; gpu jobs time limit - 2 days (48 hours), cpu jobs - 30 days (720 hours) )
#$ -l h_rt=12:00:00

# Send an email when the job is done or aborted (by default no email is sent)
#$ -m e

# Give job a name
#$ -N train_ur

# Join output and error streams into one file
#$ -j y

#$ -l gpus=1 
#$ -l gpu_c=3.5

# Load appropriate environment
module load miniconda

conda activate /projectnb/ec523kb/students/jli3469/.conda/envs/project

python /projectnb/ec523kb/projects/teams_Fall_2024/Team_9/UR-DMU/ucf_main.py
# If it's a zipped folder, you can optionally extract it
# Uncomment the line below if needed and replace 'file.zip' with the downloaded file name
# unzip file.zip
