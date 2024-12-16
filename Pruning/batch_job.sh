#!/bin/bash -l

# Specify project
#$ -P ec523kb

# Request appropriate time (default 12 hours; gpu jobs time limit - 2 days (48 hours), cpu jobs - 30 days (720 hours) )
#$ -l h_rt=12:00:00

# Send an email when the job is done or aborted (by default no email is sent)
#$ -m e

# Give job a name
#$ -N fast-ur-dmu-target-99

# Join output and error streams into one file
#$ -j y

#$ -l gpus=1
#$ -l gpu_c=5
#$ -l gpu_type=A100

# Load appropriate environment
module load python3/3.10.12
module load cuda/11.8

source /projectnb/ec523kb/projects/teams_Fall_2024/Team_9/Pruning_UR_DMU/Fast-UR-DMU/venv/bin/activate

python /projectnb/ec523kb/projects/teams_Fall_2024/Team_9/Pruning_UR_DMU/Fast-UR-DMU/ucf_main_target_99.py --root_dir /projectnb/ec523kb/projects/teams_Fall_2024/Team_9/BN-WVAD/root-data/RGB_train/ --lr '[0.0001]*1000'
# If it's a zipped folder, you can optionally extract it
# Uncomment the line below if needed and replace 'file.zip' with the downloaded file name
# unzip file.zip
