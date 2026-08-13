#!/bin/bash
#SBATCH -A upa160 
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=preprocess
#SBATCH --time=00:15:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user emarin4@uw.edu
#SBATCH --array=0-1
#SBATCH --output=logs/preprocess_newpreprocess_fulllog_%a.out


#cd /expanse/lustre/scratch/lmarin/temp_project/sbi_stream/
source ~/.bashrc
conda activate sbi-stream

export PYTHONPATH=/expanse/lustre/projects/upa160/lmarin/aau_sbi_project/temp_project:$PYTHONPATH

#cd /expanse/lustre/projects/upa160/lmarin/aau_sbi_project/temp_project
config_path=/expanse/lustre/projects/upa160/lmarin/aau_sbi_project/temp_project/sbi_stream/workspace/config_preprocess_newsims.py

python /expanse/lustre/projects/upa160/lmarin/aau_sbi_project/temp_project/sbi_stream/scripts/preprocess.py \
    --config $config_path
