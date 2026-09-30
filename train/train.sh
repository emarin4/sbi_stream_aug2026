#!/bin/bash
#SBATCH -A stf
#SBATCH -q normal
#SBATCH --gres=gpu:1  
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --job-name train_big_128HL
#SBATCH --time=20:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=emarin4@uw.edu
#SBATCH --mem=128GB
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

module load conda
conda activate sbi-stream

cd /gpfs/projects/dirac/emarin4/sbi_stream

config_path=/gpfs/projects/dirac/emarin4/sbi_stream/workspace/config_chebconv_tri.py
#config_run_id=desc_sbi_stream/aau_stream_jan2026/ctfaxck6

#config_checkpoint=/pscratch/sd/t/tvnguyen/stream_sbi_shared/graph_npe/aau_stream_jan2026/ctfaxck6/checkpoints/last.ckpt
export WANDB_CACHE_DIR=/gpfs/projects/dirac/emarin4/.cache/wandb
export WANDB_DIR=/gpfs/projects/dirac/emarin4/wandb
export WANDB_DATA_DIR=/gpfs/projects/dirac/emarin4/.cache/wandb_data

python scripts/train_npe.py --config $config_path #--config.run_id $config_run_id --config.checkpoint $config_checkpoint
