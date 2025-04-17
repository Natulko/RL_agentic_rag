#!/bin/bash
#SBATCH --job-name=run_mcts_searchr1
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="bagannatasha@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --partition="gpu-short"
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --gres=gpu:2080_ti:4

module purge

# loading modules for GPUs
module load ALICE/default
module load slurm
module load CUDA/12.3.2
module load GCC/11.3.0

cd /home/s3648885/data0/Search-R1

# loading conda module
module load Miniconda3/23.10.0-1
# defining conda source
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

echo "### activating retriever env"
conda activate retriever
# making sure retriever executables are used
export PATH="/home/s3648885/.conda/envs/retriever/bin:$PATH"

bash mcts_src/approach2/retrieval_launch.sh &
server_pid=$!
sleep 30
echo "### retriever server running in background with PID $server_pid"

echo "### activating searchr1 env"
conda activate /home/s3648885/.conda/envs/searchr1
# making sure searchr1 executables are used
export PATH="/home/s3648885/.conda/envs/searchr1/bin:$PATH"

echo "### MCTS inference"
python mcts_src/approach2/mcts.py

server_pid=$(pgrep -f "uvicorn.*7348")
echo "### killing server with PID $server_pid"
pkill -f "uvicorn.*7348"

echo "### job finished"