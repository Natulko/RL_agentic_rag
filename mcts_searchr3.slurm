#!/bin/bash
#SBATCH --job-name=run_mcts_searchr2
#SBATCH --output=%x_%j.out
#SBATCH --mail-user="randomgrill73@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --partition="gpu-short"
#SBATCH --time=03:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=70G
#SBATCH --gres=gpu:2080_ti:4

module purge

# loading modules for GPUs
module load ALICE/default
module load slurm
module load CUDA/12.3.2
module load GCC/11.3.0

cd /home/s3648885/data0/Search-R1

# Load conda module
module load Miniconda3/23.10.0-1
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# Activate retriever environment
conda activate retriever

# Launch the retrieval server
echo "### Launching retrieval server..."
bash mcts_src/approach2/retrieval_launch.sh &
sleep 60
server_pid=$!
echo "### Retrieval server launched with PID $server_pid"

# Activate searchr1 environment
conda activate searchr1

echo $PATH

echo "### Starting Ollama server..."
ollama serve > ~/data0/Search-R1/ollama_server.log 2>&1 &
ollama_pid=$!
sleep 30

run_settings="single_query"
echo "### Running MCTS inference..."
python mcts_src/approach2/mcts.py --job_id $SLURM_JOB_ID \
                                  --run_settings $run_settings

# Kill the server using the correct port
echo "### Shutting down retrieval server (PID $((server_pid + 2)))..."
kill $((server_pid + 2))

# Kill the Ollama server
echo "### Stopping Ollama server (PID $ollama_pid)..."
kill $ollama_pid

echo "### Job finished"