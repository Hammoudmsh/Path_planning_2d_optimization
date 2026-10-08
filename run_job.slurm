#!/bin/bash
#SBATCH --job-name=astar_job
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=6-00:00:00
#SBATCH --partition=mem
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G



# Create logs directory if it doesn't exist
mkdir -p logs


# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh  # or ~/anaconda3/etc/profile.d/conda.sh
conda activate astar_jps


# Optional third argument supplies an experiment JSON configuration.
extra_args=()
if [[ -n "${3:-}" ]]; then
    extra_args=(--config-json "$3")
fi

python -u run_benchmark.py "${1:?Pass a dataset name}" "${2:-full}" "${extra_args[@]}"