#!/bin/bash
#SBATCH -p gpu                # partition (queue)
#SBATCH --gres=gpu:a100:1     # request 1 A100 GPU
#SBATCH --time=04:00:00       # walltime
#SBATCH --mem=32G             # memory
#SBATCH -J distilbert_train   # job name
#SBATCH -o logs/train_%j.out  # save stdout/stderr

module load apptainer

apptainer exec --nv -B "$PWD":/work -W /work ~/qat-pipeline.sif \
    /bin/bash -lc 'source QAT_pipeline/scripts/bootstrap.sh; uv run python -u bert_train.py'
