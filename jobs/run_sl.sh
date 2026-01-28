#!/bin/sh

#SBATCH --job-name=Source_Localisation
#SBATCH --chdir=/mnt/aiongpfs/users/mansarinia/MovieEEG-SourcePipeline
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=15:00:00
#SBATCH --mem=16G
#SBATCH --output=/mnt/aiongpfs/users/mansarinia/MovieEEG-SourcePipeline/logs/SL_%j.log
#SBATCH --error=/mnt/aiongpfs/users/mansarinia/MovieEEG-SourcePipeline/logs/SL_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=y.farahzadi@gmail.com

export MNE_NUM_JOBS=${SLURM_CPUS_PER_TASK:-16}
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

LOG_DIR=/mnt/aiongpfs/users/mansarinia/MovieEEG-SourcePipeline/logs
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
fi

pixi run python -m src.MovieEEGSourcePipeline.pipeline.run_source_estimation
