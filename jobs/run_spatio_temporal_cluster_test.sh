#!/bin/sh

#SBATCH --job-name=SpatioTemporalCluster
#SBATCH --chdir=/mnt/aiongpfs/users/mansarinia/MovieEEG-SourcePipeline
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/aiongpfs/users/mansarinia/MovieEEG-SourcePipeline/logs/cluster_%j.log
#SBATCH --error=/mnt/aiongpfs/users/mansarinia/MovieEEG-SourcePipeline/logs/cluster_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=y.farahzadi@gmail.com

export MNE_NUM_JOBS=1
# export JOBLIB_BACKEND=loky
export CLUSTER_TIME_DECIM=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

LOG_DIR=/mnt/aiongpfs/users/mansarinia/MovieEEG-SourcePipeline/logs
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
fi

pixi run python -m src.MovieEEGSourcePipeline.pipeline.run_spatio_temporal_cluster_test
