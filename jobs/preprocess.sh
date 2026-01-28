#!/bin/sh

#SBATCH --job-name=Source_Localisation
#SBATCH --chdir=/mnt/aiongpfs/users/mansarinia/MovieEEG-SourcePipeline
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --output=/mnt/aiongpfs/users/mansarinia/MovieEEG-SourcePipeline/logs/SL_%j.log
#SBATCH --error=/mnt/aiongpfs/users/mansarinia/MovieEEG-SourcePipeline/logs/SL_%j.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=y.farahzadi@gmail.com


LOG_DIR=/mnt/aiongpfs/users/mansarinia/MovieEEG-SourcePipeline/logs
if [ ! -d "$LOG_DIR" ]; then
  mkdir -p "$LOG_DIR"
fi

pixi run python -m src.MovieEEGSourcePipeline.pipeline.run_preprocessing
