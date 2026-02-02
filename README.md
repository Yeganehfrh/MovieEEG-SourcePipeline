# MovieEEG – Source & ERP Pipeline
This repository contains the Movie-EEG preprocessing, source localisation, and ERP pipeline. The main executable logic lives in src/; notebooks are kept for inspection, debugging, and documentation only.

# Repository Structure
```bash
├── jobs/                              # SLURM job scripts
├── notebooks/                         # Development and inspection notebooks
├── src/                               # Core source code
│   └── MovieEEGSourcePipeline/
├── pixi.toml                          # Pixi environment specification
├── pixi.lock                          # and lockfile
├── .gitignore
└── .gitattributes
```

# Setup
From the repository root, create environment:
```
pixi install
```
