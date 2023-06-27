#!/usr/bin/bash
#SBATCH --mem 0
#SBATCH --exclusive

apptainer exec --bind $(pwd):/work --pwd /work fairclustering.sif python experiments.py
