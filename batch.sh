#!/usr/bin/bash
#SBATCH --mem 0
#SBATCH --exclusive

apptainer exec \
	--bind $(pwd):/work \
	--pwd /work \
	fairclustering.sif \
	python experiments.py ~/opt/cplex/cplex/bin/x86-64_linux/cplex
