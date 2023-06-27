#!/usr/bin/bash

apptainer exec --bind $(pwd):/work --pwd /work fairclustering.sif python experiments.py
