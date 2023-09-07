remote := "ceccarello@login.dei.unipd.it:/nfsd/lovelace/ceccarello/fair-clustering"

syncdb:
  scp {{remote}}/results.db .
  touch experiments.qmd
  # scp {{remote}}/results.hdf5 .

update_images:
  python viz.py results.hdf5 imgs

preview:
  ./analysis.sif experiments.qmd

console:
  apptainer exec ./analysis.sif radian

build-analysis-container:
  apptainer build --force analysis.sif analysis.def

