syncdb:
  scp ceccarello@login.dei.unipd.it:/nfsd/lovelace/ceccarello/fair-clustering/results.db .
  scp ceccarello@login.dei.unipd.it:/nfsd/lovelace/ceccarello/fair-clustering/results.hdf5 .
  touch experiments.qmd

update_images:
  python viz.py results.hdf5 imgs

preview:
  quarto preview experiments.qmd
