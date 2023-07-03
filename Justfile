syncdb:
  scp ceccarello@login.dei.unipd.it:/nfsd/lovelace/fair-clustering/results.db .
  scp ceccarello@login.dei.unipd.it:/nfsd/lovelace/fair-clustering/results.hdf5 .

update_images:
  python viz.py results.hdf5 imgs
