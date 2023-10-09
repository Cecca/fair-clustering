remote := "ceccarello@login.dei.unipd.it:/nfsd/lovelace/ceccarello/fair-clustering"

syncdb:
  scp {{remote}}/results.db .
  touch experiments.qmd
  # scp {{remote}}/results.hdf5 .

preview:
  ./analysis.sif experiments.qmd

console:
  apptainer exec ./analysis.sif radian

build-analysis-container:
  apptainer build --force analysis.sif analysis.def

figures:
  R -e 'targets::tar_make()'

targets-clean:
  rm -rf _targets

watch-figures:
  ls **/*.R | entr just figures

update-images: targets-clean syncdb figures

