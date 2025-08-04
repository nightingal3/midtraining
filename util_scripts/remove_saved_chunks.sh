#!/bin/bash

for i in {11..35}; do
  echo "👉 Deleting gs://cmu-gpucloud-mengyan3/c4_ctd_pythia/$i/saved_chunk_$i …"
  gcloud storage rm --recursive \
    "gs://cmu-gpucloud-mengyan3/c4_ctd_pythia/${i}/saved_chunk_${i}"
  gcloud storage rm --recursive \
    "gs://cmu-gpucloud-mengyan3/c4_ctd_pythia/${i}/${i}/saved_chunk_${i}"
done