#!/usr/bin/env bash
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
#
# Enter the dataset path ONCE here, then `source set_path.sh` before running any
# script. Every script in LIFE_code/ reads LIFE_BASE, so this is the only place a
# path needs to be set. LIFE_BASE is the folder that CONTAINS bids_dataset_LIFE/
# (so the BIDS dataset is at $LIFE_BASE/bids_dataset_LIFE).

export LIFE_BASE="/path/to/LIFE"                          # <-- edit this line
export FREESURFER_HOME="/Applications/freesurfer/7.4.1"   # structural stages only

echo "LIFE_BASE=$LIFE_BASE"
[ -d "$LIFE_BASE/bids_dataset_LIFE" ] \
  && echo "OK: found $LIFE_BASE/bids_dataset_LIFE" \
  || echo "WARN: $LIFE_BASE/bids_dataset_LIFE not found — edit set_path.sh"
