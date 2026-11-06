#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
export FSLDIR=/usr/local/fsl FSLOUTPUTTYPE=NIFTI_GZ; export PATH=$FSLDIR/bin:$PATH
# Working directory for the MNI masks. Defaults beside the derivatives so the
# script runs on any machine; override with LIFE_TSNR_SCRATCH (the same
# variable cohort_tsnr.py uses, so the two agree by default).
LIFE_BASE="${LIFE_BASE:-$(cd "$(dirname "$0")/../../.." && pwd)}"
W="${LIFE_TSNR_SCRATCH:-$LIFE_BASE/bids_dataset_LIFE/derivatives/technical_validation/rsfmri/work/tsnr}"
mkdir -p "$W"
A=$FSLDIR/data/atlases; cd $W; mkdir -p mni
# --- broad cerebrum (for GM): MNI brain minus cerebellum and brainstem ---
fslmaths $A/MNI/MNI-maxprob-thr0-2mm -thr 2 -uthr 2 -bin -dilM -dilM mni/cerebellum
fslmaths $A/HarvardOxford/HarvardOxford-sub-maxprob-thr25-2mm -thr 8 -uthr 8 -bin -dilM -dilM mni/brainstem
fslmaths $FSLDIR/data/standard/MNI152_T1_2mm_brain_mask -sub mni/cerebellum -sub mni/brainstem -thr 0.5 -bin mni/cerebrum_gm
# --- conservative cerebrum (for WM): union of MNI structural lobes/insula/BG ---
fslmaths $A/MNI/MNI-maxprob-thr25-2mm -thr 1 -uthr 1 -bin mni/acc
for L in 3 4 5 6 7 8 9; do
  fslmaths $A/MNI/MNI-maxprob-thr25-2mm -thr $L -uthr $L -bin mni/tmp
  fslmaths mni/acc -add mni/tmp mni/acc
done
fslmaths mni/acc -bin -dilM -dilM mni/cerebrum_wm
# --- subcortical gray nuclei, dilated (excluded from WM) ---
fslmaths mni/acc -mul 0 mni/subcort
for L in 4 5 6 7 9 10 11 15 16 17 18 19 20 21; do
  fslmaths $A/HarvardOxford/HarvardOxford-sub-maxprob-thr25-2mm -thr $L -uthr $L -bin mni/tmp
  fslmaths mni/subcort -add mni/tmp mni/subcort
done
fslmaths mni/subcort -bin -dilM mni/subcort
imrm mni/tmp mni/acc
for m in cerebrum_gm cerebrum_wm subcort; do printf "%-12s " $m; fslstats mni/$m -V; done
