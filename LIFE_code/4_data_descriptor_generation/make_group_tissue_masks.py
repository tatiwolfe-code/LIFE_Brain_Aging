#!/usr/bin/env python3
#For information, contact Tatiana Wolfe, PhD (tati.wolfe@gmail.com).
# ***** UAMS LIFE Study | Lifespan Imaging and Function in Elders *****
# ***** Tatiana Wolfe, Ph.D, v2026.11.06, tati.wolfe@gmail.com *****
"""
Group-level gray-matter and deep-cerebral-white-matter masks in MNI152 2 mm.

cohort_tsnr.py builds each participant's tissue masks in native EPI space and
uses them without writing them out. This script rebuilds them from the cached
inputs with exactly the same recipe, resamples each into MNI152 2 mm with the
participant's own EPI->MNI transform, and averages them into a probability map
across participants. The 50% consensus masks are what Figure 4C uses to show
the group-average tSNR map separately for each tissue.

Outputs -> derivatives/technical_validation/rsfmri/group_tissue_masks/
  {gm,wm}_prob_mni.nii.gz   fraction of participants whose mask covers a voxel
  {gm,wm}_mni.nii.gz        50% consensus binary masks
"""
import sys, os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import cohort_tsnr as T

OUT = C.DERIV_DIR / "technical_validation" / "rsfmri" / "group_tissue_masks"
OUT.mkdir(parents=True, exist_ok=True)
MNI = f"{T.FSLDIR}/data/standard/MNI152_T1_2mm_brain"


def one(sid):
    """Rebuild this participant's tissue masks and push them into MNI space."""
    w = T.WORK / sid
    e2m = w / f"{sid}_epi2mni.mat"
    if not e2m.exists():
        return None
    ld = lambda n: np.asarray(nib.load(str(w / f"{sid}_{n}.nii.gz")).dataobj)
    try:
        brain = ld("meanepi_brain_mask") > 0
        pve1, pve2 = ld("pve1_epi"), ld("pve2_epi")
        cg = ld("cerebrum_gm_epi") > 0.5
        cw = ld("cerebrum_wm_epi") > 0.5
        sub = ld("subcort_epi") > 0.3
    except Exception:                       # noqa: BLE001
        return None
    # identical recipe to cohort_tsnr.process()
    gm = (pve1 > T.GM_PVE_THR) & cg & brain
    wm = ndimage.binary_erosion((pve2 > T.WM_PVE_THR) & cw & (~sub) & brain,
                                iterations=1)
    aff = nib.load(str(w / f"{sid}_meanepi_brain_mask.nii.gz")).affine
    out = {}
    for name, m in (("gm", gm), ("wm", wm)):
        src = w / f"{sid}_{name}mask_epi.nii.gz"
        dst = w / f"{sid}_{name}mask_mni.nii.gz"
        nib.save(nib.Nifti1Image(m.astype(np.float32), aff), str(src))
        T.sh(["flirt", "-in", str(src), "-ref", MNI, "-applyxfm",
              "-init", str(e2m), "-interp", "trilinear", "-out", str(dst)])
        if not dst.exists():
            return None
        out[name] = np.asarray(nib.load(str(dst)).dataobj, dtype=np.float32)
    return out


def main():
    subs = sorted(p.name for p in T.WORK.glob("sub-*") if p.is_dir())
    acc, n = {"gm": None, "wm": None}, 0
    with ProcessPoolExecutor(max_workers=int(os.environ.get("LIFE_TSNR_WORKERS", "6"))) as ex:
        for sid, res in zip(subs, ex.map(one, subs)):
            if res is None:
                print(f"  {sid}: skipped (no cached transform/masks)", flush=True)
                continue
            for k in ("gm", "wm"):
                acc[k] = res[k] if acc[k] is None else acc[k] + res[k]
            n += 1
    if not n:
        print("no participants processed"); return
    ref = nib.load(f"{MNI}.nii.gz")
    for k in ("gm", "wm"):
        prob = acc[k] / n
        nib.save(nib.Nifti1Image(prob.astype(np.float32), ref.affine),
                 str(OUT / f"{k}_prob_mni.nii.gz"))
        nib.save(nib.Nifti1Image((prob >= 0.5).astype(np.uint8), ref.affine),
                 str(OUT / f"{k}_mni.nii.gz"))
        print(f"  {k}: {int((prob >= 0.5).sum())} voxels in the 50% consensus mask")
    print(f"GROUP_TISSUE_MASKS_DONE  n={n}  -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
