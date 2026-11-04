#!/usr/bin/env python3
"""Preprocessing check (CPU): rebuild 44-slice stacks from raw train DICOMs and compare them with the
Raptor corpus, so the inference notebook reproduces the corpus exactly.

For N corpus studies, build the stack under a few candidate settings and report the mean Pearson r and
mean absolute difference per slot against the corpus volume. The best setting goes into inference.
Also reports which DICOM decoders are available (JPEG 2000 / JPEG Lossless need gdcm or pylibjpeg).
"""
import os, glob, time, json, itertools
import numpy as np, pandas as pd

N_STUDIES = 16
IMG = 336
SLOT_SIZES = [12, 10, 8, 6, 8]
SLOT_PLANES = [("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1), ("Coronal", 0), ("Axial", -1)]
CANDIDATES = {
    "span06_crop140": dict(lo=0.06, hi=0.94, crop=140.0),
    "span15_crop140": dict(lo=0.15, hi=0.85, crop=140.0),
    "span06_crop160": dict(lo=0.06, hi=0.94, crop=160.0),
    "span06_nocrop": dict(lo=0.06, hi=0.94, crop=None),
    "span10_crop140": dict(lo=0.10, hi=0.90, crop=140.0),
}


def find(name, root="/kaggle/input"):
    for d, sub, fs in os.walk(root):
        sub[:] = sorted(x for x in sub if x not in ("train_series", "test_series", "train_images", "test_images"))
        if name in fs:
            return os.path.join(d, name)
    return None


# ---- decoders
for mod in ["pydicom", "gdcm", "pylibjpeg", "openjpeg", "libjpeg", "cv2"]:
    try:
        m = __import__(mod); print(f"[decoder] {mod} OK {getattr(m, '__version__', '')}")
    except Exception as e:
        print(f"[decoder] {mod} missing ({type(e).__name__})")

import pydicom, cv2
from pydicom.pixel_data_handlers.util import apply_modality_lut

comp = os.path.dirname(find("train_series.csv"))
print("competition root:", comp, "|", sorted(os.listdir(comp)))
ser = pd.read_csv(f"{comp}/train_series.csv")
ser_by = {k: g.to_dict("records") for k, g in ser.groupby("StudyInstanceUID", sort=False)}
img_root = next(p for p in [f"{comp}/train_series", f"{comp}/train_images", f"{comp}/train"] if os.path.isdir(p))

ids = np.load(find("all_ids.npy"), allow_pickle=True).astype(str)
vols = np.load(find("all_vols.npy"), mmap_mode="r")
masks = np.load(find("all_masks.npy"))


def series_files(sdir):
    recs, pss = [], []
    for f in glob.glob(sdir + "/*.dcm"):
        h = pydicom.dcmread(f, stop_before_pixels=True)
        iop, ipp = getattr(h, "ImageOrientationPatient", None), getattr(h, "ImagePositionPatient", None)
        if iop is not None and ipp is not None and len(iop) == 6:
            n = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
            pos = float(np.dot(np.array(ipp, float), n))
        else:
            pos = float(getattr(h, "InstanceNumber", 0) or 0)
        ps = getattr(h, "PixelSpacing", None); ps = float(ps[0]) if ps is not None else 0.5
        recs.append((pos, f, ps)); pss.append(ps)
    recs.sort(key=lambda r: r[0])
    return [(f, ps) for _, f, ps in recs], (float(np.median(pss)) if pss else 0.5)


def read_px(f):
    d = pydicom.dcmread(f)
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def crop_resize(a, ps, crop):
    h, w = a.shape
    if crop:
        c = min(int(round(crop / max(ps, 1e-3))), min(h, w))
        y0, x0 = (h - c) // 2, (w - c) // 2
        a = a[y0:y0 + c, x0:x0 + c]
    return cv2.resize(a, (IMG, IMG), interpolation=cv2.INTER_AREA)


def pick(rows, plane, fluid, used):
    c = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        p = [r for r in c if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if p:
            return p[0]
    return c[0] if c else None


def build(sid, cfg, cache):
    D = sum(SLOT_SIZES); vol = np.zeros((D, IMG, IMG), np.uint8); idx = 0; used = set()
    for (plane, fluid), k in zip(SLOT_PLANES, SLOT_SIZES):
        r = pick(ser_by.get(sid, []), plane, fluid, used)
        if r is None:
            idx += k; continue
        used.add(r["SeriesInstanceUID"])
        sdir = f"{img_root}/{sid}/{r['SeriesInstanceUID']}"
        if sdir not in cache:
            cache[sdir] = series_files(sdir)
        files, med = cache[sdir]
        if not files:
            idx += k; continue
        n = len(files); lo, hi = int(n * cfg["lo"]), int(n * cfg["hi"]) - 1; hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        arrs = []
        for p in picks:
            f, ps = files[min(p, n - 1)]
            key = ("px", f)
            if key not in cache:
                cache[key] = read_px(f)
            arrs.append((cache[key], ps))
        allpx = np.concatenate([a.ravel() for a, _ in arrs])
        lq, hq = np.percentile(allpx, [2.0, 98.0])
        for a, ps in arrs:
            aw = np.clip((a - lq) / (hq - lq + 1e-6), 0, 1)
            vol[idx] = (crop_resize(aw, ps if ps > 0 else med, cfg["crop"]) * 255).astype(np.uint8); idx += 1
    return vol


rng = np.random.RandomState(0)
rows = rng.choice(len(ids), N_STUDIES, replace=False)
res = {k: {"r": [], "mad": [], "slot_r": [[] for _ in SLOT_SIZES]} for k in CANDIDATES}
bounds = np.cumsum([0] + SLOT_SIZES)
t0 = time.time()
for row in rows:
    sid = ids[row]; ref = np.asarray(vols[row]).astype(np.float32); m = masks[row]; cache = {}
    for name, cfg in CANDIDATES.items():
        try:
            v = build(sid, cfg, cache).astype(np.float32)
        except Exception as e:
            print(f"[{name}] {sid[-12:]} failed: {type(e).__name__}: {e}"); continue
        rs = []
        for s in range(len(SLOT_SIZES)):
            a, b = bounds[s], bounds[s + 1]
            if m[a:b].sum() == 0:
                continue
            x, y = v[a:b].ravel(), ref[a:b].ravel()
            r = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")
            res[name]["slot_r"][s].append(r); rs.append(r)
        res[name]["r"].append(np.nanmean(rs)); res[name]["mad"].append(float(np.abs(v - ref)[m > 0].mean()))
    print(f"study {sid[-12:]} done | {time.time() - t0:.0f}s", flush=True)

summary = {}
for name in CANDIDATES:
    d = res[name]
    summary[name] = dict(mean_r=float(np.nanmean(d["r"])), mean_abs_diff=float(np.nanmean(d["mad"])),
                         slot_r=[round(float(np.nanmean(x)), 4) if x else None for x in d["slot_r"]],
                         n=len(d["r"]))
    print(f"{name:16s} r {summary[name]['mean_r']:.4f} | MAD {summary[name]['mean_abs_diff']:.2f} | "
          f"per-slot r {summary[name]['slot_r']}")
json.dump(summary, open("/kaggle/working/prep_check.json", "w"), indent=1)
