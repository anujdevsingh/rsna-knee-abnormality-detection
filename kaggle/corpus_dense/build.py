#!/usr/bin/env python3
"""Dense corpus (CPU): build 96-slice, 288 px stacks from the train DICOMs, one shard of the studies.

Same reader as the matched 44-slice corpus (kaggle/prep_check: r 1.0000 vs Raptor): series picked per
slot in *_series.csv order by (plane, Fluid_Sensitive), slices sorted by position along the normal,
2-98% intensity window over the picked slices, 140 mm centre crop, INTER_AREA resize.
Changes: 96 slices (27/21/18/12/18 per slot) and a wider span (6-94%) to reach the peripheral slices
(collateral ligaments, lateral meniscus). Self-check: rebuilds 3 studies with the 44-slice settings
and compares with the Raptor corpus.

Outputs (/kaggle/working): vols_s{SHARD}.npy (N,96,288,288) uint8, masks_s{SHARD}.npy (N,96) uint8,
ids_s{SHARD}.npy, build_s{SHARD}.json.
"""
import os, glob, json, time
import numpy as np, pandas as pd
from multiprocessing import Pool

SHARD, N_SHARDS = 0, 3
IMG = 288
SLOT_SIZES = [27, 21, 18, 12, 18]
SLOT_PLANES = [("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1), ("Coronal", 0), ("Axial", -1)]
PREP = dict(lo=0.06, hi=0.94, crop=140.0)
OUT = "/kaggle/working"
SKIP = ("train_series", "test_series", "train_images", "test_images")


def find(name, root="/kaggle/input"):
    for d, sub, fs in os.walk(root):
        sub[:] = sorted(x for x in sub if x not in SKIP)
        if name in fs:
            return os.path.join(d, name)
    return None


import pydicom, cv2
from pydicom.pixel_data_handlers.util import apply_modality_lut

cv2.setNumThreads(1)


def series_files(sdir):
    recs, pss = [], []
    for f in glob.glob(sdir + "/*.dcm"):
        try:
            h = pydicom.dcmread(f, stop_before_pixels=True)
            iop, ipp = getattr(h, "ImageOrientationPatient", None), getattr(h, "ImagePositionPatient", None)
            if iop is not None and ipp is not None and len(iop) == 6:
                n = np.cross(np.array(iop[:3], float), np.array(iop[3:], float))
                pos = float(np.dot(np.array(ipp, float), n))
            else:
                pos = float(getattr(h, "InstanceNumber", 0) or 0)
            ps = getattr(h, "PixelSpacing", None); ps = float(ps[0]) if ps is not None else 0.5
        except Exception:
            pos, ps = 0.0, 0.5
        recs.append((pos, f, ps)); pss.append(ps)
    recs.sort(key=lambda r: r[0])
    return [(f, ps) for _, f, ps in recs], (float(np.median(pss)) if pss else 0.5)


def read_px(f):
    d = pydicom.dcmread(f)
    a = apply_modality_lut(d.pixel_array, d).astype(np.float32)
    if str(getattr(d, "PhotometricInterpretation", "")) == "MONOCHROME1":
        a = a.max() - a
    return a


def crop_resize(a, ps, crop, img):
    h, w = a.shape
    if crop:
        c = min(int(round(crop / max(ps, 1e-3))), min(h, w))
        y0, x0 = (h - c) // 2, (w - c) // 2
        a = a[y0:y0 + c, x0:x0 + c]
    return cv2.resize(a, (img, img), interpolation=cv2.INTER_AREA)


def pick(rows, plane, fluid, used):
    c = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    if fluid in (0, 1):
        p = [r for r in c if int(r.get("Fluid_Sensitive", 0) or 0) == fluid]
        if p:
            return p[0]
    return c[0] if c else None


def build(sid, rows, img_root, sizes=SLOT_SIZES, prep=PREP, img=IMG):
    D = sum(sizes); vol = np.zeros((D, img, img), np.uint8); idx = 0; used = set(); nerr = 0
    for (plane, fluid), k in zip(SLOT_PLANES, sizes):
        r = pick(rows, plane, fluid, used)
        if r is None:
            idx += k; continue
        used.add(r["SeriesInstanceUID"])
        files, med = series_files(f"{img_root}/{sid}/{r['SeriesInstanceUID']}")
        if not files:
            idx += k; continue
        n = len(files); lo, hi = int(n * prep["lo"]), int(n * prep["hi"]) - 1; hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        cache, arrs = {}, []
        for p in picks:
            f, ps = files[min(p, n - 1)]
            if f not in cache:
                try:
                    cache[f] = read_px(f)
                except Exception:
                    cache[f] = None; nerr += 1
            arrs.append((cache[f], ps))
        ok = [a for a, _ in arrs if a is not None]
        lq, hq = np.percentile(np.concatenate([a.ravel() for a in ok]), [2.0, 98.0]) if ok else (0.0, 1.0)
        for a, ps in arrs:
            if a is not None:
                aw = np.clip((a - lq) / (hq - lq + 1e-6), 0, 1)
                vol[idx] = (crop_resize(aw, ps if ps > 0 else med, prep["crop"], img) * 255).astype(np.uint8)
            idx += 1
    mask = (vol.reshape(D, -1).max(1) > 0).astype(np.uint8)
    return vol, mask, nerr


comp = os.path.dirname(find("train_series.csv"))
ser = pd.read_csv(f"{comp}/train_series.csv")
SER_BY = {str(k): g.to_dict("records") for k, g in ser.groupby("StudyInstanceUID", sort=False)}
IMG_ROOT = f"{comp}/train_series"
tr = pd.read_csv(f"{comp}/train.csv")
all_ids = sorted(tr.StudyInstanceUID.astype(str))
ids = all_ids[SHARD::N_SHARDS]


def job(sid):
    try:
        v, m, e = build(sid, SER_BY.get(sid, []), IMG_ROOT)
        return sid, v, m, e, None
    except Exception as ex:
        return sid, None, None, 0, repr(ex)


def main():
    t0 = time.time()
    # ---- self-check against the Raptor corpus with the 44-slice settings
    rid = find("all_ids.npy")
    if rid:
        cids = np.load(rid, allow_pickle=True).astype(str); cv = np.load(find("all_vols.npy"), mmap_mode="r")
        for row in (0, 1000, 2000):
            v, m, _ = build(cids[row], SER_BY.get(cids[row], []), IMG_ROOT, sizes=[12, 10, 8, 6, 8],
                            prep=dict(lo=0.15, hi=0.85, crop=140.0), img=336)
            r = np.corrcoef(v.ravel().astype(np.float32), np.asarray(cv[row]).ravel().astype(np.float32))[0, 1]
            print(f"[self-check] corpus row {row}: r {r:.4f}", flush=True)
    D = sum(SLOT_SIZES)
    vols = np.lib.format.open_memmap(f"{OUT}/vols_s{SHARD}.npy", mode="w+", dtype=np.uint8, shape=(len(ids), D, IMG, IMG))
    masks = np.zeros((len(ids), D), np.uint8)
    fails, nerr = [], 0
    with Pool(os.cpu_count() or 4) as pool:
        for i, (sid, v, m, e, err) in enumerate(pool.imap(job, ids, chunksize=2)):
            if err:
                fails.append((sid, err))
            else:
                vols[i] = v; masks[i] = m; nerr += e
            if i % 100 == 0:
                print(f"{i}/{len(ids)} | {time.time() - t0:.0f}s | fails {len(fails)} | slice read errors {nerr}", flush=True)
    vols.flush(); del vols
    np.save(f"{OUT}/masks_s{SHARD}.npy", masks)
    np.save(f"{OUT}/ids_s{SHARD}.npy", np.array(ids, dtype=object))
    meta = dict(shard=SHARD, n=len(ids), slots=list(zip([p for p, _ in SLOT_PLANES], SLOT_SIZES)), img=IMG,
                prep=PREP, fails=fails[:50], n_fails=len(fails), slice_read_errors=nerr,
                mean_valid=float(masks.sum(1).mean()), seconds=time.time() - t0)
    json.dump(meta, open(f"{OUT}/build_s{SHARD}.json", "w"), indent=1)
    print("DONE", json.dumps({k: meta[k] for k in ("n", "n_fails", "slice_read_errors", "mean_valid", "seconds")}))


if __name__ == "__main__":
    main()
