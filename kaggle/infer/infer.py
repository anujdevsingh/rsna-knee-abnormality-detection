#!/usr/bin/env python3
"""Step 5 (GPU, internet off): build 44-slice stacks from the test DICOMs exactly like the training
corpus, run every fold checkpoint with mirror TTA, average, and write submission.csv.

Preprocessing settings come from kaggle/prep_check (best match to the corpus). Model members are the
*.pt checkpoints found under /kaggle/input (student folds, optionally the teacher folds); within a
member set probabilities are averaged over folds, across member sets they are rank-averaged.
DICOM reading runs in a thread pool on CPU while the GPUs score finished studies.
"""
import os, glob, fnmatch, json, time, threading, queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
IN = "/kaggle/input"
OUT = "/kaggle/working"

# ---- preprocessing per training corpus: must match how that corpus was built
SLOT_PLANES = [("Sagittal", 1), ("Sagittal", 0), ("Coronal", 1), ("Coronal", 0), ("Axial", -1)]
BUILDS = {
    # Raptor 44-slice corpus; exact match (prep_check: r 1.0000, MAD 0.00)
    "raptor44": dict(sizes=[12, 10, 8, 6, 8], lo=0.15, hi=0.85, crop=140.0, img=336),
    # our dense corpus (kaggle/corpus_dense)
    "dense96": dict(sizes=[27, 21, 18, 12, 18], lo=0.06, hi=0.94, crop=140.0, img=288),
    # our dense corpus at 384 px (kaggle/corpus384)
    "dense384": dict(sizes=[27, 21, 18, 12, 18], lo=0.06, hi=0.94, crop=140.0, img=384),
}
MEMBER_SETS = {
    "teacher": dict(glob="teacher_fold*.pt", build="raptor44"),
    "student": dict(glob="student_fold*.pt", build="raptor44"),
    "dense": dict(glob="dense_fold*.pt", build="dense96"),
    "s384": dict(glob="s384_fold*.pt", build="dense384"),
    "densev2": dict(glob="densev2_fold*.pt", build="dense96"),
    "effv2s": dict(glob="effv2s_fold*.pt", build="dense96"),
    "densev2f": dict(glob="densev2f_fold*.pt", build="dense96"),
}
USE_SETS = ["densev2f", "effv2s"]   # v7: rank blend D3 + EfficientNetV2-S (E1)


def find(name, root=IN):
    for d, sub, fs in os.walk(root):
        sub[:] = sorted(x for x in sub if x not in ("train_series", "test_series", "train_images", "test_images"))
        if name in fs:
            return os.path.join(d, name)
    return None


def find_glob(pattern, root=IN):
    hits = []
    for d, sub, fs in os.walk(root):
        sub[:] = [x for x in sub if x not in ("train_series", "test_series", "train_images", "test_images")]
        hits += [os.path.join(d, f) for f in fs if fnmatch.fnmatch(f, pattern)]
    return sorted(hits)


# ============================================================================ DICOM -> stack
import pydicom, cv2
from pydicom.pixel_data_handlers.util import apply_modality_lut


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


def build(sid, rows, img_root, B):
    D = sum(B["sizes"]); vol = np.zeros((D, B["img"], B["img"]), np.uint8); idx = 0; used = set()
    for (plane, fluid), k in zip(SLOT_PLANES, B["sizes"]):
        r = pick(rows, plane, fluid, used)
        if r is None:
            idx += k; continue
        used.add(r["SeriesInstanceUID"])
        files, med = series_files(f"{img_root}/{sid}/{r['SeriesInstanceUID']}")
        if not files:
            idx += k; continue
        n = len(files); lo, hi = int(n * B["lo"]), int(n * B["hi"]) - 1; hi = max(hi, lo)
        picks = np.linspace(lo, hi, k).round().astype(int) if n > 1 else [0] * k
        arrs = []
        for p in picks:
            f, ps = files[min(p, n - 1)]
            try:
                arrs.append((read_px(f), ps))
            except Exception:
                arrs.append((None, ps))
        ok = [a for a, _ in arrs if a is not None]
        lq, hq = np.percentile(np.concatenate([a.ravel() for a in ok]), [2.0, 98.0]) if ok else (0.0, 1.0)
        for a, ps in arrs:
            if a is not None:
                aw = np.clip((a - lq) / (hq - lq + 1e-6), 0, 1)
                vol[idx] = (crop_resize(aw, ps if ps > 0 else med, B["crop"], B["img"]) * 255).astype(np.uint8)
            idx += 1
    mask = (vol.reshape(D, -1).max(1) > 0).astype(np.uint8)
    return vol, mask


# ============================================================================ model
import torch, torch.nn as nn, torch.nn.functional as F
import timm

torch.backends.cudnn.benchmark = True


class Net(nn.Module):
    def __init__(self, arch, n=len(LAB)):
        super().__init__()
        self.bb = timm.create_model(arch, pretrained=False, num_classes=0)
        f = self.bb.num_features
        self.norm = nn.LayerNorm(f)
        self.att = nn.Sequential(nn.Linear(f, 256), nn.Tanh(), nn.Linear(256, n))
        self.W = nn.Parameter(torch.zeros(n, f))
        self.b = nn.Parameter(torch.zeros(n))

    def forward(self, x, B, K):
        h = self.norm(self.bb(x).view(B, K, -1))
        a = torch.softmax(self.att(h).float(), dim=1)
        pooled = torch.einsum("bkn,bkf->bnf", a, h.float())
        return (pooled * self.W).sum(-1) + self.b


def slot_bounds(slots):
    t, s = [], 0
    for plane, n in slots:
        t.append((plane, s, s + n)); s += n
    return t


def windows(vol, mask, slots, k, flip):
    bounds = slot_bounds(slots); D = vol.shape[0]
    cs = [c for _, a, b in bounds for c in range(a + 1, b - 1) if mask[c - 1] and mask[c] and mask[c + 1]]
    if not cs:
        cs = [D // 2]
    cs = [cs[i] for i in np.linspace(0, len(cs) - 1, k).round().astype(int)]
    x = np.stack([vol[c - 1:c + 2] for c in cs])
    if flip:
        sag = np.zeros(D, bool)
        for plane, a, b in bounds:
            sag[a:b] = plane == "Sagittal"
        s_ = sag[cs]
        x[s_] = x[s_][:, ::-1]
        x[~s_] = x[~s_][..., ::-1]
    return np.ascontiguousarray(x)


def load_members(dev):
    sets = {}
    for name in USE_SETS:
        paths = find_glob(MEMBER_SETS[name]["glob"])
        ms = []
        for p in paths:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            m = Net(ck["cfg"]["arch"]); m.load_state_dict(ck["model"]); m.eval().to(dev)
            if dev != "cpu":
                m.half()
            ms.append((m, ck["cfg"]["res"], ck["slots"], ck["cfg"].get("k_eval", 24)))
        print(f"[members] {name}: {len(ms)} checkpoints {[os.path.basename(p) for p in paths]}")
        if ms:
            sets[name] = ms
    return sets


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def score(members, stacks, dev):
    out = {}
    for name, ms in members.items():
        vol, mask = stacks[MEMBER_SETS[name]["build"]]
        ps = []
        for m, res, slots, k_eval in ms:
            assert sum(n for _, n in slots) == vol.shape[0], f"{name}: slots do not match the stack"
            for flip in (False, True):
                x = torch.from_numpy(windows(vol, mask, slots, k_eval, flip)).to(dev).float().div_(255)
                x = F.interpolate(x, size=(res, res), mode="bilinear", align_corners=False)
                x = (x - MEAN.to(dev)) / STD.to(dev)
                if dev != "cpu":
                    x = x.half()
                ps.append(torch.sigmoid(m(x, 1, x.shape[0]).float())[0].cpu().numpy())
        out[name] = np.mean(ps, 0)
    return out


# ============================================================================ main
def main():
    t0 = time.time()
    comp = os.path.dirname(find("test_series.csv"))
    test = pd.read_csv(f"{comp}/test.csv"); test["StudyInstanceUID"] = test["StudyInstanceUID"].astype(str)
    ser = pd.read_csv(f"{comp}/test_series.csv")
    ser_by = {str(k): g.to_dict("records") for k, g in ser.groupby("StudyInstanceUID", sort=False)}
    img_root = next(p for p in [f"{comp}/test_series", f"{comp}/test_images", f"{comp}/test"] if os.path.isdir(p))
    sids = test.StudyInstanceUID.tolist()
    ngpu = max(1, torch.cuda.device_count())
    devs = [f"cuda:{i}" for i in range(ngpu)] if torch.cuda.is_available() else ["cpu"]
    members = [load_members(d) for d in devs]
    print(f"test studies {len(sids)} | devices {devs} | {time.time() - t0:.0f}s", flush=True)

    q = queue.Queue(maxsize=16)
    preds, errors = {}, []

    need = sorted({MEMBER_SETS[n]["build"] for n in USE_SETS})
    print(f"stack builds per study: {need}", flush=True)

    def build_all(s):
        return {b: build(s, ser_by.get(s, []), img_root, BUILDS[b]) for b in need}

    def builder():
        with ThreadPoolExecutor(4) as ex:
            futs = {ex.submit(build_all, s): s for s in sids}
            for f in futs:
                s = futs[f]
                try:
                    q.put((s, f.result()))
                except Exception as e:
                    errors.append((s, repr(e))); q.put((s, None))
        for _ in devs:
            q.put(None)

    def worker(i):
        while True:
            item = q.get()
            if item is None:
                return
            s, stacks = item
            if stacks is None:
                continue
            try:
                preds[s] = score(members[i], stacks, devs[i])
            except Exception as e:
                errors.append((s, repr(e)))

    th = [threading.Thread(target=builder)] + [threading.Thread(target=worker, args=(i,)) for i in range(len(devs))]
    [t.start() for t in th]; [t.join() for t in th]
    print(f"scored {len(preds)}/{len(sids)} | errors {len(errors)} {errors[:3]} | {time.time() - t0:.0f}s", flush=True)

    sub = pd.DataFrame({"StudyInstanceUID": sids})
    names = [n for n in USE_SETS if any(n in p for p in preds.values())]
    blend = np.zeros((len(sids), len(LAB)))
    for n in names:
        P = np.array([preds[s][n] if s in preds else np.full(len(LAB), np.nan) for s in sids])
        P = np.where(np.isnan(P), np.nanmean(P, 0) if np.isfinite(P).any() else 0.5, P)
        blend += pd.DataFrame(P).rank(pct=True).values if len(names) > 1 else P
    blend /= max(1, len(names))
    sub[LAB] = blend
    sub.to_csv(f"{OUT}/submission.csv", index=False)
    print(sub.head(), flush=True)
    print(f"wrote submission.csv ({len(sub)} rows) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
