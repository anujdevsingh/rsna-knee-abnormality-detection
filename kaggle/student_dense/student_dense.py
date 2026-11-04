#!/usr/bin/env python3
"""Dense student (GPU, T4 x2): same model as student_cnxt, trained on our 96-slice 288 px corpus
(kaggle/corpus_dense, 3 shard kernels) instead of the 44-slice Raptor corpus.

Student notes follow.
Step 4 (GPU, T4 x2): ConvNeXt-T 2.5D student at 288 px, 5 folds, trained on the step-3 pseudo labels.

Same model and data path as the teacher (kaggle/teacher_cnxt). Differences: targets and per-cell
weights come from pseudo_labels.parquet (w_<label>: 1 mentioned, 0.5 silent, 0 dropped); evaluation
stays on the original merged report labels so scores compare with the teacher; OOM at the first step
switches on gradient checkpointing.

Teacher notes follow.

Data: dreaddevelopment/knee-raptor-corpus (+ -ext), 4,407 studies as (44, 336, 336) uint8 stacks
plus a (44,) valid-slice mask. Labels: our merged soft labels (kernel output of rsna-knee-labels-merge).

Per study, K windows of 3 consecutive slices from one slot (one series) form a 3-channel image.
Each window goes through the backbone; a per-label attention layer pools the K features.
Loss: soft-target BCE, silent cells (report says nothing) down-weighted to SILENT_W.
Augmentation: anatomical left/right mirror (sagittal slots reverse slice order, coronal/axial flip
the image width), mild gain jitter. Selection: best epoch by macro AUC on the held-out fold's weak
labels (forum: tracks the LB better than gold58); gold58 is reported alongside.

The script runs one fold per GPU in parallel subprocesses (no DataParallel). Outputs in
/kaggle/working: oof_fold{f}.csv, gold_fold{f}.csv, metrics_fold{f}.json, teacher_fold{f}.pt,
then oof_teacher.csv and summary.json once all folds finish.
Uses parse_known_args because notebook kernels pass extra argv.
"""
import os, sys, glob, json, time, math, random, argparse, subprocess
import numpy as np, pandas as pd

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
OUT = os.environ.get("OUT_DIR", "/kaggle/working")
IN_ROOT = os.environ.get("IN_ROOT", "/kaggle/input")

# Slot layout of the 44-slice corpus: (plane, n_slices) in stack order. Confirmed 2026-10-14 by the
# corpus probe in rsna-knee-labels-merge v2: empty runs at slices 12-21 (2nd sagittal, 3% of studies)
# and 30-35 (2nd coronal, 26%). Plane order follows the Raptor builder (sag, sag, cor, cor, axial).
SLOTS = [("Sagittal", 27), ("Sagittal", 21), ("Coronal", 18), ("Coronal", 12), ("Axial", 18)]   # dense corpus

CFG = dict(
    arch="convnext_tiny.fb_in22k_ft_in1k", res=288, k=16, k_eval=32, epochs=8, bs=4,
    bb_lr=1e-4, head_lr=1e-3, wd=0.05, warmup=0.1, silent_w=0.3, mirror_p=0.5,
    workers=4, seed=42, n_folds=4, fold_col="fold4", train_labels="pseudo_labels.parquet",
)


def find(name, root=None):
    root = root or IN_ROOT
    hits = sorted(glob.glob(f"{root}/**/{name}", recursive=True))
    return hits[0] if hits else None


def slot_table(D):
    assert sum(n for _, n in SLOTS) == D, f"SLOTS sum {sum(n for _, n in SLOTS)} != corpus depth {D}"
    t, s = [], 0
    for plane, n in SLOTS:
        t.append((plane, s, s + n)); s += n
    return t


# ============================================================================ worker (one fold)
def run_fold(fold, gpu, cfg, smoke=False):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    import torch, torch.nn as nn, torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    from sklearn.metrics import roc_auc_score
    import timm

    random.seed(cfg["seed"] + fold); np.random.seed(cfg["seed"] + fold)
    torch.manual_seed(cfg["seed"] + fold)
    torch.backends.cudnn.benchmark = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log = open(f"{OUT}/log_fold{fold}.txt", "w")

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(f"[fold{fold}] {s}", flush=True); log.write(s + "\n"); log.flush()

    # ---- corpus (memmap, two parts presented as one)
    parts = [np.load(find(f"vols_s{k}.npy"), mmap_mode="r") for k in range(3)]
    masks = np.concatenate([np.load(find(f"masks_s{k}.npy")) for k in range(3)])
    ids = np.concatenate([np.load(find(f"ids_s{k}.npy"), allow_pickle=True).astype(str) for k in range(3)])
    starts = np.cumsum([0] + [p.shape[0] for p in parts])
    D = parts[0].shape[1]
    slots = slot_table(D)
    id2row = {u: i for i, u in enumerate(ids)}

    lab = pd.read_parquet(find("merged_labels.parquet"))
    lab["StudyInstanceUID"] = lab["StudyInstanceUID"].astype(str)
    lab = lab[lab.StudyInstanceUID.isin(id2row)].reset_index(drop=True)
    Yall = lab[LAB].values.astype(np.float32)                      # eval truth: report labels
    Sall = lab[[f"silent_{l}" for l in LAB]].values.astype(np.float32)
    pl = pd.read_parquet(find(cfg["train_labels"]))
    pl["StudyInstanceUID"] = pl["StudyInstanceUID"].astype(str)
    pl = pl.set_index("StudyInstanceUID").loc[lab.StudyInstanceUID]
    Tall = pl[LAB].values.astype(np.float32)                        # training targets
    Wall = pl[[f"w_{l}" for l in LAB]].values.astype(np.float32)    # per-cell weights
    fcol = lab[cfg["fold_col"]].values
    tr_idx = np.where((fcol >= 0) & (fcol != fold))[0]
    va_idx = np.where(fcol == fold)[0]
    gd_idx = np.where(lab.is_gold.values == 1)[0]
    if smoke:
        tr_idx, va_idx = tr_idx[:48], va_idx[:24]
        cfg = dict(cfg, epochs=1, k=4, k_eval=6)
    say(f"train {len(tr_idx)} | val {len(va_idx)} | gold {len(gd_idx)} | D {D} | slots {slots}")

    def vol_of(row):
        k = int(np.searchsorted(starts, row, side="right") - 1)
        return parts[k][row - starts[k]]

    def centers(m, k, train):
        # window centres whose 3 slices are valid and inside one slot (one series)
        cs = [c for _, a, b in slots for c in range(a + 1, b - 1) if m[c - 1] and m[c] and m[c + 1]]
        if not cs:
            cs = [D // 2]
        if train:
            return [cs[i] for i in np.random.randint(0, len(cs), k)]
        return [cs[i] for i in np.linspace(0, len(cs) - 1, k).round().astype(int)]

    # anatomical L/R mirror, per window: a sagittal window's slices run across the knee, so the
    # mirror reverses their order (channels); coronal/axial windows flip the image width. Window
    # positions do not need remapping because the attention pool is order-invariant.
    sag = np.zeros(D, bool)
    for plane, a_, b_ in slots:
        sag[a_:b_] = plane == "Sagittal"

    class DS(Dataset):
        def __init__(self, idx, k, train, flip=False):
            self.idx, self.k, self.train, self.flip = idx, k, train, flip

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            j = self.idx[i]
            row = id2row[lab.StudyInstanceUID.iat[j]]
            vol = vol_of(row)                                       # memmap view, nothing read yet
            cs = centers(masks[row], self.k, self.train)
            x = np.stack([np.asarray(vol[c - 1:c + 2]) for c in cs])   # (K,3,H,W) uint8
            if self.flip or (self.train and random.random() < cfg["mirror_p"]):
                s_ = sag[cs]
                x[s_] = x[s_][:, ::-1]
                x[~s_] = x[~s_][..., ::-1]
            return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(Tall[j]), \
                torch.from_numpy(Wall[j])

    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)

    def prep(x, train):
        # uint8 (B,K,3,H,W) -> normalised float (B*K,3,res,res) on GPU
        B, K = x.shape[:2]
        x = x.to(dev, non_blocking=True).flatten(0, 1).float().div_(255)
        if train:
            x = x * (1 + (torch.rand(B * K, 1, 1, 1, device=dev) - 0.5) * 0.2)
        x = F.interpolate(x, size=(cfg["res"], cfg["res"]), mode="bilinear", align_corners=False)
        return ((x.clamp(0, 1) - mean) / std), B, K

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.bb = timm.create_model(cfg["arch"], pretrained=os.environ.get("NO_PRETRAIN") != "1", num_classes=0)
            f = self.bb.num_features
            self.norm = nn.LayerNorm(f)
            self.att = nn.Sequential(nn.Linear(f, 256), nn.Tanh(), nn.Linear(256, len(LAB)))
            self.W = nn.Parameter(torch.randn(len(LAB), f) * 0.02)
            self.b = nn.Parameter(torch.zeros(len(LAB)))

        def forward(self, x, B, K):
            h = self.norm(self.bb(x).view(B, K, -1))                    # (B,K,F)
            a = torch.softmax(self.att(h).float(), dim=1)               # (B,K,12)
            pooled = torch.einsum("bkn,bkf->bnf", a, h.float())
            return (pooled * self.W).sum(-1) + self.b                   # (B,12)

    model = Net().to(dev)
    ldr = lambda ds, bs, sh: DataLoader(ds, batch_size=bs, shuffle=sh, num_workers=cfg["workers"],
                                        pin_memory=True, drop_last=sh, persistent_workers=False)
    tl = ldr(DS(tr_idx, cfg["k"], True), cfg["bs"], True)
    bb_p = [p for n, p in model.named_parameters() if n.startswith("bb.")]
    hd_p = [p for n, p in model.named_parameters() if not n.startswith("bb.")]
    opt = torch.optim.AdamW([{"params": bb_p, "lr": cfg["bb_lr"]},
                             {"params": hd_p, "lr": cfg["head_lr"]}], weight_decay=cfg["wd"])
    total = max(1, len(tl) * cfg["epochs"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[cfg["bb_lr"], cfg["head_lr"]],
                                                total_steps=total, pct_start=cfg["warmup"])
    scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")

    @torch.no_grad()
    def predict(idx, tta=True):
        model.eval(); out = []
        for flip in ([False, True] if tta else [False]):
            P = []
            for x, _, _ in ldr(DS(idx, cfg["k_eval"], False, flip=flip), 4, False):
                xx, B, K = prep(x, False)
                with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                    P.append(torch.sigmoid(model(xx, B, K).float()).cpu().numpy())
            out.append(np.concatenate(P))
        return np.mean(out, 0)

    def weak_auc(P, idx):
        # truth = merged label >= 0.5 on cells the report mentions
        Y, S = Yall[idx], Sall[idx]; aucs = {}
        for j, l in enumerate(LAB):
            m = S[:, j] == 0; y = (Y[m, j] >= 0.5).astype(int)
            if 0 < y.sum() < len(y):
                aucs[l] = roc_auc_score(y, P[m, j])
        return float(np.mean(list(aucs.values()))), aucs

    def gold_auc(P):
        Y = Yall[gd_idx]
        aucs = {l: roc_auc_score(Y[:, j], P[:, j]) for j, l in enumerate(LAB) if 0 < Y[:, j].sum() < len(Y)}
        return float(np.mean(list(aucs.values()))), aucs

    best, hist = -1, []
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train(); tot, n = 0.0, 0
        for x, y, w in tl:
            xx, B, K = prep(x, True)
            y, w = y.to(dev), w.to(dev)

            def step():
                with torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
                    logit = model(xx, B, K)
                loss = (F.binary_cross_entropy_with_logits(logit.float(), y, reduction="none") * w).sum() \
                    / w.sum().clamp_min(1.0)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                return loss

            try:
                loss = step()
            except torch.cuda.OutOfMemoryError:
                if ep or n:
                    raise
                say("OOM on first step: enabling gradient checkpointing")
                opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                model.bb.set_grad_checkpointing(True)
                loss = step()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item(); n += 1
        Pv = predict(va_idx, tta=False)
        wa, _ = weak_auc(Pv, va_idx)
        ga, _ = gold_auc(predict(gd_idx, tta=False))
        hist.append(dict(ep=ep, loss=tot / max(n, 1), weak_auc=wa, gold_auc=ga, sec=time.time() - t0))
        say(f"ep{ep} loss {tot / max(n, 1):.4f} | weak AUC {wa:.4f} | gold58 {ga:.4f} | {time.time() - t0:.0f}s")
        if wa > best:
            best = wa
            torch.save({"model": model.state_dict(), "cfg": cfg, "slots": SLOTS, "lab": LAB, "epoch": ep},
                       f"{OUT}/dense_fold{fold}.pt")

    # ---- OOF + gold at the best epoch, with mirror TTA
    model.load_state_dict(torch.load(f"{OUT}/dense_fold{fold}.pt", map_location=dev)["model"])
    Po, Pg = predict(va_idx), predict(gd_idx)
    wa, wper = weak_auc(Po, va_idx); ga, gper = gold_auc(Pg)
    uid = lab.StudyInstanceUID.values
    pd.DataFrame(Po, columns=LAB).assign(StudyInstanceUID=uid[va_idx], fold=fold).to_csv(
        f"{OUT}/oof_fold{fold}.csv", index=False)
    pd.DataFrame(Pg, columns=LAB).assign(StudyInstanceUID=uid[gd_idx], fold=fold).to_csv(
        f"{OUT}/gold_fold{fold}.csv", index=False)
    json.dump(dict(fold=fold, weak_auc_tta=wa, gold_auc_tta=ga, weak_per=wper, gold_per=gper,
                   hist=hist, cfg=cfg, slots=SLOTS), open(f"{OUT}/metrics_fold{fold}.json", "w"), indent=1)
    say(f"BEST (mirror TTA) weak AUC {wa:.4f} | gold58 {ga:.4f} | total {time.time() - t0:.0f}s")


# ============================================================================ driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a, _ = ap.parse_known_args()
    if a.fold >= 0:
        return run_fold(a.fold, a.gpu, CFG, a.smoke)

    # Kaggle runs scripts as a converted notebook (no __file__), so folds run in forked processes.
    # CUDA is never initialised in this parent; each child pins its GPU before importing torch.
    import multiprocessing as mp
    try:
        ngpu = len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines())
    except Exception:
        ngpu = 0
    ngpu = max(1, ngpu)
    smoke = a.smoke or os.environ.get("SMOKE") == "1"
    print(f"GPUs {ngpu} | cfg {CFG} | smoke {smoke}", flush=True)
    ctx = mp.get_context("fork")
    pending = list(range(CFG["n_folds"]))
    running = []
    while pending or running:
        while pending and len(running) < ngpu:
            used = {g for g, _, _ in running}
            g = next(i for i in range(ngpu) if i not in used)
            f = pending.pop(0)
            p = ctx.Process(target=run_fold, args=(f, g, CFG, smoke))
            p.start()
            running.append((g, f, p))
        time.sleep(5)
        for r in list(running):
            if not r[2].is_alive():
                r[2].join()
                print(f"fold {r[1]} exit {r[2].exitcode}", flush=True)
                if r[2].exitcode != 0:
                    raise SystemExit(f"fold {r[1]} failed")
                running.remove(r)

    from sklearn.metrics import roc_auc_score
    oof = pd.concat([pd.read_csv(f"{OUT}/oof_fold{f}.csv") for f in range(CFG["n_folds"])])
    oof.to_csv(f"{OUT}/oof_dense.csv", index=False)
    gold = pd.concat([pd.read_csv(f"{OUT}/gold_fold{f}.csv") for f in range(CFG["n_folds"])])
    gold_avg = gold.groupby("StudyInstanceUID")[LAB].mean()
    lab = pd.read_parquet(find("merged_labels.parquet")).set_index("StudyInstanceUID")
    yg = lab.loc[gold_avg.index, LAB].values
    g_auc = np.mean([roc_auc_score(yg[:, j], gold_avg.values[:, j]) for j in range(len(LAB))])
    ms = [json.load(open(f"{OUT}/metrics_fold{f}.json")) for f in range(CFG["n_folds"])]
    summ = dict(weak_auc_folds=[m["weak_auc_tta"] for m in ms], weak_auc_mean=float(np.mean([m["weak_auc_tta"] for m in ms])),
                gold58_fold_avg_ensemble=float(g_auc), n_oof=len(oof))
    json.dump(summ, open(f"{OUT}/summary.json", "w"), indent=1)
    print("SUMMARY", json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
