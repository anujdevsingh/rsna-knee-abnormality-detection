#!/usr/bin/env python3
"""Pseudo labels v2 (CPU): same rule as v1, from the dense student's out-of-fold predictions (oof_dense.csv).
Gold58 check of the rule: dense 0.9292 vs teacher 0.9213.

v1 notes follow.
Step 3 (CPU): pseudo labels from the teacher's out-of-fold predictions.

Per cell (study x label), with L = merged report label and P = teacher OOF (mirror TTA):
  silent cell (report says nothing)  -> P,                 weight W_SILENT
  mentioned cell                     -> 0.5 * L + 0.5 * P, weight 1
  mentioned and |L - P| >= DROP      -> dropped,           weight 0 (likely label or model error)
Gold rows keep their expert labels (never trained on).

Inputs: merged_labels.parquet (rsna-knee-labels-merge), oof_teacher.csv (rsna-knee-teacher-cnxt).
Output: pseudo_labels.parquet = StudyInstanceUID, 12 labels, 12 `w_<label>` weights, silent_*,
is_gold, fold, fold3; pseudo_stats.json.
"""
import os, glob, json
import numpy as np, pandas as pd

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
ON_KAGGLE = os.path.isdir("/kaggle/input")
ROOT = "/kaggle/input" if ON_KAGGLE else "outputs"
OUT = "/kaggle/working" if ON_KAGGLE else "outputs/pseudo_v2"
os.makedirs(OUT, exist_ok=True)
W_SILENT, DROP, BLEND = 0.5, 0.8, 0.5


def find(name):
    hits = sorted(glob.glob(f"{ROOT}/**/{name}", recursive=True))
    return hits[0] if hits else None


lab = pd.read_parquet(find("merged_labels.parquet")).set_index("StudyInstanceUID")
oof = pd.read_csv(find("oof_dense.csv")).set_index("StudyInstanceUID")[LAB]
print(f"labels {len(lab)} | oof {len(oof)} | gold {int(lab.is_gold.sum())}")
assert lab.index[lab.is_gold == 0].isin(oof.index).all(), "every non-gold study needs an OOF prediction"

out = lab.copy()
stats = {}
ng = lab.index[lab.is_gold == 0]
for l in LAB:
    L = lab.loc[ng, l].values; P = oof.loc[ng, l].values
    S = lab.loc[ng, f"silent_{l}"].values.astype(bool)
    drop = (~S) & (np.abs(L - P) >= DROP)
    y = np.where(S, P, BLEND * L + (1 - BLEND) * P)
    w = np.where(S, W_SILENT, 1.0); w[drop] = 0.0
    out.loc[ng, l] = y
    out[f"w_{l}"] = 1.0
    out.loc[ng, f"w_{l}"] = w
    stats[l] = dict(silent=float(S.mean()), dropped=float(drop.mean()),
                    mean_label=float(L.mean()), mean_pseudo=float(y.mean()))
    print(f"{l:18s} silent {S.mean():.3f} | dropped {drop.mean():.4f} | mean label {L.mean():.3f} -> {y.mean():.3f}")

tot_drop = float(np.mean([stats[l]["dropped"] for l in LAB]))
print(f"cells dropped overall: {tot_drop:.4f}")
stats["_overall_dropped"] = tot_drop
stats["_params"] = dict(W_SILENT=W_SILENT, DROP=DROP, BLEND=BLEND)
out.reset_index().to_parquet(f"{OUT}/pseudo_labels_v2.parquet", index=False)
json.dump(stats, open(f"{OUT}/pseudo_stats.json", "w"), indent=1)
print(f"wrote {OUT}/pseudo_labels_v2.parquet")
