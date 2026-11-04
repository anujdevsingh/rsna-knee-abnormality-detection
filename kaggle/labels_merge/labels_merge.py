#!/usr/bin/env python3
"""Step 1 (CPU): merge the public CC0 LLM label sets into soft labels + a silent-cell mask.

Inputs (Kaggle: /kaggle/input/..., local: data/):
  train.csv                                   competition (58 gold studies)
  pilkwang  report_labels_v2.csv              value + __conf + __verdict (YES/NO/UNK)
  steven    llm_labels_{full,v2,v4_blend}.csv value only
  lixin     report_labels_gpt56sol.csv        value + __conf
  (barun is not used: its pseudo_* columns are 0.5 everywhere except the 58 gold rows.)

Outputs (Kaggle: /kaggle/working, local: outputs/labels/):
  merged_labels.parquet  StudyInstanceUID, 12 soft labels, 12 `silent_<label>` flags (1 = report silent)
  label_scores.json      gold58 macro AUC per set and per merge variant
  folds.csv              StudyInstanceUID, fold (0..4) and fold3 (0..2), gold = -1,, stratified on the merged labels

If the Raptor corpus is attached, it also prints its layout (ids, masks, window_index), which the
teacher and the inference notebook need. No volumes are read.
"""
import os, glob, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
ON_KAGGLE = os.path.isdir("/kaggle/input")
ROOT = "/kaggle/input" if ON_KAGGLE else "data"
OUT = "/kaggle/working" if ON_KAGGLE else "outputs/labels"
os.makedirs(OUT, exist_ok=True)
N_FOLDS, SEED = 5, 42


def find(name):
    # walk the inputs but never descend into DICOM folders (the competition holds 819k files)
    for d, sub, fs in os.walk(ROOT):
        sub[:] = sorted(x for x in sub if x not in ("train_series", "test_series", "train_images", "test_images"))
        if name in fs:
            return os.path.join(d, name)
    return None


def load_set(name):
    p = find(name)
    if p is None:
        print(f"[skip] {name} not found")
        return None
    d = pd.read_csv(p)
    d["StudyInstanceUID"] = d["StudyInstanceUID"].astype(str)
    return d.drop_duplicates("StudyInstanceUID").set_index("StudyInstanceUID")


def macro_auc(y, p):
    aucs = {}
    for j, l in enumerate(LAB):
        m = ~np.isnan(p[:, j])
        if len(np.unique(y[m, j])) > 1:
            aucs[l] = roc_auc_score(y[m, j], p[m, j])
    return float(np.mean(list(aucs.values()))), aucs


def rank01(s):
    return s.rank(pct=True, method="average")


# ---------------------------------------------------------------- inputs
tr = pd.read_csv(find("train.csv"))
tr["StudyInstanceUID"] = tr["StudyInstanceUID"].astype(str)
tr = tr.set_index("StudyInstanceUID")
uids = tr.index
gold = tr[tr[LAB].notna().all(axis=1)]
print(f"train {len(tr)} | gold {len(gold)}")

raw = {
    "pilkwang": load_set("report_labels_v2.csv"),
    "steven_full": load_set("llm_labels_full.csv"),
    "steven_v2": load_set("llm_labels_v2.csv"),
    "steven_v4": load_set("llm_labels_v4_blend.csv"),
    "lixin": load_set("report_labels_gpt56sol.csv"),
}
raw = {k: v for k, v in raw.items() if v is not None}
sets = {k: v.reindex(uids)[LAB].astype(float) for k, v in raw.items()}

# ---------------------------------------------------------------- score each set on gold58
scores = {}
yg = gold[LAB].values.astype(float)
for k, v in sets.items():
    pg = v.loc[gold.index].values
    au, per = macro_auc(yg, pg)
    # leak check: a set that copies gold scores ~1.0 and is useless for validation
    exact = float(np.nanmean(np.abs(pg - yg) < 1e-6))
    scores[k] = {"gold_auc": round(au, 4), "exact_match_gold": round(exact, 3),
                 "per_label": {l: round(a, 3) for l, a in per.items()}}
    print(f"{k:12s} gold macro AUC {au:.4f} | exact-match-gold {exact:.2f} | coverage "
          f"{1 - v.isna().values.mean():.3f}")

# one file per author so no source is counted twice (steven files are versions of one labeller)
steven_best = max([k for k in sets if k.startswith("steven")], key=lambda k: scores[k]["gold_auc"],
                  default=None)
members = [k for k in ["pilkwang", steven_best, "lixin"] if k in sets]
print("merge members:", members)

# ---------------------------------------------------------------- silent mask
# pilkwang gives an explicit verdict; UNK = the report does not mention the finding.
pk = raw["pilkwang"].reindex(uids)
silent = pd.DataFrame({l: (pk[f"{l}__verdict"].fillna("UNK") == "UNK").astype(np.uint8) for l in LAB},
                      index=uids)
print("silent fraction per label:", silent.mean().round(3).to_dict())
print(f"silent fraction overall: {silent.values.mean():.3f}")

# ---------------------------------------------------------------- merge variants
# mean: plain probability mean. rank: mean of per-label percentile ranks, mapped back onto the
# members' pooled value distribution (keeps each label's prevalence, which BCE targets need).
M = np.stack([sets[k].values for k in members])               # (S, N, 12)
merged_mean = np.nanmean(M, axis=0)
R = np.stack([sets[k].apply(rank01).values for k in members])
rmean = np.nanmean(R, axis=0)
merged_rank = np.empty_like(rmean)
for j in range(len(LAB)):
    pool = np.sort(merged_mean[:, j][~np.isnan(merged_mean[:, j])])
    order = pd.Series(rmean[:, j]).rank(pct=True).values
    merged_rank[:, j] = np.quantile(pool, np.clip(order, 0, 1))
variants = {"merge_mean": merged_mean, "merge_rank": merged_rank}

gi = uids.get_indexer(gold.index)
for k, v in variants.items():
    au, per = macro_auc(yg, v[gi])
    scores[k] = {"gold_auc": round(au, 4), "per_label": {l: round(a, 3) for l, a in per.items()}}
    print(f"{k:12s} gold macro AUC {au:.4f}")

# pairwise agreement on non-silent cells (Spearman over all cells, per label averaged)
agree = {}
for a_ in members:
    for b_ in members:
        if a_ < b_:
            agree[f"{a_}~{b_}"] = round(float(np.nanmean(
                [sets[a_][l].corr(sets[b_][l], method="spearman") for l in LAB])), 3)
print("spearman agreement:", agree)
scores["agreement_spearman"] = agree
scores["members"] = members

# ---------------------------------------------------------------- write labels
final = pd.DataFrame(merged_rank, index=uids, columns=LAB)
final.loc[gold.index, LAB] = gold[LAB].values.astype(float)   # gold rows keep the expert labels
for l in LAB:
    final[f"silent_{l}"] = silent[l].values
final["is_gold"] = final.index.isin(gold.index).astype(np.uint8)
final = final.reset_index()

# ---------------------------------------------------------------- folds (gold = -1, never trained on)
# iterative greedy stratification on binarised merged labels: assign each study to the fold that
# most needs its positives. fold (5) is for students, fold3 for the 3-fold teacher.
def strat_folds(n_folds, seed=SEED):
    rng = np.random.RandomState(seed)
    nong = final[final.is_gold == 0].index.values
    Y = (final.loc[nong, LAB].values >= 0.5).astype(int)
    need = np.tile(Y.sum(0) / n_folds, (n_folds, 1)).astype(float)
    size_need = np.full(n_folds, len(nong) / n_folds)
    fold = np.full(len(final), -1)
    order = rng.permutation(len(nong))
    order = order[np.argsort(-Y[order].sum(1), kind="stable")]   # positive-rich studies first
    for i in order:
        f = int(np.argmax((need * Y[i]).sum(1) + 1e-3 * size_need))
        fold[nong[i]] = f
        need[f] -= Y[i]; size_need[f] -= 1
    return fold

final["fold"] = strat_folds(N_FOLDS)
final["fold3"] = strat_folds(3)
final["fold4"] = strat_folds(4)   # dense student: 4 folds = 2 rounds on T4 x2
print("fold sizes:", final.fold.value_counts().sort_index().to_dict(),
      "| fold3:", final.fold3.value_counts().sort_index().to_dict())

final.to_parquet(f"{OUT}/merged_labels.parquet", index=False)
final[["StudyInstanceUID", "fold", "fold3", "fold4"]].to_csv(f"{OUT}/folds.csv", index=False)
json.dump(scores, open(f"{OUT}/label_scores.json", "w"), indent=1)
print(f"wrote {OUT}/merged_labels.parquet, folds.csv, label_scores.json")

# ---------------------------------------------------------------- Raptor corpus layout probe
ids_p, ext_p = find("all_ids.npy"), find("extra_ids.npy")
if ids_p:
    ids = np.concatenate([np.load(ids_p, allow_pickle=True).astype(str)]
                         + ([np.load(ext_p, allow_pickle=True).astype(str)] if ext_p else []))
    print(f"[corpus] ids {ids.shape} | unique {len(set(ids))} | in train {np.isin(ids, uids).sum()}"
          f" | gold present {np.isin(gold.index, ids).sum()}")
    masks = np.concatenate([np.load(find("all_masks.npy"))]
                           + ([np.load(find("extra_masks.npy"))] if ext_p else []))
    print(f"[corpus] masks {masks.shape} {masks.dtype} | mean valid slices {masks.sum(1).mean():.1f}")
    # slot layout: slices that are empty in many studies mark the slot boundaries
    print("[corpus] fraction valid per slice index:", np.round(masks.mean(0), 2).tolist())
    # contiguous empty runs per study show which slot was missing
    runs = {}
    for m in masks:
        z = np.where(m == 0)[0]
        if len(z):
            seg = np.split(z, np.where(np.diff(z) > 1)[0] + 1)
            for s in seg:
                runs[(int(s[0]), int(s[-1]))] = runs.get((int(s[0]), int(s[-1])), 0) + 1
    print("[corpus] most common empty runs (start,end):count",
          sorted(runs.items(), key=lambda kv: -kv[1])[:12])
    wi = find("window_index.npy")
    if wi:
        w = np.load(wi, allow_pickle=True)
        print(f"[corpus] window_index {w.shape} {w.dtype} | head {w[:3]!r}")
    meta = find("extra_meta.json")
    if meta:
        print("[corpus] extra_meta:", open(meta).read())
