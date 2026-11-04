#!/usr/bin/env python3
"""Metadata stacker check (CPU): does PatientSex (and age, when present) from the DICOM headers add to the dense model?

Reads one header per train study (first series), then per label fits a logistic regression on
[logit(dense OOF), sex, age] with 5-fold CV over the training studies (target: merged report label
>= 0.5 on mentioned cells). Compares macro AUC with and without metadata on the weak labels (CV) and on
gold58 (model fitted on all training studies, applied to the gold fold-average predictions).
Only aggregate numbers are printed; the per-study metadata stays in /kaggle/working.
"""
import os, glob, json
import numpy as np, pandas as pd
import pydicom
from concurrent.futures import ThreadPoolExecutor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

LAB = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA", "PF OA",
       "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SKIP = ("train_series", "test_series", "train_images", "test_images")


def find(name, root="/kaggle/input"):
    for d, sub, fs in os.walk(root):
        sub[:] = sorted(x for x in sub if x not in SKIP)
        if name in fs:
            return os.path.join(d, name)
    return None


comp = os.path.dirname(find("train_series.csv"))
ser = pd.read_csv(f"{comp}/train_series.csv")
first = ser.groupby("StudyInstanceUID").first().reset_index()


def header(row):
    fs = sorted(glob.glob(f"{comp}/train_series/{row.StudyInstanceUID}/{row.SeriesInstanceUID}/*.dcm"))
    if not fs:
        return row.StudyInstanceUID, None, None
    h = pydicom.dcmread(fs[0], stop_before_pixels=True)
    sex = str(getattr(h, "PatientSex", "") or "")
    age = str(getattr(h, "PatientAge", "") or "")
    age = float(age[:-1]) if age[:-1].isdigit() and age.endswith("Y") else np.nan
    return row.StudyInstanceUID, sex, age


with ThreadPoolExecutor(8) as ex:
    meta = pd.DataFrame(list(ex.map(header, first.itertuples())), columns=["StudyInstanceUID", "sex", "age"])
meta.to_csv("/kaggle/working/meta.csv", index=False)
print("sex counts:", meta.sex.value_counts(dropna=False).to_dict())
print(f"age known: {meta.age.notna().mean():.3f} | median {meta.age.median()}")

lab = pd.read_parquet(find("merged_labels.parquet")).set_index("StudyInstanceUID")
oof = pd.read_csv(find("oof_dense.csv")).set_index("StudyInstanceUID")[LAB]
gold = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.dirname(find("oof_dense.csv")) + "/gold_fold*.csv"))])
gold = gold.groupby("StudyInstanceUID")[LAB].mean()
m = meta.set_index("StudyInstanceUID")
male = (m.sex == "M").astype(float); fem = (m.sex == "F").astype(float)
USE_AGE = m.age.notna().mean() > 0.5   # PatientAge is blank in this data (2026-10-15 run)
age = (m.age.fillna(m.age.median()) / 100.0) if USE_AGE else None
logit = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))


def feats(P, idx, with_meta):
    X = [logit(P.loc[idx].values)]
    if with_meta:
        X += [male.loc[idx].values[:, None], fem.loc[idx].values[:, None]]
        if USE_AGE:
            X.append(age.loc[idx].values[:, None])
    return np.hstack(X)


res = {}
gidx = gold.index
for l in LAB:
    sl = lab.loc[oof.index, f"silent_{l}"] == 0
    idx = oof.index[sl.values]; y = (lab.loc[idx, l] >= 0.5).astype(int).values
    P = oof[[l]]
    row = {}
    for wm in (False, True):
        pr = np.zeros(len(idx))
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(idx, y):
            c = LogisticRegression(C=1.0, max_iter=1000).fit(feats(P, idx[tr], wm), y[tr])
            pr[va] = c.predict_proba(feats(P, idx[va], wm))[:, 1]
        c = LogisticRegression(C=1.0, max_iter=1000).fit(feats(P, idx, wm), y)
        pg = c.predict_proba(feats(gold[[l]], gidx, wm))[:, 1]
        row["meta" if wm else "base"] = (roc_auc_score(y, pr), roc_auc_score(lab.loc[gidx, l], pg))
    row["raw"] = (roc_auc_score(y, P.loc[idx, l]), roc_auc_score(lab.loc[gidx, l], gold[l]))
    res[l] = row
    print(f"{l:18s} weak raw {row['raw'][0]:.4f} base {row['base'][0]:.4f} meta {row['meta'][0]:.4f} | "
          f"gold raw {row['raw'][1]:.4f} base {row['base'][1]:.4f} meta {row['meta'][1]:.4f}")
for k in ("raw", "base", "meta"):
    print(f"MACRO {k:4s} weak {np.mean([res[l][k][0] for l in LAB]):.4f} | gold58 {np.mean([res[l][k][1] for l in LAB]):.4f}")
json.dump({l: {k: list(v) for k, v in r.items()} for l, r in res.items()}, open("/kaggle/working/meta_stack.json", "w"), indent=1)
