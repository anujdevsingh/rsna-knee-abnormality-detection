# Decisions

Each choice, why, what was tried instead, measured gain.

## 2026-10-14: Strategy
- Build an own pipeline instead of copying public notebooks; no OAI or other external training data.
- Validate on weak (LLM) labels plus the 58 gold studies. Do not tune on the public LB.

## 2026-10-14: Step 1, label merge (kaggle/labels_merge)
- Members: pilkwang v2, steven v4_blend (best of the 3 steven files on gold58; one file per author
  so no labeller counts twice), lixin gpt56sol.
- Dropped barun2104: its `pseudo_*` columns are 0.5 everywhere except the 58 gold rows, so it has no
  LLM labels. Its 5 folds cover only the gold rows.
- Merge = rank average per label, mapped back onto the members' value distribution (keeps each
  label's prevalence for BCE). Gold58: rank 0.8933, plain mean 0.8948, best single set 0.8927. The
  differences are inside gold58 noise; rank kept because the sets are calibrated differently.
- Silent mask = pilkwang verdict UNK (27.3% of cells, Synovitis 84%, matches the forum figure).
- Gold rows keep the expert labels, fold -1, never trained on. Folds: 5-fold (`fold`) and 3-fold
  (`fold3`), greedy stratified on merged labels >= 0.5.

## 2026-10-14: Step 2, teacher design (kaggle/teacher_cnxt)
- Read the Raptor training and inference code as reference only. Facts used: corpus is
  one (44,336,336) uint8 stack per study, part 1 then part 2 in id order, mask (N,44); stack =
  fixed plane slots filled from chosen series, 140 mm crop, 2-98% intensity window. The public
  inference notebook uses a newer 64-slice layout, so the 44-slice slot sizes are NOT known yet.
  `SLOTS` in teacher_cnxt.py is a guess; the labels-merge kernel prints the corpus mask layout to fix it.
- ConvNeXt-T (timm convnext_tiny.fb_in22k_ft_in1k, ImageNet weights, Apache-2.0), 224 px, 3-slice
  windows kept inside one slot, per-label attention pool, K=12 train / 24 eval windows.
- Soft BCE; silent cells weight 0.3. Mirror aug p=0.5 and mirror TTA for OOF: sagittal windows
  reverse slice order, coronal/axial flip width (attention pool is order-invariant, so window
  positions need no remap).
- Best epoch by held-out-fold weak-label AUC (mentioned cells, label >= 0.5); gold58 logged too.
- One fold per T4 in parallel subprocesses (forum: DataParallel is slow). 8 epochs.

## 2026-10-14: Teacher T1 result
- 3-fold ConvNeXt-T 224: OOF weak AUC 0.8712, gold58 ensemble 0.8856 (label merge on gold58: 0.8933,
  so the image model is already near the report-label ceiling on gold58). Runtime ~65 min on T4 x2.
- Mirror TTA: weak AUC +0.0014 / +0.0003 on folds 0/1 (gold58 +0.007 / -0.003, noise). Keep.
- Weak-AUC plateau at epochs 5-7 with train loss still falling: more epochs will not help at 224 px.

## 2026-10-14: Step 3 pseudo labels (kaggle/pseudo_labels)
- Rule from RESEARCH.md §6: silent -> teacher OOF (weight 0.5), mentioned -> 50/50 label/OOF (weight 1),
  mentioned with |label - OOF| >= 0.8 -> weight 0 (0.74% of cells).
- Gold58 check of the rule applied to gold rows (teacher fold-avg preds): 0.9213 vs report labels 0.8933
  vs teacher 0.8856. Kept.

## 2026-10-14: Step 4 student design (kaggle/student_cnxt)
- Same code as the teacher; 288 px, 5 folds (`fold`), targets/weights from pseudo_labels.parquet.
  Evaluation stays on the original report labels (mentioned cells) so it is comparable with T1.
- First-step OOM falls back to gradient checkpointing instead of failing the run.

## 2026-10-14: Preprocessing check before inference (kaggle/prep_check)
- The 44-slice corpus builder is not public. The kernel rebuilds stacks from train DICOMs under
  candidate settings (slice span, mm crop) and compares with the corpus (Pearson r per slot), and lists
  the available DICOM decoders. The inference notebook uses the best match.

## 2026-10-14: Corpus preprocessing reproduced exactly (prep_check v1)
- 16 random corpus studies rebuilt from train DICOMs. Slice span 15-85% of each series + 140 mm
  centre crop + 2-98% intensity window over the picked slices + INTER_AREA to 336 gives r 1.0000,
  MAD 0.00 vs the corpus. Span 6-94% (the public 64-slice notebook) gives r 0.76; 10-90% r 0.85.
- Series pick: first series in *_series.csv order matching (plane, Fluid_Sensitive) per slot
  (sag fluid, sag non-fluid, cor fluid, cor non-fluid, axial any).
- Decoders: pydicom 3.0.2 only (no gdcm / pylibjpeg) decoded all 16 studies. Inference logs any
  per-study failure.
- infer.py PREP set to lo=0.15, hi=0.85, crop=140.

## 2026-10-14: First submission (teacher T1) = public 0.901
- Pipeline works end to end on the hidden test set (~15 min). Rank ~4012 of 5958; bronze line 0.956.
- CV -> LB for this model: weak OOF 0.871, gold58 0.886 -> public 0.901. The gap to bronze is ~0.055,
  far more than the student alone is likely to close. Main suspect: the 44-slice corpus (forum: slice
  density is the second lever after labels; strong single models use ~96 slices at 320-384 px).

## 2026-10-14: Dense corpus (kaggle/corpus_dense, 3 CPU shard kernels)
- Decided to build our own denser corpus after the 0.901 first score.
- 96 slices (slots 27/21/18/12/18, same series picks), 288 px, span 6-94% (wider, to reach the
  collateral ligaments and lateral meniscus; MCL and Lat Meniscus are our weakest labels), 140 mm crop,
  2-98% window. Reader is the one that reproduced the Raptor corpus exactly; each shard re-checks
  3 studies against Raptor with the 44-slice settings.
- 3 shards (~1,470 studies, ~11.7 GB each) to stay under the Kaggle output limit; CPU only.

## 2026-10-14: Inference handles several corpora
- infer.py: each member set names its corpus build (raptor44 or dense96); every test study is built
  once per needed corpus; probabilities averaged over folds within a set, rank-averaged across sets.
  Tested locally with stubbed DICOM builds and smoke checkpoints (student 44 + dense 96).
- Dense trainer saves dense_fold*.pt / oof_dense.csv so it can be attached next to the student.
- Dense student uses a new 4-fold split (fold4, 1,087-1,088 per fold) = 2 rounds on T4 x2 instead of 3.

## 2026-10-14: Student S2 = public 0.902 (teacher 0.901)
- Weak OOF +0.02 and gold58 +0.011 over the teacher gave only +0.001 public (SD ~0.003): no real gain.
  Weak-label CV is inflated by the pseudo labels (teacher saw the val studies); gold58 is too small.
- Checked: test_series.csv has the same format and plane names as train; the test-time build is the
  reader that reproduced the corpus exactly. Preprocessing mismatch is unlikely.
- Next evidence: the dense 96-slice student (D1). If it also lands ~0.90, the cap is elsewhere
  (labels / model), not slice density.

## 2026-10-14: Dense 96-slice corpus = public 0.909 (44-slice student 0.902)
- Same model, labels and resolution; only the input changed (96 slices, span 6-94%, 288 px stored).
  Gold58 0.8965 -> 0.9028, public 0.902 -> 0.909. Slice density is a real lever; keep the dense corpus.
- Raptor's public single model (CoAtNet-2 at 384 px, 64 slices) claims 0.924, so backbone and
  resolution are the next levers.

## 2026-10-14: Dense + 44-slice blend not submitted
- Offline rank blend (OOF weak / gold58): dense alone 0.8959 / 0.9028; 50/50 0.8964 / 0.9004;
  70/30 0.8968 / 0.9024. Change is inside noise, so the 44-slice student adds nothing; kept the
  submission slot. infer.py stays on the dense model.

## 2026-10-14: Next test = ConvNeXt-Small at 384 px
- Corpus rebuilt at 384 px (same 96-slice recipe), 6 CPU shards with a `v384_` file prefix
  (Kaggle allows 5 CPU sessions at once, so shard 5 starts when one finishes).
- Trainer kaggle/student_s384: ConvNeXt-Small (fb_in22k_ft_in1k), 384 px, K 12/24, 6 epochs, folds 0-1 of
  fold4 only, same pseudo labels. Compared fold by fold with dense D1 (ConvNeXt-T 288). About 2-2.5 h
  of the ~4.5 h GPU left this week; the two fold models can join the final ensemble.
- infer.py knows the dense384 build and the s384 member set.

## 2026-10-14: ConvNeXt-Small 384 test (X1) did not beat ConvNeXt-T 288
- Same folds 0-1: weak 0.891 vs 0.895, gold58 2-fold ensemble 0.899 vs 0.903 (dense D1). A 50/50 rank
  blend adds nothing (0.8949 / 0.9025). Not submitted.
- Confounded: X1 had 6 epochs and K=12 windows vs 8 and 16, and was still improving at its last epoch.
  At equal GPU time, the bigger/higher-res model is not better here. Keep ConvNeXt-T 288 dense (0.909).
- GPU used this week: ~8.4 h of ~10.7 h.

## 2026-10-15: Pseudo labels v2 from the dense model
- Same rule (silent -> OOF w0.5, mentioned 50/50, drop |L-P| >= 0.8) with dense D1 OOF instead of the
  teacher. Gold58 check of the rule: 0.9292 (v2) vs 0.9213 (v1); blend 0.5 best of 0.3/0.5/0.7;
  averaging dense+student OOF is slightly worse (0.9273). Kernel rsna-knee-pseudo-labels-v2.

## 2026-10-15: JEV labels (riadmohamed42/jev-knee-labels, CC0) not used
- README: score = 0.95 x per-target pick of steven v4 / pilkwang v2 (pick made on gold58) + 0.05 x a JEV
  ensemble weighted on 67 gold studies. Almost no new signal for us, and any gold58 score is inflated
  by selection on gold. Skipped. Choosing per-target winners on gold58 ourselves would overfit the same way.

## 2026-10-15: PatientSex stacker not used
- Headers: sex M 2076 / F 1894 / blank 238 / O 199; PatientAge is blank for every study.
- Per-label logistic stack on [logit(dense OOF), male, female], 5-fold CV: weak 0.8954 vs 0.8953 without
  sex; gold58 0.9011 vs 0.9028. No gain; the image model already carries the sex signal.

## 2026-10-15: Next week's second backbone prepared (kaggle/student_effv2s)
- tf_efficientnetv2_s.in21k_ft_in1k (timm, ImageNet-21k weights) on the dense 288 corpus, pseudo labels v2,
  4 folds, same head and schedule as dense v2. Smoke test passed (4 folds, CPU, fake data). Not run yet.

## 2026-10-15: Pseudo labels v2 help the image model (D2, folds 0-1)
- Same setup as D1, only the labels changed: weak +0.004 / +0.002, gold58 +0.003 / +0.003 per fold;
  2-fold gold ensemble 0.9072 vs 0.9032. Next week's full models train on v2 labels.
- Submission v5 = rank blend of D2 (folds 0-1) and D1 (4 folds): gold58 0.9055 vs D1 alone 0.9028.
- v5 public 0.909, the same as D1 alone: the +0.003 gold58 gain of the blend is inside LB noise.
  Rank moved ~3925 -> ~3969 only because other teams improved.

## 2026-10-19: Final-week GPU runs started
- D3: dense ConvNeXt-T 288, pseudo labels v2, all 4 folds, 12 epochs (D1 still improved at epoch 8).
- E1: EfficientNetV2-S 288, same data/labels, 4 folds, 10 epochs: second backbone for the ensemble.
- Both run as parallel T4 x2 sessions; outputs densev2f_fold*.pt and effv2s_fold*.pt (infer.py knows both).

## 2026-10-19: D3 (dense, v2 labels, 4 folds, 12 epochs) is the best model
- OOF weak 0.9035 (D1 0.8959), gold58 0.9072 (D1 0.9028). EfficientNetV2-S E1: 0.8965 / 0.9006.
- Rank blends do not beat D3 alone offline (D3+E1 0.9027 / 0.9053). Submitted D3 alone (v6) and
  D3+E1 (v7) to see if the backbone mix helps on the LB anyway.
- Result: v6 (D3) 0.914 and v7 (D3+E1) 0.914 public. Pseudo labels v2 + 12 epochs gave +0.005; EfficientNetV2-S adds nothing on the LB.

## 2026-10-19: Pseudo labels v3 not built
- Gold58 check of the rule with D3 OOF: 0.9258 (best blend 0.3: 0.9266) vs v2 0.9292. No gain, so no further label round; no GPU spent.

