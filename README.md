# RSNA Knee Abnormality Detection (Kaggle 2026)

My solution to the [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
competition. The task was to predict 12 knee MRI findings per study (ACL, menisci, MCL, osteoarthritis in three
compartments, effusion, synovitis, Baker's cyst, bone contusion, fracture). Scoring was macro AUC.

**Final result: private LB 0.900, rank 4,045 of about 6,300 teams** (public LB 0.914).
I used only competition data and ImageNet-pretrained backbones, with no OAI or other external training data.

No competition images, labels or derived data are stored in this repo.

## The challenge

- 4,407 training studies. Each has several DICOM series (sagittal, coronal, axial; fluid-sensitive or not)
  and a free-text radiology report in one of several languages.
- Only **58 studies have expert labels**. All other labels have to come from the reports, but the test
  studies (about 1,300) have no reports. So the main problem is getting good labels.

## My pipeline

Each folder in [`kaggle/`](kaggle) is one Kaggle kernel. All training and inference ran on Kaggle (T4 x2).

1. **Label merge** (`labels_merge`): a rank average of three public CC0 report-label sets
   (pilkwang, stevenleehans, lixin73). Cells the reports do not mention (27%; 84% for synovitis) are
   masked as "silent". Macro AUC on the 58 gold studies: 0.893.
2. **Teacher** (`teacher_cnxt`): ConvNeXt-T, 2.5D (3-slice windows), with a per-label attention pool, at 224 px.
   It trains on soft BCE with down-weighted silent cells, plus anatomical left/right mirror augmentation and TTA.
3. **Pseudo labels** (`pseudo_labels`, `pseudo_labels_v2`): silent cells take the out-of-fold prediction.
   Mentioned cells are a 50/50 mix of report label and prediction, and large disagreements are dropped.
   Gold58 AUC of the labels went from 0.893 to 0.921 (v1) to 0.929 (v2).
4. **Own dense corpus** (`corpus_dense*`): 96 slices per study at 288 px across 5 plane/contrast slots,
   built from the DICOMs. A check against a public corpus (`prep_check`) confirmed the preprocessing matches exactly.
5. **Students** (`student_*`): the same model trained on the pseudo labels and the dense corpus.
6. **Inference** (`infer`): builds the stacks from the test DICOMs, runs mirror TTA, averages over folds,
   and rank-averages across model sets.

## Results

| Run | Data / labels | gold58 AUC | Public | Private |
|---|---|---|---|---|
| Teacher, ConvNeXt-T 224, 3 folds | 44-slice / merged | 0.886 | 0.901 | 0.887 |
| Student, ConvNeXt-T 288, 5 folds | 44-slice / pseudo v1 | 0.897 | 0.902 | 0.890 |
| Dense student D1, 4 folds | dense 96 / pseudo v1 | 0.903 | 0.909 | 0.894 |
| **Dense student D3, 4 folds, 12 epochs** | dense 96 / pseudo v2 | **0.907** | **0.914** | **0.900** |
| D3 + EfficientNetV2-S rank blend | dense 96 / pseudo v2 | 0.905 | 0.914 | 0.899 |

The full experiment log is in [`experiments/log.csv`](experiments/log.csv). Every design choice, with
the numbers behind it, is in [`docs/decisions.md`](docs/decisions.md).

**What helped:** pseudo labels, the denser 96-slice corpus (+0.007 LB), longer training on better labels
(+0.005 LB), and mirror TTA.
**What did not help:** ConvNeXt-S at 384 px (within my GPU budget), EfficientNetV2-S as a blend member,
a PatientSex stacker, and a third pseudo-label round.

## What the winners did differently

The top private scores were about 0.957 to 0.960. From the published write-ups (1st, 5th and 7th place):

- **They treated it as a labelling competition.** Frontier LLMs read each training report *together with
  the MRI slices*, which raised label quality on gold58 to about 0.94 to 0.95. My report-only labels reached 0.893
  (0.929 after pseudo labelling). That gap explains most of the difference in score.
- **They calibrated labels on the 58 gold studies.** For example, hedged or low-grade report findings were
  pushed down to match the competition's definitions.
- **They used anatomy-aware crops.** ROI boxes (1st place) or bone segmentation (5th place) focused the model
  on the menisci, ligaments and compartments at higher resolution (384 px+).
- Simple 2D or 2.5D ConvNeXt or EfficientNet backbones were enough once the labels and crops were good.

My lesson: with weak supervision, put the effort into label quality first and model architecture second.

## Reproducing

Kernel metadata (`kernel-metadata.json`) is included for each step. Run them in the order above with the
Kaggle CLI (`kaggle kernels push -p kaggle/<step>`). They need the competition data and the three public label datasets.

## Author

Anuj Dev Singh ([Kaggle](https://www.kaggle.com/anujdevsingh))

Licence: MIT (see [LICENSE](LICENSE)).
