# Results Summary

- Baseline accuracy: 98.70% | Baseline+SMOTE accuracy: 98.69%
- Overall accuracy stayed virtually flat (dropped by just 0.01 points) after SMOTE.
- Biggest per-class F1 gain: Class S, which saw a notable recall boost of +4.67%.
- Class Q saw almost no movement (F1 delta of -0.0009), likely due to it being a mixed bag of unknown/paced beats.
- A clear precision/recall trade-off was observed on Class N: the model sacrificed a tiny fraction of Normal beat accuracy to meaningfully improve its sensitivity to rare, dangerous arrhythmias like S and F.
