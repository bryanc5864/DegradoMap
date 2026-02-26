# E3 Ligase Generalization: Honest Assessment

## Overview

This document provides an honest assessment of DegradoMap's E3 ligase generalization capabilities, including limitations and appropriate use cases.

## Dataset Composition

The PROTAC-8K dataset has a highly imbalanced E3 ligase distribution:

| E3 Ligase | Samples | Percentage |
|-----------|---------|------------|
| CRBN | 2,011 | 62% |
| VHL | 1,124 | 34% |
| cIAP1 | 62 | 2% |
| MDM2 | 21 | 0.6% |
| Others | 43 | 1.4% |

**Key Limitation:** 96% of the data comes from just two E3 ligases (CRBN and VHL).

## What "E3-Unseen" Actually Tests

Our E3-unseen evaluation holds out one E3 ligase entirely from training:

| Held-out E3 | Test AUROC | n_test | Interpretation |
|-------------|------------|--------|----------------|
| VHL | 0.811 | 1,106 | Train on CRBN, test on VHL |
| CRBN | 0.606 | 1,871 | Train on VHL+others, test on CRBN |
| cIAP1 | 0.271 | 62 | Train on VHL+CRBN, test on cIAP1 |

### Honest Interpretation

1. **VHL holdout (AUROC=0.81):** The model trained primarily on CRBN generalizes well to VHL. This is encouraging but represents a narrow test case.

2. **CRBN holdout (AUROC=0.61):** When VHL is the primary training E3, performance on CRBN is modest - only slightly better than the target-unseen baseline (0.657).

3. **cIAP1 holdout (AUROC=0.27):** Performance is worse than random. This reflects:
   - Very small sample size (n=62)
   - Fundamentally different E3 mechanism
   - Insufficient training signal

## This is NOT True E3 Generalization

A fair description of our evaluation:

- **What we test:** Cross-transfer between CRBN and VHL
- **What we don't test:** Generalization to novel E3 ligases with different mechanisms
- **Why:** The dataset lacks sufficient diversity in E3 ligases

### The 2-Class Problem

Effectively, E3-unseen evaluation is a 2-class problem:
- Can a CRBN-trained model predict VHL degradation? **Yes (0.81 AUROC)**
- Can a VHL-trained model predict CRBN degradation? **Partially (0.61 AUROC)**
- Can the model generalize to truly novel E3s? **Unknown/Unlikely**

## Recommended Framing for Publication

### Do NOT claim:
- "Generalizes to novel E3 ligases"
- "E3-agnostic predictions"
- "Works for any E3 ligase"

### DO claim:
- "Demonstrates cross-E3 transfer between CRBN and VHL"
- "Suggests structural features may transfer across related E3 systems"
- "Performance degrades significantly for underrepresented E3 ligases"

## E3 Ranking Task (More Realistic)

A complementary evaluation: given a target protein and multiple E3 options, can the model rank the correct E3 higher?

| Metric | Value | Interpretation |
|--------|-------|----------------|
| MRR | 0.64 | Correct E3 ranks ~1.5 on average |
| Hit@1 | 46% | Correct E3 is top choice 46% of time |
| Hit@3 | 74% | Correct E3 in top 3 choices 74% of time |

This task is more practical: "Which E3 should I try first?" rather than "Will this E3 work?"

## Improving E3 Generalization

Future work should:

1. **Collect more diverse E3 data:** Beyond CRBN/VHL duopoly
2. **Include E3 structure:** Current model uses E3 name embedding, not E3 3D structure
3. **Pre-train on E3 binding:** Use E3-ligand binding data before PROTAC-specific fine-tuning
4. **Evaluate on emerging E3s:** DCAF15, RNF114, etc.

## Conclusion

DegradoMap shows promising cross-E3 transfer between CRBN and VHL, the two dominant E3 ligases in PROTAC development. However, this should not be interpreted as general E3 generalization. The model's utility for novel E3 ligases remains unvalidated and likely limited given current training data.

**Recommended use case:** Prioritizing CRBN or VHL-based PROTAC candidates for experimental validation.

**Not recommended:** Predictions for underrepresented E3 ligases without additional validation data.
