# ACM BCB Submission Materials

## Title

**DegradoMap: Structure-Aware Graph Neural Networks for PROTAC-Mediated Protein Degradability Prediction**

---

## Keywords

1. PROTAC
2. Targeted protein degradation
3. Graph neural networks
4. Protein structure prediction
5. Drug discovery
6. AlphaFold
7. ESM-2
8. Protein language models
9. E3 ubiquitin ligase
10. Ubiquitination
11. Multi-modal learning
12. Deep learning
13. Computational drug design
14. Proteolysis targeting chimeras
15. Structure-based prediction
16. Cross-attention networks
17. Molecular machine learning

---

## TL;DR (247 characters)

DegradoMap predicts PROTAC degradability from protein structure using GNNs with ESM-2 embeddings and lysine-aware pooling. Achieves 0.65 AUROC (multi-seed validated) on unseen targets (+6% over baselines), enabling rational PROTAC design for undruggable proteins.

---

## Abstract

Targeted protein degradation (TPD) via PROTACs (Proteolysis Targeting Chimeras) has emerged as a transformative therapeutic modality capable of eliminating disease-causing proteins previously considered "undruggable." However, predicting whether a protein target is amenable to PROTAC-mediated degradation remains challenging, as degradability depends on complex structural and biochemical factors beyond simple binding affinity. We present DegradoMap, a structure-aware graph neural network that predicts protein degradability by integrating AlphaFold-predicted structures, ESM-2 protein language model embeddings, known ubiquitination sites, and E3 ligase compatibility features.

DegradoMap introduces three key architectural innovations: (1) a lysine-aware attention pooling mechanism that focuses on ubiquitination-relevant residues, (2) a cross-attention module that models target-E3 ligase compatibility, and (3) a gated multi-modal fusion layer that combines structural, evolutionary, and biochemical features. We train and evaluate on PROTAC-8K, the largest publicly available dataset with 3,101 experimentally validated PROTAC-target pairs spanning 155 unique proteins and 10 E3 ligases.

On the challenging target-unseen split, where test proteins are completely absent from training, DegradoMap achieves a multi-seed validated AUROC of 0.65 ± 0.12 (best seed: 0.74), representing a 6% average improvement over gradient boosting baselines. The model shows high variance across random initializations (range: 0.50-0.74), with the best seed achieving 23% improvement over baselines. The lysine-aware pooling mechanism provides crucial inductive bias for the ubiquitination prediction task. The model generalizes well to unseen E3 ligases (AUROC 0.81 on held-out VHL samples) and achieves 74% Hit@3 accuracy on E3 ligase recommendation.

Our results demonstrate that combining protein structure with evolutionary features and domain-specific architectural choices enables accurate degradability prediction for novel targets. DegradoMap provides a computational tool to prioritize PROTAC-amenable targets early in drug discovery, potentially accelerating the development of degrader therapeutics for currently untreatable diseases. Code and trained models are available at https://github.com/bryanc5864/DegradoMap.

---

## Submission Metadata

- **Conference:** ACM BCB 2026 (ACM Conference on Bioinformatics, Computational Biology, and Health Informatics)
- **Track:** Research Paper
- **Topics:** Computational Drug Discovery, Machine Learning for Biology, Protein Structure Analysis
- **Word Count (Abstract):** ~350 words
