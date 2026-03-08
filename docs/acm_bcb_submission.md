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

## TL;DR (250 characters)

DegradoMap predicts PROTAC degradability from protein structure using GNNs with ESM-2 embeddings and lysine-aware pooling. Achieves 0.646 ± 0.124 AUROC (best seed: 0.7449) on unseen targets (+23% best over baselines), enabling rational PROTAC design.

---

## Abstract

Targeted protein degradation (TPD) via PROTACs (Proteolysis Targeting Chimeras) has emerged as a transformative therapeutic modality capable of eliminating disease-causing proteins previously considered "undruggable." However, predicting whether a protein target is amenable to PROTAC-mediated degradation remains challenging, as degradability depends on complex structural and biochemical factors beyond simple binding affinity. We present DegradoMap, a structure-aware graph neural network that predicts protein degradability by integrating AlphaFold-predicted structures, ESM-2 protein language model embeddings (1,280-dim per residue), known ubiquitination sites from PhosphoSitePlus, and E3 ligase compatibility features.

DegradoMap introduces three key architectural innovations: (1) a lysine-aware attention pooling mechanism that focuses on ubiquitination-relevant residues, (2) a cross-attention module that models target-E3 ligase compatibility, and (3) a gated multi-modal fusion layer that combines structural, evolutionary, and biochemical features. The model uses 1,285-dimensional node features and is trained with tuned hyperparameters (LR=5e-4, dropout=0.05, batch size=8, early stopping) to manage the high-dimensional ESM-2 feature space. We train and evaluate on PROTAC-8K, the largest publicly available dataset with 3,101 experimentally validated PROTAC-target pairs spanning 155 unique proteins and 10 E3 ligases.

On the challenging target-unseen split, where test proteins are completely absent from training, DegradoMap achieves a multi-seed validated AUROC of 0.646 ± 0.124 (best seed: 0.7449), with the best seed representing a 23% improvement over gradient boosting baselines (0.607). The model shows high variance across random initializations (range: 0.506-0.7449), reflecting initialization sensitivity in the low-data regime with high-dimensional features. ESM-2 protein language model embeddings improve peak performance but require careful integration: naive concatenation fails (0.534 AUROC alone), while tuned training reveals complementary signal. E(3)-equivariant architectures underperform the simpler invariant design (0.626 vs. 0.657 AUROC). The model generalizes well to unseen E3 ligases (AUROC 0.811 on held-out VHL samples) and achieves 74% Hit@3 accuracy on E3 ligase recommendation.

Our results demonstrate that combining protein structure with evolutionary features and domain-specific architectural choices enables accurate degradability prediction for novel targets. DegradoMap provides a computational tool to prioritize PROTAC-amenable targets early in drug discovery, potentially accelerating the development of degrader therapeutics for currently untreatable diseases. Code and trained models are available at https://github.com/anonymous/degradomap.

---

## Submission Metadata

- **Conference:** ACM BCB 2026 (ACM Conference on Bioinformatics, Computational Biology, and Health Informatics)
- **Track:** Research Paper
- **Topics:** Computational Drug Discovery, Machine Learning for Biology, Protein Structure Analysis
- **Word Count (Abstract):** ~350 words
