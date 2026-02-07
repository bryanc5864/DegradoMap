# DegradoMap: Structure-Conditioned, E3-Aware, Context-Dependent Prediction of Protein Degradability

## 1. Vision

DegradoMap predicts whether a target protein can be degraded via Targeted Protein Degradation (TPD/PROTACs) **before any PROTAC is designed**. It produces:

```
DegradoScore(T, E, C) ∈ [0, 1]
```

Given target protein T (structure + sequence), E3 ligase family E, and cellular context C.

---

## 2. Problem Statement

The PROTAC drug discovery pipeline bottleneck: target selection and E3 ligase pairing. Before medicinal chemistry begins, researchers need to know:
1. Is this target intrinsically degradable?
2. Which E3 ligase should be recruited?
3. Does this work in the disease-relevant cell type?

Currently answered through expensive empirical screening. DegradoMap provides a computational oracle.

---

## 3. Three Axes of Novelty

### Axis 1: Structural Ubiquitination Geometry (SUG)
- Per-lysine Ubiquitination Geometry Score (UGS) integrating:
  - SASA of lysine side chain
  - Angular accessibility from E2 active site given E3-imposed geometry
  - Local flexibility (pLDDT from AlphaFold2)
  - Intrinsically disordered region proximity for proteasomal initiation

### Axis 2: Cross-E3 Compatibility Learning
- Cross-attention between target surface and E3 substrate recognition domains
- Trained on PROTAC-DB + UbiBrowser E3-substrate interactions
- Enables: optimal E3 prediction, transfer to new E3s, mechanistic interpretation

### Axis 3: Cellular Context Conditioning
- Cell-line-specific feature vectors from DepMap, ProteomicsDB
- Encodes: E3 expression, proteasome capacity, competing substrates, DUB expression, target half-life

---

## 4. Data Sources

| Source | Content | Scale | Role |
|--------|---------|-------|------|
| PROTAC-DB 3.0 | PROTACs with DC50/Dmax | ~6,111 molecules; ~1,600 labeled | Primary supervision |
| UbiBrowser 2.0 | E3-substrate interactions | 4,068 known + 2.2M predicted | Pre-training E3 compatibility |
| PhosphoSitePlus | Ubiquitination sites | ~70,000 Ub sites | SUG calibration |
| AlphaFold DB | Predicted structures | ~20,000 human proteins | 3D structural input |
| DepMap 24Q4 | Gene expression + dependency | ~1,800 cell lines | Context encoder |
| ProteomicsDB | Protein half-lives | ~14,000 proteins | Context features |
| UbiNet 2.0 | E3-substrate interactions | 3,332 ESIs | Supplementary ESI data |
| DEGRONOPEDIA | Degron motifs | Proteome-wide | Feature engineering |

### Label Construction
- **Positive**: DC50 < 100nM AND Dmax >= 80% (~620 examples)
- **Hard negative**: DC50 > 1µM OR Dmax < 30%
- **Soft negative**: Never targeted proteins (lower weight)

---

## 5. Model Architecture

### Module A: Structural Ubiquitination Geometry (SUG)
- E(3)-equivariant GNN on protein structure
- Per-lysine geometric scoring
- Output: SUG vector + per-residue importance scores

### Module B: E3 Compatibility Network
- E3 substrate recognition domain encoding (same GNN architecture)
- Bidirectional cross-attention: target surface ↔ E3 SRD
- Output: Compatibility vector v_{T,E}

### Module C: Cellular Context Encoder
- DepMap expression features + ProteomicsDB abundance
- 3-layer MLP with residual connections
- Output: Context embedding c_C ∈ R^128

### Module D: Fusion and Prediction Head
- Gated fusion: z = σ(W_g [...]) ⊙ tanh(W_h [...])
- Prediction heads: DegradoScore (binary), DC50 regression, Dmax regression, lysine importance

---

## 6. Training Strategy (3-Phase)

### Phase 1: Pre-training (Weeks 1-4)
- Task 1A: E3-substrate interaction prediction (UbiBrowser)
- Task 1B: Ubiquitination site prediction (PhosphoSitePlus)
- Task 1C: Protein half-life prediction (ProteomicsDB)

### Phase 2: Semi-supervised Fine-tuning (Weeks 5-8)
- Supervised on ~1,600 labeled PROTAC-DB entries
- Memory-based pseudolabeling on ~6,500 unlabeled entries
- Multi-task loss: degradation + DC50 + Dmax + ESI consistency + Ub site

### Phase 3: Proteome-wide Inference (Weeks 9-12)
- All ~20,000 human proteins × 5 E3 ligases × 10 cell lines
- Generate DegradoMap Atlas

### Loss Function
```
L = λ₁·L_degrad(BCE+label_smoothing) + λ₂·L_DC50(Huber) + λ₃·L_Dmax(Huber)
  + λ₄·L_ESI(KL_div) + λ₅·L_UbSite(focal_loss)
```

---

## 7. Evaluation Plan

### Splits
1. **Random split**: 80/10/10 (baseline)
2. **Target-unseen**: Held-out target proteins → test
3. **E3-unseen**: Hold out one E3 ligase family
4. **Cell-line-unseen**: Hold out unseen cell lines

### External Validation
- Bondeson et al. (2018) multi-kinase degrader panel (~200 kinases)
- KRAS mutant panel (ACBI3, 17 mutants)
- Retrospective clinical PROTACs (~20 candidates)

### Metrics
- Accuracy, AUROC, AUPRC, F1
- Lysine prediction: precision@k
- E3 recommendation: mean reciprocal rank

---

## 8. Timeline

| Phase | Weeks | Activities |
|-------|-------|------------|
| Setup | 1-2 | Data acquisition, cleaning, integration |
| SUG | 3-4 | Implement & pre-train SUG module |
| E3 Net | 5-6 | Implement & pre-train E3 compatibility |
| Full Model | 7-8 | Context encoder, end-to-end fine-tuning |
| Eval | 9-10 | Rigorous evaluation, external validation |
| Atlas | 11-12 | Proteome-wide inference, atlas generation |
| Paper | 13-16 | Manuscript, figures, code release |

---

## 9. Computational Requirements

- GPU: 4x A100 80GB
- Storage: ~550 GB total
- Training: ~144 hours total (Phase 1: 48h, Phase 2: 24h, Phase 3: 72h)
- Software: PyTorch Geometric, ESM-2, BioPython, DSSP, IUPred3, RDKit
