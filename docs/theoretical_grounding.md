# DegradoMap: Theoretical Grounding

## 1. Problem Formulation

### 1.1 Degradability Prediction Task

Given:
- **T**: Target protein structure (from AlphaFold2)
- **E**: E3 ligase identity (categorical: CRBN, VHL, MDM2, cIAP1, DCAF16)
- **L**: Binary label indicating successful degradation

**Goal**: Learn function f(T, E) → [0, 1] predicting P(degradation | T, E)

### 1.2 Key Hypotheses

**H1: Structural Ubiquitination Geometry (SUG)**
> The geometric accessibility of lysine residues for ubiquitination is a key determinant of degradability.

*Biological basis*: PROTAC-mediated degradation requires:
1. Formation of ternary complex (target-PROTAC-E3)
2. Transfer of ubiquitin to accessible lysine(s)
3. Recognition by 26S proteasome

**H2: E3 Ligase Compatibility**
> Different E3 ligases have different substrate recognition preferences.

**H3: Feature Complementarity**
> Combining structural features with ESM-2 embeddings captures complementary information.

## 2. Architecture Justification

### 2.1 Graph Neural Network for Structure
- Proteins naturally represented as graphs
- Message passing captures local structural context
- 4 layers × 8Å radius = ~32Å context per residue

### 2.2 Cross-Attention for E3 Compatibility
- Learns target-E3 interface residues
- Bidirectional attention
- Interpretable

### 2.3 Multi-Scale Gated Fusion
- Learned weighting of modalities
- Prevents single modality dominance

## 3. Feature Engineering

| Feature | Dim | Source | Rationale |
|---------|-----|--------|-----------|
| ESM-2 | 1280 | ESM-2 650M | Evolutionary context |
| pLDDT | 1 | AlphaFold2 | Flexibility proxy |
| SASA | 1 | Computed | Solvent accessibility |
| is_lysine | 1 | Sequence | Ub sites |
| disorder | 1 | Computed | Unstructured regions |
| ub_mask | 1 | PhosphoSitePlus | Known Ub sites |
| E3 one-hot | 6 | Categorical | E3 identity |
| Global stats | 8 | Computed | pLDDT/SASA aggregates |

## 4. Training Strategy

### 4.1 Target-Unseen Split
- Ensures no target in both train and test
- Tests true generalization
- E3-balanced selection

### 4.2 Regularization
- Dropout: 0.05-0.1
- Weight decay: 1e-5
- Early stopping: patience=5

## 5. Theoretical Properties

### 5.1 Expressiveness
k-layer GNN distinguishes graphs differing within k-hop neighborhoods.

### 5.2 Invariances
- Uses edge lengths (invariant to rigid transforms)
- NOT fully E(3)-equivariant (memory constraints)

### 5.3 Generalization Bounds
With n=473 test samples: 95% CI ≈ ±0.045 for AUROC

## 6. Results Summary

| Model | Test AUROC | Improvement |
|-------|------------|-------------|
| GradientBoosting | 0.607 | baseline |
| **DegradoMap** | **0.741** | **+22%** |

## 7. Limitations

1. Single E3 per forward pass
2. No ternary complex modeling  
3. 94% of data is CRBN/VHL
