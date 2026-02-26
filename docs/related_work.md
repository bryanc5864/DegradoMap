# Related Work and Positioning

## 1. PROTAC Degradability Prediction Methods

### 1.1 DeepPROTACs (2022)

**Input**: Complete PROTAC molecule (warhead + linker + E3 ligand) + target pocket + E3 pocket
**Method**: 3D CNN on binding pocket voxels
**Output**: DC50/Dmax prediction

**Key differences from DegradoMap**:
- DeepPROTACs requires **full PROTAC structure** including synthesized linker
- We use only **target protein + E3 identity** (early-stage feasibility)
- Different use case: they optimize existing PROTACs; we assess novel targets

**Why no direct comparison**: Incompatible inputs. DeepPROTACs needs mol2 pocket files and complete PROTAC SMILES; we need only UniProt ID and E3 name.

### 1.2 PROTAC-BERT (2023)

**Input**: Target protein sequence + PROTAC SMILES
**Method**: Pre-trained protein language model + molecular transformer
**Output**: Binary degradability

**Key differences**:
- Sequence-based (no structural information)
- Requires PROTAC SMILES (we don't)
- Tested on proprietary dataset

### 1.3 PROTACability (2021)

**Input**: Target protein features (sequence-derived)
**Method**: Random forest with hand-crafted features
**Output**: Likelihood of successful PROTAC development

**Comparison**: Similar problem setup. We provide GNN alternative with structural features. Our baseline comparison includes similar feature sets.

## 2. Geometric Deep Learning for Proteins

### 2.1 GVP-GNN (Jing et al., 2021)

Geometric Vector Perceptrons maintain SO(3)-equivariance using:
- Scalar features (invariant)
- Vector features (equivariant)

**Relation to our work**: Our equivariant SUG module draws from GVP design principles. Key adaptation: lysine-aware pooling for ubiquitination.

### 2.2 GearNet (Zhang et al., 2023)

Graph encoder using:
- Relational graph convolution
- Edge message passing
- Multiple edge types (sequential, spatial)

**Relation**: We use similar multi-edge-type graphs. GearNet targets general protein representation; we specialize for degradability.

### 2.3 ESM-IF (Hsu et al., 2022)

Structure-conditioned protein language model combining:
- ESM-2 sequence embeddings
- GNN structural encoder

**Our findings**: ESM embeddings don't help degradability prediction (see ablations). Degradability is NOT captured by evolutionary features.

### 2.4 SchNet (Schütt et al., 2017)

Continuous-filter convolutions with:
- Radial basis function edge features
- Invariant message passing

**Our baseline**: We implement SchNet-style architecture as baseline comparison.

### 2.5 EGNN (Satorras et al., 2021)

E(n)-equivariant graph neural networks:
- Coordinate updates alongside feature updates
- Simple, efficient equivariance

**Our equivariant module**: Builds on EGNN principles with adaptations for lysine-focused protein representation.

## 3. Protein Function Prediction

### 3.1 DeepFRI (Gligorijevic et al., 2021)

Structure-based function prediction using:
- Contact map from structure
- Graph convolution
- Multi-task GO term prediction

**Distinction**: Function prediction vs degradability. DeepFRI predicts intrinsic function; we predict interaction-dependent degradability.

### 3.2 AlphaFold (Jumper et al., 2021)

State-of-the-art structure prediction.

**Our use**: AlphaFold-predicted structures as input features. pLDDT scores as confidence/flexibility proxy.

### 3.3 ProtTrans/ESM-2 (Elnaggar et al., 2022; Lin et al., 2023)

Large protein language models with:
- Self-supervised pre-training
- Rich sequence representations

**Our finding**: ESM-2 embeddings alone achieve ~0.52 AUROC (near random). Evolutionary conservation doesn't predict degradability. Structure + E3 compatibility are the relevant factors.

## 4. Ubiquitin System Modeling

### 4.1 UbiBrowser/DeepUbi

Predict natural ubiquitination sites from sequence.

**Distinction**: We predict PROTAC-induced degradation, which depends on:
- Artificial E3 recruitment (not natural substrates)
- Surface accessibility for synthetic E3-target complexes
- Different kinetics than natural ubiquitination

### 4.2 E3 Ligase Specificity Prediction

Methods predicting natural E3-substrate recognition.

**Key difference**: PROTAC degradation bypasses natural E3 recognition. The PROTAC warhead provides target binding; E3 compatibility matters differently.

## 5. Drug Discovery ML Methods

### 5.1 Molecular Property Prediction

GNNs for small molecule properties (e.g., MPNN, DMPNN, SchNet).

**Relation**: We adapt GNN architectures for protein graphs. Message passing principles transfer; domain-specific features differ.

### 5.2 Protein-Ligand Binding

Methods like EquiBind, DiffDock predict binding poses.

**Distinction**: We don't predict binding (assumed through PROTAC design). We predict downstream degradation given binding occurs.

### 5.3 Target Identification

Methods identifying druggable targets.

**Complementary**: Target identification asks "is this protein druggable?" We ask "is this druggable target degradable?"

## 6. Our Positioning

### 6.1 Problem Niche

DegradoMap fills a specific gap:

| Method | Input | Use Case |
|--------|-------|----------|
| DeepPROTACs | Full PROTAC + pockets | Optimize existing PROTACs |
| PROTAC-BERT | Sequence + PROTAC | General degradability |
| **DegradoMap** | **Structure + E3 name** | **Target feasibility assessment** |

### 6.2 Key Contributions

1. **Structure-first approach**: Use AlphaFold structures, not just sequences
2. **Lysine-aware architecture**: Explicit attention to ubiquitination sites
3. **E3 compatibility modeling**: Cross-attention for protein-E3 matching
4. **Minimal input requirements**: Only target UniProt + E3 identity needed
5. **E(3)-equivariant option**: Theoretically grounded geometric learning

### 6.3 When to Use DegradoMap

**Use DegradoMap when**:
- Assessing target degradability before PROTAC synthesis
- Prioritizing E3 ligases for a target
- Screening protein families for degradable members

**Use other methods when**:
- Optimizing existing PROTAC molecules (use DeepPROTACs)
- Predicting exact degradation kinetics (requires experimental data)
- Predicting natural ubiquitination (use UbiSite predictors)

## 7. Benchmark Datasets

### 7.1 PROTAC-DB

Source of positive examples. Curated database of validated PROTACs.
- Limited to successful PROTACs (selection bias)
- Missing failed attempts

### 7.2 PROTAC-8K (Zenodo)

Our primary dataset:
- 9,384 entries, 3,260 labeled
- 1,222 positive, 2,038 negative
- 8 E3 ligases represented

### 7.3 Comparison Challenges

No standard benchmark exists for structure-based PROTAC prediction. Prior methods use:
- Different input modalities
- Different label definitions
- Proprietary datasets

This limits direct comparison with published results.
