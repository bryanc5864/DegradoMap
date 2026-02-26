# Theoretical Grounding for DegradoMap

## 1. Problem Formulation

### 1.1 Degradability as a Structural Property

PROTAC-mediated degradation depends on forming a productive ternary complex:
**Target ↔ PROTAC ↔ E3 Ligase**

The key insight is that degradability is fundamentally determined by:
1. **Accessibility of surface lysines** for ubiquitin transfer
2. **Geometric compatibility** between target protein surface and E3 ligase
3. **Structural flexibility** allowing conformational adaptation

### 1.2 Formal Problem Statement

Given:
- Protein structure P = (V, E, X, C) where V is nodes (residues), E is edges (spatial contacts), X is node features, C is 3D coordinates
- E3 ligase identity e ∈ {CRBN, VHL, ...}

Predict: P(degradable | P, e) ∈ [0, 1]

## 2. Graph Neural Network Foundations

### 2.1 Message Passing Neural Networks

Our SUG module implements message passing:

$$h_i^{(l+1)} = \phi\left(h_i^{(l)}, \bigoplus_{j \in \mathcal{N}(i)} \psi(h_i^{(l)}, h_j^{(l)}, e_{ij})\right)$$

where:
- $h_i^{(l)}$ is node $i$'s representation at layer $l$
- $\mathcal{N}(i)$ is the spatial neighborhood within radius $r$
- $\psi$ is the message function
- $\bigoplus$ is a permutation-invariant aggregation
- $\phi$ is the update function

### 2.2 Spatial Inductive Bias

The radius graph construction encodes a strong inductive bias:

$$\mathcal{N}(i) = \{j : \|c_i - c_j\|_2 < r \text{ and } i \neq j\}$$

This captures:
- Local tertiary structure contacts
- Surface accessibility patterns
- Potential protein-protein interaction interfaces

**Theoretical justification**: Ubiquitination requires E3-target proximity. Surface lysines within ~10Å of the binding interface are most likely ubiquitination sites.

### 2.3 Lysine-Weighted Pooling

We weight the graph readout by lysine positions:

$$h_{\text{protein}} = \frac{1}{\sqrt{N}} \sum_{i=1}^{N} (1 + \alpha \cdot \mathbb{1}[\text{Lys}_i]) \cdot h_i^{(L)}$$

**Justification**: Lysines are the exclusive sites of ubiquitin attachment. Weighting by lysine presence creates an attention-like mechanism that focuses on relevant residues.

The $1/\sqrt{N}$ normalization ensures consistent scale across proteins of different sizes (important for comparing across the dataset).

## 3. E(3)-Equivariance Theory

### 3.1 Why Equivariance Matters

Protein function is rotation/translation invariant. A rotated protein should have identical degradability predictions:

$$f(R \cdot P + t) = f(P) \quad \forall R \in SO(3), t \in \mathbb{R}^3$$

Standard GNNs with coordinate features are NOT equivariant by default.

### 3.2 Equivariant Message Passing

The equivariant SUG module implements:

**Scalar messages** (invariant):
$$m_{ij}^{(l)} = \phi_m(h_i^{(l)}, h_j^{(l)}, \|c_i - c_j\|_2)$$

**Vector messages** (equivariant):
$$\vec{m}_{ij}^{(l)} = \phi_v(h_i^{(l)}, h_j^{(l)}) \cdot \frac{c_j - c_i}{\|c_j - c_i\|_2}$$

**Update**:
$$h_i^{(l+1)} = \phi_h\left(h_i^{(l)}, \sum_j m_{ij}^{(l)}\right)$$
$$\vec{v}_i^{(l+1)} = \phi_u(\vec{v}_i^{(l)}) + \sum_j \vec{m}_{ij}^{(l)}$$

The scalar channel remains invariant; the vector channel transforms correctly under rotations.

### 3.3 Spherical Harmonics Encoding

For richer angular information, we use spherical harmonics:

$$Y_l^m(\theta, \phi) : \mathbb{S}^2 \to \mathbb{C}$$

Decomposing relative positions into spherical harmonics provides:
- Rotation-equivariant features
- Multi-resolution angular information
- Efficient parameterization of directional interactions

## 4. Cross-Attention Mechanism

### 4.1 Protein-E3 Compatibility

The cross-attention between protein representation and E3 embedding captures:

$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

where:
- Q = protein lysine-weighted representation
- K, V = E3 ligase embedding

**Interpretation**: The attention scores indicate how compatible each E3 ligase's binding mode is with the protein's surface structure.

### 4.2 Why Cross-Attention Over Concatenation?

Concatenation: $z = [h_{\text{protein}}; h_{\text{E3}}]$
- Treats protein and E3 as independent
- No interaction modeling

Cross-attention: $z = \text{CrossAttn}(h_{\text{protein}}, h_{\text{E3}})$
- Models compatibility explicitly
- Allows E3-specific weighting of protein features
- Better generalization to unseen E3s

## 5. Generalization Bounds

### 5.1 Target-Unseen Generalization

The target-unseen split tests generalization to new proteins. This is challenging because:
- Training set may not cover all protein folds
- Surface patterns vary across protein families

**Theoretical framework**: We rely on the structural hypothesis that degradability correlates with:
1. Surface lysine accessibility (transferable across proteins)
2. Local structural motifs (learnable from diverse training set)

### 5.2 E3-Unseen Generalization

Harder generalization: predicting for novel E3 ligases.

**Our approach**: Learn E3-independent protein features that capture "degradability potential":
- Surface accessibility patterns
- Lysine clustering
- Structural flexibility

The E3 module then modulates this potential based on compatibility.

## 6. Connection to Biophysics

### 6.1 Ubiquitin Transfer Mechanism

Degradability requires:
1. **E3-target binding**: Geometric complementarity
2. **E2~Ub positioning**: Lysine within ~10Å of E2 active site
3. **Polyubiquitin chain formation**: K48-linked chains for proteasomal degradation

Our model captures (1) and (2) through:
- Spatial graph structure
- Lysine-focused attention
- E3 compatibility modeling

### 6.2 AlphaFold Confidence as Flexibility Proxy

pLDDT scores from AlphaFold correlate with:
- Structural order (high pLDDT = ordered)
- Dynamics (low pLDDT = flexible/disordered)

Flexible regions may:
- Accommodate E3 binding conformational changes
- Present lysines in accessible conformations

Including pLDDT as a node feature captures this structural dynamics information.

## 7. Limitations and Assumptions

### 7.1 Modeling Assumptions

1. **Static structure sufficient**: We use single AlphaFold structures, ignoring conformational ensembles
2. **Residue-level representation**: Coarse-grained; ignores side-chain conformations
3. **Binary degradability**: Real degradation efficiency is continuous

### 7.2 Data Limitations

1. **PROTAC dependency**: Degradability depends on the specific PROTAC (linker, warhead), which we don't model
2. **E3 simplification**: E3 identity as categorical variable ignores E3 structure
3. **Cell-type effects**: Experimental degradation varies by cell line

### 7.3 What We Cannot Learn

The model CANNOT learn:
- PROTAC binding affinity (no PROTAC structure input)
- Cellular permeability
- Metabolic stability
- Exact ubiquitination sites (we predict overall degradability)

## 8. References

1. Dwane et al. (2021) "Approaches for degradation design..."
2. Bronstein et al. (2021) "Geometric Deep Learning" - theoretical foundations
3. Satorras et al. (2021) "E(n) Equivariant GNNs" - equivariant architecture
4. Schütt et al. (2017) "SchNet" - continuous-filter convolutions
5. Jumper et al. (2021) "AlphaFold" - structure prediction confidence
