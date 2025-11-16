# Geometric Logic in NeuraLog

## Overview

NeuraLog integrates **geometric algebra (GA)** as a differentiable, geometric substrate for neurosymbolic reasoning. This provides a principled mathematical framework for representing logical propositions, knowledge graph triplets, and policy rules as geometric transformations in Clifford algebra Cl(3,0).

## The Triplet Decomposition Principle

### Core Idea

Any complex logical expression can be systematically decomposed into a chain of **subject-predicate-object (SPO) triplets**, each represented as a fundamental geometric structure:

```
"A senior citizen visiting in low season is eligible for a discount"
```

Decomposes into:
```
Triplet 1: (Person, hasAge, Age)
Triplet 2: (Age, exceeds, Threshold)
Triplet 3: (Person, isSenior, True)
Triplet 4: (Visit, inSeason, LowSeason)
Triplet 5: (Person, hasDiscount, Eligible)
```

Each triplet becomes a **rotor** (geometric transformation) in Cl(3,0) that can be:
- Composed via geometric product
- Differentiated for gradient-based learning
- Interpreted geometrically
- Used for counterfactual reasoning

### Why Geometric Algebra?

1. **Fixed Dimensionality**: Complex policies stay in 3D space regardless of complexity
2. **Differentiable**: All operations support backpropagation
3. **Geometric Interpretability**: Rotors, bivectors, truth via alignment have clear meaning
4. **Natural for KGs**: (subject, predicate, object) triplets map directly to GA structures
5. **Counterfactual Reasoning**: Rotor trajectories enable "what-if" analysis

## Mathematical Framework

### Clifford Algebra Cl(3,0)

The Euclidean geometric algebra over R³ with basis:
- **Scalars**: `1`
- **Vectors**: `e1, e2, e3` (3D space)
- **Bivectors**: `e12, e13, e23` (oriented planes)
- **Trivector**: `e123` (pseudoscalar, oriented volume)

Total: 2³ = 8 basis elements

A multivector M ∈ Cl(3,0) is represented as an 8D tensor:
```
M = [scalar, e1, e2, e3, e12, e13, e23, e123]
```

### Geometric Product

The fundamental operation combining inner and outer products:
```
uv = u·v + u∧v
```

Where:
- `u·v` = scalar inner product
- `u∧v` = bivector (oriented plane spanned by u and v)

### Rotors

A **rotor** represents rotation in geometric algebra:
```
R(θ) = cos(θ/2) + sin(θ/2)B
```

Where:
- `B` = unit bivector (rotation plane)
- `θ` = rotation angle

Rotors act on vectors via the **sandwich product**:
```
v' = RvR†
```

This rotates vector `v` by angle `θ` in the plane defined by `B`.

## Triplet Representation

### General Triplet Encoding

For triplet τ = (subject, predicate, object):

1. **Embed as vectors**: s, p, o ∈ R³
2. **Define rotation plane**: B = (s∧o) / ||s∧o||
3. **Define rotation angle**: θ = π·σ(predicate_strength)
4. **Construct rotor**: R_τ = cos(θ/2) + sin(θ/2)B

### Canonical Triplet Encoding

For simpler cases, use fixed axes:
- **e1**: Subject axis
- **e2**: Predicate axis
- **e3**: Truth/value axis

Triplet state:
```python
x_τ = x_s·e1 + x_p·e2 + x_t·e3
```

Where:
- `x_s ∈ [0,1]`: subject presence
- `x_p ∈ [0,1]`: predicate strength
- `x_t ∈ R`: truth value or numeric value

### Truth Degree

Truth of a triplet measured by geometric alignment:

**Rotor-based**:
```
truth(τ) = (R_τ s R_τ†)·o / (||R_τ s R_τ†|| ||o||)
```

**Canonical**:
```
truth(τ) = |x_t| / sqrt(x_s² + x_p² + x_t²)
```

Returns value in [0,1] indicating how strongly the relationship holds.

## Logical Operations

### Conjunction (AND)

**Sequential composition** (chain triplets):
```
R_τ1∧τ2 = R_τ2 · R_τ1
```

**Truth degree** (minimum t-norm):
```
truth(τ1 ∧ τ2) = min(truth(τ1), truth(τ2))
```

### Disjunction (OR)

**Probabilistic sum**:
```
truth(τ1 ∨ τ2) = truth(τ1) + truth(τ2) - truth(τ1)·truth(τ2)
```

### Negation (NOT)

**Inverse rotor**:
```
R_¬τ = R_τ†
```

**Truth degree**:
```
truth(¬τ) = 1 - truth(τ)
```

### Implication (τ1 ⇒ τ2)

**Differentiable loss**:
```
L_⇒(τ1, τ2) = max(0, truth(τ1) - truth(τ2))²
```

Penalizes cases where τ1 is true but τ2 is false.

### Equivalence (τ1 ⇔ τ2)

**Differentiable loss**:
```
L_⇔(τ1, τ2) = (truth(τ1) - truth(τ2))²
```

Enforces bidirectional equality.

## Threshold Modes

For numeric predicates (e.g., "age ≥ 65"):

### HARD Mode (Binary)
```python
x_t = 1 if value >= threshold else 0
```
- Clear semantics at boundaries
- Preferred for policy enforcement
- Achieved 100% accuracy in experiments

### SOFT Mode (Sigmoid)
```python
x_t = sigmoid(β·(value - threshold))
```
- Smooth gradients
- Ambiguous at exact thresholds
- Failed on edge cases in experiments

### MARGIN Mode (Safety Margin)
```python
x_t = sigmoid(β·(value - threshold + margin))
```
- Treats threshold as "inclusive with safety"
- Smooth gradients with clear semantics
- Achieved 100% accuracy in experiments

**Recommendation**: Use HARD or MARGIN for regulatory compliance.

## Verification Modes

### Implication Mode (One-Directional)
```
L_impl = max(0, truth_LLM - truth_policy)²
```
Catches **false positives** (LLM claims without support).

### Equivalence Mode (Symmetric)
```
L_equiv = (truth_LLM - truth_policy)²
```
Penalizes both false positives and false negatives equally.

### Bidirectional Mode (Recommended)
```
L_FP = max(0, truth_LLM - truth_policy)²
L_FN = max(0, truth_policy - truth_LLM)²
```
Explicitly tracks both error types. Essential for complete auditing.

## Counterfactual Reasoning

Geometric algebra enables smooth counterfactual analysis via **rotor trajectories**.

### Example: Age Counterfactual

"What if age were 60 instead of 70?"

Trace smooth path:
```python
x_τ(age) = R_age(age) · x_τ(70) · R_age(age)†
```

Where R_age(a) is a rotor that varies the age component.

### Counterfactual Plane

For triplet τ = (s, p, o), the natural counterfactual plane is:
```
B_τ = s ∧ o
```

Modifications to predicate strength or object value = rotations in this plane.

### Robustness Analysis

Quantify counterfactual robustness:
```
L_CF = E_θ∼D[|truth(θ) - truth(0)|]
```

Where θ indexes perturbations, D is perturbation distribution.

## Integration with NeuraLog

### Architecture Integration

```
┌─────────────────────────────────────────────────────┐
│              NeuraLog Knowledge Extraction           │
└────────────────────┬────────────────────────────────┘
                     │
        ┌────────────┴──────────────┐
        │                            │
  ┌─────▼──────┐            ┌───────▼────────┐
  │  Symbolic  │            │   Geometric    │
  │   Layer    │            │   Algebra      │
  │  (Z3 SMT)  │            │  Layer (GA)    │
  └─────┬──────┘            └───────┬────────┘
        │                            │
        │    ┌──────────────────────┐│
        └────►  Neurosymbolic       ││
             │  Reasoner (Hybrid)   ││
             └──────────┬────────────┘
                        │
              ┌─────────▼──────────┐
              │  Verified Output   │
              │  + Explanations    │
              └────────────────────┘
```

### Usage Example

```python
from neuralog.geometric import TripletEncoder, PolicyChain, VerificationMode

# Initialize geometric encoder
encoder = TripletEncoder(mode="canonical", threshold=ThresholdMode.HARD)

# Encode policy triplets
triplets = [
    ("Person", "hasAge", 70),
    ("Age", "exceeds", 65),
    ("Person", "isSenior", True),
]

# Encode as rotors
triplet_states = [encoder.encode(s, p, o) for s, p, o in triplets]

# Compute policy truth degree
policy_chain = PolicyChain(triplet_states)
policy_truth = policy_chain.evaluate()

# Verify LLM answer
llm_claim = "Person is senior"  # LLM output
llm_truth = encoder.encode_claim(llm_claim)

# Bidirectional verification
loss_fp, loss_fn, verdict = verify_llm_answer(
    llm_truth=llm_truth,
    policy_truth=policy_truth,
    mode=VerificationMode.BIDIRECTIONAL
)

if verdict == "VERIFIED":
    print("✅ LLM answer is consistent with policy")
elif verdict == "FALSE_POSITIVE":
    print("❌ LLM claims without support (hallucination)")
elif verdict == "FALSE_NEGATIVE":
    print("⚠️ LLM is overly conservative")
```

### Integration with Existing Components

**Knowledge Graph Distiller** → **GA Triplet Encoder**:
```python
# Extract triplets from text
extraction = kg_distiller.extract_from_text(text)

# Convert to GA representation
ga_triplets = [
    triplet_encoder.encode(t.subject, t.predicate, t.object)
    for t in extraction.triples
]

# Compose into policy chain
policy = PolicyChain(ga_triplets)
```

**Neurosymbolic Reasoner** → **Hybrid Verification**:
```python
# Symbolic verification (Z3)
symbolic_valid, proof = formal_verifier.verify_extraction(extraction)

# Geometric verification (GA)
geometric_truth = policy.evaluate()

# Combine for high-confidence decision
if symbolic_valid and geometric_truth > 0.8:
    confidence = ConfidenceLevel.VERIFIED
```

## Advantages over Purely Symbolic Approaches

| Aspect | Symbolic (Z3/SMT) | Geometric (GA) |
|--------|------------------|----------------|
| **Differentiability** | ❌ Discrete | ✅ Continuous gradients |
| **Dimensionality** | Grows with complexity | Fixed (3D for Cl(3,0)) |
| **Soundness** | >99% formal guarantees | Approximate (trainable) |
| **Interpretability** | Logic formulas | Geometric transformations |
| **Counterfactuals** | Discrete scenarios | Smooth trajectories |
| **Training** | Cannot train | End-to-end learnable |
| **Speed** | Slower (NP-hard) | Fast (linear algebra) |

**Recommended approach**: Hybrid verification using both methods.

## Experimental Validation

Based on the reference paper's park admission policy:

**Policy**: "Seniors (age ≥ 65) visiting in low season (Jan/Feb/Nov/Dec) with budget ≥ $22 are eligible"

**Results** (7 test cases including edge cases):

| Configuration | Policy Accuracy | Verification Accuracy |
|--------------|----------------|----------------------|
| HARD + BIDIRECTIONAL | 100% (7/7) | 100% (7/7) |
| SOFT + BIDIRECTIONAL | 85.7% (6/7) | 85.7% (6/7) |
| MARGIN + BIDIRECTIONAL | 100% (7/7) | 100% (7/7) |

**Key Findings**:
- HARD and MARGIN modes achieve perfect accuracy
- SOFT mode fails at exact thresholds (age=65, budget=$22)
- Bidirectional verification essential for catching both false positives and false negatives
- Canonical encoding sufficient for realistic policies

## Extensions

### Higher-Dimensional GA

**Cl(4,0)**: Add temporal or contextual dimension
```
(subject, predicate, object, time)
```

**Conformal GA Cl(4,1)**: Enable translations, dilations, inversions
- Useful for modal logic
- Geometric negation via inversion

### Knowledge Graph Embeddings

GA provides geometric KG embedding framework:
1. Embed entities as vectors in R³
2. Represent relations as rotors
3. Predict missing links via rotor composition
4. Train with rotor consistency loss

### Multi-Stakeholder Policies

Extend triplets with stakeholder dimension:
```
(subject, predicate, object, stakeholder)
```

Reserve basis vector for stakeholder identity.

### Temporal Policies

Model time-varying policies:
```
R(t) = e^(Bt)
```

Where B is bivector generator of temporal change.

## References

1. **Geometric Algebra for Physicists**. Doran & Lasenby (2003)
2. **Clifford Algebra to Geometric Calculus**. Hestenes & Sobczyk (1984)
3. **RotatE: Knowledge Graph Embedding by Relational Rotation** (ICLR 2019)
4. **Neurosymbolic Verification using Geometric Logic** (geometric view paper)

## Implementation Status

- ✅ Core Cl(3,0) implementation (`neuralog/geometric/clifford.py`)
- 🚧 Triplet logic layer (`neuralog/geometric/triplet_logic.py`)
- 🚧 Policy encoder (`neuralog/geometric/policy_encoder.py`)
- 🚧 Counterfactual reasoner (`neuralog/geometric/counterfactual.py`)
- 🚧 Integration with neurosymbolic reasoner
- 📋 Comprehensive examples and tutorials

## Further Reading

- [ARCHITECTURE.md](ARCHITECTURE.md) - Overall system design
- [SEMANTIC_LAYER.md](SEMANTIC_LAYER.md) - Graph dynamics and episodic memory
- [PRODUCTION_DEPLOYMENT.md](PRODUCTION_DEPLOYMENT.md) - vLLM deployment guide
