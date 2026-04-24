# Model Mathematics and Fitting Reference

Complete documentation for the RLWM computational models, mathematical formulations, and Bayesian fitting procedures.

---

## 1. Model Overview

Seven models are implemented for fitting human behavioral data, organized by complexity:

| Model | Description | Free Parameters |
|-------|-------------|-----------------|
| **M1: Q-Learning** | Model-free RL baseline with asymmetric learning rates | 3: α₊, α₋, ε |
| **M2: WM-RL Hybrid** | Combines working memory and RL systems | 6: α₊, α₋, φ, ρ, K, ε |
| **M3: WM-RL + Perseveration** | M2 + global choice persistence kernel | 7: α₊, α₋, φ, ρ, K, κ, ε |
| **M5: WM-RL + RL Forgetting** | M3 + Q-value decay toward baseline | 8: α₊, α₋, φ, ρ, K, κ, φ_rl, ε |
| **M6a: WM-RL + Stim-Specific** | M2 + stimulus-specific perseveration | 7: α₊, α₋, φ, ρ, K, κ_s, ε |
| **M6b: WM-RL + Dual Perseveration** | M2 + global + stimulus-specific kernels (stick-breaking) | 8: α₊, α₋, φ, ρ, K, κ_total, κ_share, ε |
| **M4: RLWM-LBA** | M3 learning + Linear Ballistic Accumulator for joint choice+RT | 10: α₊, α₋, φ, ρ, K, κ, v_scale, A, δ, t₀ |

> **Note on model numbering:** The numbering gap (M3 to M5) is intentional. M4 is listed last because it is the only joint choice+RT model. Its AIC is not comparable to choice-only models M1-M3, M5, M6a, M6b, because the likelihood domains differ (choice-only vs. joint choice+RT).

> **Current winning model (choice-only):** M6b (dual perseveration with stick-breaking kappa_share). M6b has the lowest aggregate AIC and BIC across N=154 participants, with effectively unit Akaike weight. See [K Parameterization](#k-parameterization) (section 12) for K bounds.

**Key Design Principles:**
- **Learning phase only**: This task has no separate testing phase
- **Fixed β = 50**: Inverse temperature fixed during learning for identifiability
- **Epsilon noise**: Small random exploration term for robustness (choice-only models M1-M6)
- **Asymmetric learning**: Separate rates for positive/negative prediction errors
- **Model hierarchy**: M2 extends M1; M3 extends M2; M5/M6a/M6b/M4 extend M3 or M2

---

## 2. Q-Learning Model

### 2.1 Mathematical Formulation

#### Q-Value Update

Temporal difference learning with asymmetric learning rates:

```
Prediction Error: δ = r - Q(s,a)

If δ > 0:  Q(s,a) ← Q(s,a) + α₊ · δ  (positive PE, correct trials)
If δ ≤ 0:  Q(s,a) ← Q(s,a) + α₋ · δ  (negative PE, incorrect trials)
```

For this task, γ=0 (no bootstrapping), so the update simplifies to learning from immediate rewards only.

#### Action Selection

Softmax policy with epsilon noise:

```
p_softmax(a|s) = exp(β·Q(s,a)) / Σ_a' exp(β·Q(s,a'))

p(a|s) = ε/nA + (1-ε)·p_softmax(a|s)
```

Where:
- β = 50 (fixed)
- nA = 3 (number of actions)
- ε = epsilon noise parameter (fitted)

### 2.2 Parameters

| Parameter | Symbol | Range | Prior | Description |
|-----------|--------|-------|-------|-------------|
| Learning rate (positive PE) | α₊ | [0, 1] | Beta(3, 2) | How quickly Q-values increase from correct trials |
| Learning rate (negative PE) | α₋ | [0, 1] | Beta(2, 3) | How quickly Q-values decrease from incorrect trials |
| Epsilon noise | ε | [0, 1] | Beta(1, 19) | Random exploration probability |
| Inverse temperature | β | **50 (fixed)** | — | Exploitation sharpness (not fitted) |
| Initial Q-value | Q₀ | 0.5 | — | Optimistic initialization (not fitted) |

### 2.3 Parameter Interpretation

**Asymmetric Learning Rates (α₊, α₋)**
- **α₊ > α₋**: Optimistic learning (faster from rewards than punishments)
- **α₊ < α₋**: Pessimistic learning (faster from punishments)
- **Trauma hypothesis**: May show altered asymmetry (e.g., heightened α₋)

**Epsilon Noise (ε)**
- Captures random errors, lapses in attention, motor noise
- Typical values: 0.01-0.10 (1-10% random choices)
- Higher ε indicates less reliable responding

---

## 3. WM-RL Hybrid Model

### 3.1 Mathematical Formulation

The WM-RL model combines two learning systems with adaptive weighting.

#### 3.1.1 Working Memory Module

**WM Matrix**: State-action value matrix `WM[s, a]` storing immediate reward outcomes.

**Global Decay** (applied before update, every trial):
```
∀s,a: WM(s,a) ← (1 - φ)·WM(s,a) + φ·WM₀
```
Where WM₀ is the baseline value (typically 1/nA = 0.333).

**WM Update** (one-shot learning, α=1):
```
WM(s,a) ← r
```
This is an immediate overwrite with the observed reward.

**WM Policy**:
```
p_WM(a|s) = softmax(β · WM(s,:))
```
Where β = 50 (fixed).

#### 3.1.2 RL Module

Same asymmetric Q-learning as the standalone model:
```
δ = r - Q(s,a)
α = α₊ if δ > 0 else α₋
Q(s,a) ← Q(s,a) + α · δ
```

**RL Policy**:
```
p_RL(a|s) = softmax(β · Q(s,:))
```
Where β = 50 (fixed).

#### 3.1.3 Adaptive Hybrid Decision

**Adaptive Weighting**:
```
ω = ρ · min(1, K/Ns)
```
Where:
- ρ: Base WM reliance parameter (0-1)
- K: WM capacity
- Ns: Current set size (number of stimuli in block)

**Hybrid Policy** (before epsilon noise):
```
p_hybrid(a|s) = ω·p_WM(a|s) + (1-ω)·p_RL(a|s)
```

**Final Policy** (with epsilon noise):
```
p(a|s) = ε/nA + (1-ε)·p_hybrid(a|s)
```

### 3.2 Trial Sequence

The order of operations within each trial is:

1. **Decay WM**: `WM ← (1-φ)WM + φ·WM₀`
2. **Compute hybrid policy**: Use decayed WM and current Q for choice probabilities
3. **Update WM**: `WM(s,a) ← r` (after observing reward)
4. **Update Q**: `Q(s,a) ← Q(s,a) + α·(r - Q(s,a))`

### 3.3 Parameters

| Parameter | Symbol | Range | Prior | Description |
|-----------|--------|-------|-------|-------------|
| RL learning rate (positive) | α₊ | [0, 1] | Beta(3, 2) | RL update rate for correct trials |
| RL learning rate (negative) | α₋ | [0, 1] | Beta(2, 3) | RL update rate for incorrect trials |
| WM decay rate | φ | [0, 1] | Beta(2, 8) | Global decay toward baseline |
| Base WM reliance | ρ | [0, 1] | Beta(5, 2) | Base WM weight in adaptive formula |
| WM capacity | K | [1, 7] | TruncNorm(4, 1.5) | Capacity for adaptive weighting |
| Epsilon noise | ε | [0, 1] | Beta(1, 19) | Random exploration probability |
| Inverse temperature | β | **50 (fixed)** | — | Shared by WM and RL (not fitted) |
| WM baseline | WM₀ | 1/nA | — | Decay target (not fitted) |
| Initial Q-value | Q₀ | 0.5 | — | Optimistic initialization (not fitted) |

### 3.4 Parameter Interpretation

**WM Capacity (K)**
- Determines adaptive weighting via ω = ρ · min(1, K/Ns)
- When K ≥ set_size: Full WM reliance possible (ω = ρ)
- When K < set_size: Reduced WM reliance (ω = ρ·K/Ns)
- **Trauma hypothesis**: WM capacity deficits expected

**WM Decay Rate (φ)**
- **Low φ (0.0-0.1)**: Slow global decay; WM values persist
- **High φ (0.3-1.0)**: Fast decay; WM values quickly return to baseline
- Controls forgetting rate of all stimulus-action associations

**Base WM Reliance (ρ)**
- **ρ ≈ 0**: Minimal WM use (RL-dominant)
- **ρ ≈ 1**: Maximal WM use (WM-dominant when K ≥ Ns)
- Reflects individual preference for one-shot vs. incremental learning

### 3.5 Step-by-Step Pseudocode

This section provides detailed algorithms for both models, showing exactly how each quantity is computed and updated.

#### 3.5.1 Q-Learning Algorithm

```
ALGORITHM: Q-Learning Trial Loop
════════════════════════════════════════════════════════════════════════

INPUTS:
    α₊         = learning rate for positive prediction errors
    α₋         = learning rate for negative prediction errors
    β          = 50 (fixed inverse temperature)
    ε          = epsilon noise for random exploration
    nA         = 3 (number of actions)

INITIALIZATION (start of block):
    Q[s, a] = 0.5 for all s ∈ {0..5}, a ∈ {0, 1, 2}    # Optimistic init

FOR each trial t:
    ┌─────────────────────────────────────────────────────────────────┐
    │ STEP 1: Observe stimulus                                        │
    │─────────────────────────────────────────────────────────────────│
    │   s = current_stimulus                                          │
    └─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │ STEP 2: Compute action probabilities (softmax policy)           │
    │─────────────────────────────────────────────────────────────────│
    │   FOR each action a:                                            │
    │       exp_Q[a] = exp(β × Q[s, a])                               │
    │                                                                 │
    │   sum_exp = exp_Q[0] + exp_Q[1] + exp_Q[2]                      │
    │                                                                 │
    │   FOR each action a:                                            │
    │       p_softmax[a] = exp_Q[a] / sum_exp                         │
    └─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │ STEP 3: Apply epsilon noise                                     │
    │─────────────────────────────────────────────────────────────────│
    │   FOR each action a:                                            │
    │       p[a] = ε/nA + (1-ε) × p_softmax[a]                        │
    │            = ε/3 + (1-ε) × p_softmax[a]                         │
    │                                                                 │
    │   # This ensures p[a] ≥ ε/3 for all actions (exploration floor) │
    └─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │ STEP 4: Agent chooses action, environment returns reward        │
    │─────────────────────────────────────────────────────────────────│
    │   a_chosen ~ Categorical(p)     # Sampled from policy           │
    │   r = env.step(a_chosen)        # r ∈ {0, 1}                    │
    └─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │ STEP 5: Compute prediction error                                │
    │─────────────────────────────────────────────────────────────────│
    │   δ = r - Q[s, a_chosen]                                        │
    │                                                                 │
    │   # If r=1 (correct) and Q≈0.5:  δ ≈ +0.5 (positive PE)        │
    │   # If r=0 (incorrect) and Q≈0.5: δ ≈ -0.5 (negative PE)       │
    └─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │ STEP 6: Update Q-value with asymmetric learning rate            │
    │─────────────────────────────────────────────────────────────────│
    │   IF δ > 0:                                                     │
    │       α = α₊                     # Use positive learning rate   │
    │   ELSE:                                                         │
    │       α = α₋                     # Use negative learning rate   │
    │                                                                 │
    │   Q[s, a_chosen] ← Q[s, a_chosen] + α × δ                       │
    │                                                                 │
    │   # Example: Q=0.5, r=1, α₊=0.3                                 │
    │   #   δ = 1 - 0.5 = 0.5                                         │
    │   #   Q ← 0.5 + 0.3 × 0.5 = 0.65                                │
    └─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │ STEP 7: Record log-likelihood (for model fitting)               │
    │─────────────────────────────────────────────────────────────────│
    │   log_lik_t = log(p[a_chosen])                                  │
    │                                                                 │
    │   # This is computed BEFORE the Q-update (using choice probs    │
    │   # that generated the action)                                  │
    └─────────────────────────────────────────────────────────────────┘

END FOR

RETURN: sum(log_lik_t) for all trials
```

#### 3.5.2 WM-RL Hybrid Algorithm

```
ALGORITHM: WM-RL Hybrid Trial Loop
════════════════════════════════════════════════════════════════════════

INPUTS:
    α₊         = RL learning rate for positive prediction errors
    α₋         = RL learning rate for negative prediction errors
    β          = 50 (fixed inverse temperature)
    φ          = WM global decay rate
    ρ          = base WM reliance
    K          = WM capacity
    ε          = epsilon noise for random exploration
    nA         = 3 (number of actions)
    Ns         = current set size (stimuli in block)

INITIALIZATION (start of block):
    Q[s, a]  = 0.5     for all s ∈ {0..5}, a ∈ {0, 1, 2}  # RL values
    WM[s, a] = 1/nA    for all s ∈ {0..5}, a ∈ {0, 1, 2}  # WM values
                       = 0.333...                          # (uniform)

COMPUTE adaptive weight (constant for block):
    ω = ρ × min(1, K/Ns)

    # Examples:
    #   Ns=3, K=4: ω = ρ × min(1, 4/3) = ρ × 1 = ρ
    #   Ns=6, K=4: ω = ρ × min(1, 4/6) = ρ × 0.667

FOR each trial t:
    ┌─────────────────────────────────────────────────────────────────┐
    │ STEP 1: Observe stimulus                                        │
    │─────────────────────────────────────────────────────────────────│
    │   s = current_stimulus                                          │
    │   Ns = current_set_size                                         │
    └─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │ STEP 2: Apply global WM decay (BEFORE computing policy)         │
    │─────────────────────────────────────────────────────────────────│
    │   FOR all stimuli s' ∈ {0..5}:                                  │
    │       FOR all actions a' ∈ {0, 1, 2}:                           │
    │           WM[s', a'] ← (1 - φ) × WM[s', a'] + φ × (1/nA)        │
    │                                                                 │
    │   # Decay pulls ALL WM values toward 1/3 (uniform baseline)     │
    │   # High φ → fast forgetting; Low φ → persistent memory         │
    │                                                                 │
    │   # Example: WM=0.9, φ=0.1, baseline=0.333                      │
    │   #   WM ← 0.9 × 0.9 + 0.1 × 0.333 = 0.81 + 0.033 = 0.843      │
    └─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │ STEP 3: Compute WM policy (softmax over decayed WM)             │
    │─────────────────────────────────────────────────────────────────│
    │   FOR each action a:                                            │
    │       exp_WM[a] = exp(β × WM[s, a])                             │
    │                                                                 │
    │   sum_exp_WM = exp_WM[0] + exp_WM[1] + exp_WM[2]                │
    │                                                                 │
    │   FOR each action a:                                            │
    │       p_WM[a] = exp_WM[a] / sum_exp_WM                          │
    └─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │ STEP 4: Compute RL policy (softmax over Q-values)               │
    │─────────────────────────────────────────────────────────────────│
    │   FOR each action a:                                            │
    │       exp_Q[a] = exp(β × Q[s, a])                               │
    │                                                                 │
    │   sum_exp_Q = exp_Q[0] + exp_Q[1] + exp_Q[2]                    │
    │                                                                 │
    │   FOR each action a:                                            │
    │       p_RL[a] = exp_Q[a] / sum_exp_Q                            │
    └─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │ STEP 5: Compute adaptive weight                                 │
    │─────────────────────────────────────────────────────────────────│
    │   ω = ρ × min(1, K/Ns)                                          │
    │                                                                 │
    │   # ω determines WM vs RL contribution to final policy          │
    │   # High ω → WM-dominant; Low ω → RL-dominant                   │
    └─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │ STEP 6: Combine WM and RL policies (hybrid)                     │
    │─────────────────────────────────────────────────────────────────│
    │   FOR each action a:                                            │
    │       p_hybrid[a] = ω × p_WM[a] + (1 - ω) × p_RL[a]             │
    │                                                                 │
    │   # Linear interpolation between the two systems                │
    └─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │ STEP 7: Apply epsilon noise                                     │
    │─────────────────────────────────────────────────────────────────│
    │   FOR each action a:                                            │
    │       p[a] = ε/nA + (1-ε) × p_hybrid[a]                         │
    │            = ε/3 + (1-ε) × p_hybrid[a]                          │
    └─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │ STEP 8: Agent chooses action, environment returns reward        │
    │─────────────────────────────────────────────────────────────────│
    │   a_chosen ~ Categorical(p)     # Sampled from policy           │
    │   r = env.step(a_chosen)        # r ∈ {0, 1}                    │
    └─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │ STEP 9: Update WM (immediate overwrite, one-shot learning)      │
    │─────────────────────────────────────────────────────────────────│
    │   WM[s, a_chosen] ← r                                           │
    │                                                                 │
    │   # If correct (r=1): WM[s,a] = 1.0 (remember: this works!)     │
    │   # If incorrect (r=0): WM[s,a] = 0.0 (remember: avoid this!)   │
    │   # No learning rate — immediate overwrite (α_WM = 1)           │
    └─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │ STEP 10: Update RL Q-value (prediction error learning)          │
    │─────────────────────────────────────────────────────────────────│
    │   δ = r - Q[s, a_chosen]                                        │
    │                                                                 │
    │   IF δ > 0:                                                     │
    │       α = α₊                                                    │
    │   ELSE:                                                         │
    │       α = α₋                                                    │
    │                                                                 │
    │   Q[s, a_chosen] ← Q[s, a_chosen] + α × δ                       │
    └─────────────────────────────────────────────────────────────────┘

    ┌─────────────────────────────────────────────────────────────────┐
    │ STEP 11: Record log-likelihood (for model fitting)              │
    │─────────────────────────────────────────────────────────────────│
    │   log_lik_t = log(p[a_chosen])                                  │
    │                                                                 │
    │   # Computed using choice probabilities BEFORE updates          │
    └─────────────────────────────────────────────────────────────────┘

END FOR

RETURN: sum(log_lik_t) for all trials
```

#### 3.5.3 Worked Example: WM-RL First 3 Trials

```
WORKED EXAMPLE: Set size = 2, Stimuli = {0, 1}
═══════════════════════════════════════════════════════════════════════

Parameters: α₊=0.3, α₋=0.1, β=50, φ=0.1, ρ=0.8, K=4, ε=0.05

Initial values:
    Q  = [[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], ...]  # All 0.5
    WM = [[0.33, 0.33, 0.33], ...]                 # All 1/3

Adaptive weight: ω = 0.8 × min(1, 4/2) = 0.8 × 1 = 0.8

───────────────────────────────────────────────────────────────────────
TRIAL 1: stimulus=0, correct_action=1
───────────────────────────────────────────────────────────────────────

1. DECAY WM (all entries):
   WM[0,0] = 0.9×0.33 + 0.1×0.33 = 0.33  (no change at baseline)
   (same for all)

2. COMPUTE POLICIES:
   WM[0,:] = [0.33, 0.33, 0.33]
   p_WM = softmax(50×[0.33,0.33,0.33]) = [0.33, 0.33, 0.33] (uniform)

   Q[0,:] = [0.5, 0.5, 0.5]
   p_RL = softmax(50×[0.5,0.5,0.5]) = [0.33, 0.33, 0.33] (uniform)

3. HYBRID + EPSILON:
   p_hybrid = 0.8×[0.33,0.33,0.33] + 0.2×[0.33,0.33,0.33] = [0.33,0.33,0.33]
   p = 0.05/3 + 0.95×[0.33,0.33,0.33] ≈ [0.33, 0.33, 0.33]

4. CHOICE: Agent randomly selects action=2 (incorrect)
   Reward: r=0

5. UPDATE WM:
   WM[0,2] = 0  (remember: action 2 was wrong for stimulus 0)

6. UPDATE Q:
   δ = 0 - 0.5 = -0.5 (negative PE)
   Q[0,2] = 0.5 + 0.1×(-0.5) = 0.45

State after trial 1:
    WM[0,:] = [0.33, 0.33, 0.00]  ← Updated
    Q[0,:]  = [0.50, 0.50, 0.45]  ← Updated

───────────────────────────────────────────────────────────────────────
TRIAL 2: stimulus=1, correct_action=0
───────────────────────────────────────────────────────────────────────

1. DECAY WM (all entries):
   WM[0,0] = 0.9×0.33 + 0.1×0.33 = 0.33
   WM[0,1] = 0.9×0.33 + 0.1×0.33 = 0.33
   WM[0,2] = 0.9×0.00 + 0.1×0.33 = 0.033  ← Decay from 0 toward baseline
   (stimulus 1 still at baseline)

2. COMPUTE POLICIES for stimulus=1:
   WM[1,:] = [0.33, 0.33, 0.33]  → p_WM = [0.33, 0.33, 0.33]
   Q[1,:]  = [0.50, 0.50, 0.50]  → p_RL = [0.33, 0.33, 0.33]

3. HYBRID + EPSILON:
   p ≈ [0.33, 0.33, 0.33] (still uniform for stimulus 1)

4. CHOICE: Agent selects action=0 (correct!)
   Reward: r=1

5. UPDATE WM:
   WM[1,0] = 1  (remember: action 0 was correct for stimulus 1)

6. UPDATE Q:
   δ = 1 - 0.5 = 0.5 (positive PE)
   Q[1,0] = 0.5 + 0.3×0.5 = 0.65

State after trial 2:
    WM[1,:] = [1.00, 0.33, 0.33]  ← Updated
    Q[1,:]  = [0.65, 0.50, 0.50]  ← Updated

───────────────────────────────────────────────────────────────────────
TRIAL 3: stimulus=0, correct_action=1 (same stimulus as trial 1)
───────────────────────────────────────────────────────────────────────

1. DECAY WM:
   WM[0,0] = 0.9×0.33 + 0.1×0.33 = 0.33
   WM[0,1] = 0.9×0.33 + 0.1×0.33 = 0.33
   WM[0,2] = 0.9×0.033 + 0.1×0.33 = 0.063  ← Continues recovering
   WM[1,0] = 0.9×1.00 + 0.1×0.33 = 0.933   ← Decays from 1
   WM[1,1] = 0.9×0.33 + 0.1×0.33 = 0.33
   WM[1,2] = 0.9×0.33 + 0.1×0.33 = 0.33

2. COMPUTE POLICIES for stimulus=0:
   WM[0,:] = [0.33, 0.33, 0.063]

   exp(50×WM) ≈ [exp(16.5), exp(16.5), exp(3.15)]
              ≈ [1.5e7, 1.5e7, 23]

   p_WM ≈ [0.50, 0.50, 0.00]  ← Now avoids action 2!

   Q[0,:] = [0.50, 0.50, 0.45]
   p_RL ≈ [0.37, 0.37, 0.26]  ← Slight bias away from action 2

3. HYBRID + EPSILON:
   p_hybrid = 0.8×[0.50,0.50,0.00] + 0.2×[0.37,0.37,0.26]
            = [0.47, 0.47, 0.05]

   p = 0.05/3 + 0.95×[0.47,0.47,0.05]
     = [0.46, 0.46, 0.07]

4. NOW agent is much more likely to avoid action 2!

KEY INSIGHT: WM quickly learned to avoid action 2 after one trial,
while RL is still catching up. The hybrid model leverages both systems.
```

### 3.6 M3: WM-RL + Perseveration (kappa)

M3 extends M2 with a **choice kernel** that captures the tendency to repeat the last action taken for a given stimulus, independent of value.

#### 3.6.1 Perseveration Mechanism

**Choice Kernel:** A one-hot vector C_k tracking the last action taken for each stimulus:
```
C_k(a|s) = 1 if a was the last action chosen for stimulus s
C_k(a|s) = 0 otherwise
```

At the first presentation of a stimulus within a block (no prior action), the kernel is not applied (uniform fallback).

**Perseveration-augmented policy** (before epsilon noise):
```
p_persist(a|s) = (1 - κ) * p_hybrid(a|s) + κ * C_k(a|s)
```

Where `p_hybrid(a|s)` is the M2 hybrid policy (Section 3.1.3) and κ is the perseveration strength.

**Final policy with epsilon noise:**
```
p(a|s) = ε/nA + (1-ε) * p_persist(a|s)
```

#### 3.6.2 Trial Sequence (extends M2)

1. **Decay WM**: `WM <- (1-φ)WM + φ*WM_0`
2. **Compute hybrid policy**: `p_hybrid = ω*p_WM + (1-ω)*p_RL`
3. **Apply perseveration kernel**: `p_persist = (1-κ)*p_hybrid + κ*C_k` (if stimulus seen before in block)
4. **Apply epsilon noise**: `p = ε/nA + (1-ε)*p_persist`
5. **Update WM**: `WM(s,a) <- r`
6. **Update Q**: `Q(s,a) <- Q(s,a) + α*(r - Q(s,a))`
7. **Update kernel**: `C_k(s) <- one_hot(a_chosen)` (unconditional)

#### 3.6.3 Parameters

| Parameter | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| RL learning rate (positive) | α₊ | [0, 1] | RL update rate for correct trials |
| RL learning rate (negative) | α₋ | [0, 1] | RL update rate for incorrect trials |
| WM decay rate | φ | [0, 1] | Global WM decay toward baseline |
| Base WM reliance | ρ | [0, 1] | Base WM weight in adaptive formula |
| WM capacity | K | [1, 7] | Capacity for adaptive weighting |
| Perseveration strength | κ | [0, 1] | Strength of choice persistence; κ=0 reduces to M2 |
| Epsilon noise | ε | [0, 1] | Random exploration probability |

**Code reference:** `WMRL_M3_PARAMS` in `mle_utils.py` — 7 parameters: `['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa', 'epsilon']`

### 3.7 M5: WM-RL + RL Forgetting (phi_rl)

M5 extends M3 with **Q-value decay toward baseline** before each delta-rule update, modeling RL forgetting.

#### 3.7.1 RL Forgetting Mechanism

Each trial, **before** the standard asymmetric delta-rule update, ALL Q-values decay toward the baseline:
```
For ALL (s,a) pairs:
    Q(s,a) <- (1 - φ_rl) * Q(s,a) + φ_rl * Q_0
```

Where Q_0 = 1/nA = 0.333 (uniform baseline, matching WM baseline convention).

Then the standard asymmetric delta-rule update is applied for the observed (s,a) pair:
```
δ = r - Q(s,a)
α = α₊ if δ > 0 else α₋
Q(s,a) <- Q(s,a) + α * δ
```

**Key property:** When φ_rl = 0, no decay occurs and M5 reduces exactly to M3 (verified: 0.00e+00 difference).

#### 3.7.2 Trial Sequence (extends M3)

1. **Decay WM**: `WM <- (1-φ)WM + φ*WM_0`
2. **Decay Q-values**: `Q <- (1-φ_rl)*Q + φ_rl*Q_0` (ALL state-action pairs)
3. **Compute hybrid policy**: `p_hybrid = ω*p_WM + (1-ω)*p_RL`
4. **Apply perseveration kernel**: `p_persist = (1-κ)*p_hybrid + κ*C_k`
5. **Apply epsilon noise**: `p = ε/nA + (1-ε)*p_persist`
6. **Update WM**: `WM(s,a) <- r`
7. **Update Q**: `Q(s,a) <- Q(s,a) + α*(r - Q(s,a))`
8. **Update kernel**: `C_k(s) <- one_hot(a_chosen)`

#### 3.7.3 Parameters

| Parameter | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| RL learning rate (positive) | α₊ | [0, 1] | RL update rate for correct trials |
| RL learning rate (negative) | α₋ | [0, 1] | RL update rate for incorrect trials |
| WM decay rate | φ | [0, 1] | Global WM decay toward baseline |
| Base WM reliance | ρ | [0, 1] | Base WM weight in adaptive formula |
| WM capacity | K | [1, 7] | Capacity for adaptive weighting |
| Perseveration strength | κ | [0, 1] | Global choice persistence (inherited from M3) |
| RL forgetting rate | φ_rl | [0, 1] | Q-value decay rate toward Q_0; φ_rl=0 reduces to M3 |
| Epsilon noise | ε | [0, 1] | Random exploration probability |

**Code reference:** `WMRL_M5_PARAMS` in `mle_utils.py` — 8 parameters: `['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa', 'phi_rl', 'epsilon']`

### 3.8 M6a: WM-RL + Stimulus-Specific Perseveration (kappa_s)

M6a replaces M3's **global** perseveration with **per-stimulus** tracking. Instead of a single scalar tracking the last action across all stimuli, M6a maintains a separate last-action record for each stimulus.

#### 3.8.1 Stimulus-Specific Kernel

**Per-stimulus tracking:** An array `last_actions` of shape `(num_stimuli,)`, initialized to -1 (sentinel for "not yet seen in this block").

On each trial for stimulus s:
```
If last_actions[s] >= 0:  (stimulus seen before in this block)
    C_k(a|s) = 1 if a == last_actions[s], else 0
    p_persist(a|s) = (1 - κ_s) * p_hybrid(a|s) + κ_s * C_k(a|s)
Else:  (first presentation in block)
    p_persist(a|s) = p_hybrid(a|s)  (no kernel, uniform fallback)

After action: last_actions[s] = a_chosen  (unconditional update)
```

**Block boundaries:** All `last_actions` reset to -1 at the start of each block.

#### 3.8.2 Key Difference from M3

M3 uses a single global `last_action` scalar: whatever action was taken on the previous trial (regardless of stimulus). M6a tracks per-stimulus, so perseveration only reflects repeating the action most recently taken for *that specific stimulus*.

M6a has the same number of free parameters as M3 (7) but a different perseveration mechanism.

#### 3.8.3 Parameters

| Parameter | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| RL learning rate (positive) | α₊ | [0, 1] | RL update rate for correct trials |
| RL learning rate (negative) | α₋ | [0, 1] | RL update rate for incorrect trials |
| WM decay rate | φ | [0, 1] | Global WM decay toward baseline |
| Base WM reliance | ρ | [0, 1] | Base WM weight in adaptive formula |
| WM capacity | K | [1, 7] | Capacity for adaptive weighting |
| Stimulus-specific perseveration | κ_s | [0, 1] | Per-stimulus choice persistence; κ_s=0 reduces to M2 |
| Epsilon noise | ε | [0, 1] | Random exploration probability |

**Code reference:** `WMRL_M6A_PARAMS` in `mle_utils.py` — 7 parameters: `['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa_s', 'epsilon']`

### 3.9 M6b: WM-RL + Dual Perseveration (stick-breaking)

M6b combines **both** global (M3-style) and stimulus-specific (M6a-style) perseveration, using a stick-breaking reparameterization to enforce a total budget constraint.

#### 3.9.1 Stick-Breaking Reparameterization

**Fitted parameters:**
```
κ_total ∈ [0, 1]  — Total perseveration budget
κ_share ∈ [0, 1]  — Proportion allocated to global kernel
```

**Decoded parameters:**
```
κ       = κ_total * κ_share          — Global perseveration strength
κ_s     = κ_total * (1 - κ_share)    — Stimulus-specific perseveration strength
```

This enforces κ + κ_s = κ_total <= 1, preventing the kernels from dominating the hybrid policy.

**Special cases:**
- κ_share = 1 reduces exactly to M3 (only global kernel; verified: 0.0e+00 difference)
- κ_share = 0 reduces exactly to M6a (only stimulus-specific kernel; verified: 0.0e+00 difference)

#### 3.9.2 Dual Carry

M6b maintains **both** carry variables:
- `last_action` (scalar): The globally last-chosen action (M3-style)
- `last_actions` (array, shape num_stimuli): Per-stimulus last-chosen actions (M6a-style)

Both are reset at block boundaries. Both are updated unconditionally after each action.

**Effective-weight gating:** If a stimulus has not yet been seen in the current block, the κ_s portion is zeroed out:
```
eff_kappa_s = κ_s if last_actions[s] >= 0 else 0
```

#### 3.9.3 Combined Policy

```
p_persist(a|s) = (1 - κ - eff_kappa_s) * p_hybrid(a|s)
              + κ * C_global(a)
              + eff_kappa_s * C_stim(a|s)
```

Where:
- `C_global(a) = 1 if a == last_action` (global kernel)
- `C_stim(a|s) = 1 if a == last_actions[s]` (stimulus-specific kernel)

#### 3.9.4 Parameters

| Parameter | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| RL learning rate (positive) | α₊ | [0, 1] | RL update rate for correct trials |
| RL learning rate (negative) | α₋ | [0, 1] | RL update rate for incorrect trials |
| WM decay rate | φ | [0, 1] | Global WM decay toward baseline |
| Base WM reliance | ρ | [0, 1] | Base WM weight in adaptive formula |
| WM capacity | K | [1, 7] | Capacity for adaptive weighting |
| Total perseveration budget | κ_total | [0, 1] | Combined perseveration strength |
| Global share | κ_share | [0, 1] | Proportion of κ_total for global kernel |
| Epsilon noise | ε | [0, 1] | Random exploration probability |

**Code reference:** `WMRL_M6B_PARAMS` in `mle_utils.py` — 8 parameters: `['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa_total', 'kappa_share', 'epsilon']`

> **Note:** The stick-breaking decode (κ = κ_total * κ_share; κ_s = κ_total * (1 - κ_share)) is performed in objective functions only, not in transform functions. The block likelihood function receives decoded κ and κ_s.

### 3.10 M4: RLWM-LBA Joint Choice+RT

M4 combines M3's hybrid learning with a **Linear Ballistic Accumulator** (Brown & Heathcote, 2008) for joint choice and response time modeling.

#### 3.10.1 Learning Module

Same as M3: hybrid WM-RL policy with kappa perseveration. The learning module produces a choice probability vector `p_hybrid(a|s)` for each trial.

#### 3.10.2 Decision Module: Linear Ballistic Accumulator

The LBA replaces softmax action selection with a racing-accumulator model that jointly predicts **which** action is chosen and **how fast** the response is.

**Accumulator setup** (one per action, i = 1, ..., nA):
```
Drift rate:   v_i = v_scale * p_hybrid(a_i | s_t)
Start point:  k_i ~ Uniform(0, A)
Threshold:    b = A + δ   (b > A by construction via reparameterization)
Noise:        s = 0.1     (fixed, not a free parameter)
```

Where:
- `v_scale` scales the hybrid policy probabilities into drift rates
- `A` is the maximum start point (uniform start-point variability)
- `δ` is the threshold gap (ensures b > A)
- `t_0` is the non-decision time

**Race dynamics:** Each accumulator i races from start point k_i toward threshold b with constant velocity v_i. The winner is the first accumulator to reach threshold.

**Response time:**
```
RT = accumulation_time(winner) + t_0
```

#### 3.10.3 Joint Likelihood

The joint log-likelihood of observing choice i at time t is:
```
log P(choice=i, RT=t) = log f_i(t - t_0) + Σ_{j≠i} log S_j(t - t_0)
```

Where:
- `f_i(t)` = LBA probability density function for accumulator i at time t
- `S_j(t)` = LBA survivor function (1 - CDF) for accumulator j at time t

The LBA PDF and CDF involve the normal distribution:
```
f_i(t) = (1/A) * [ φ((b - v_i*t) / (s*t)) - φ((b - A - v_i*t) / (s*t)) ]
         + (v_i / A) * [ Φ((b - v_i*t) / (s*t)) - Φ((b - A - v_i*t) / (s*t)) ]
```

Where φ is the standard normal PDF and Φ is the standard normal CDF.

#### 3.10.4 Key Differences from Choice-Only Models

- **No epsilon parameter**: Start-point variability A subsumes undirected exploration
- **Requires float64**: CDF computations need double precision for numerical stability
- **AIC not comparable**: Joint choice+RT likelihood operates on a different domain than choice-only likelihoods, so AIC values are incommensurable
- **Log-density can be positive**: The defective PDF integrates to <1 over (0, inf) but can exceed 1 at a point

#### 3.10.5 Parameters

| Parameter | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| RL learning rate (positive) | α₊ | [0, 1] | RL update rate for correct trials |
| RL learning rate (negative) | α₋ | [0, 1] | RL update rate for incorrect trials |
| WM decay rate | φ | [0, 1] | Global WM decay toward baseline |
| Base WM reliance | ρ | [0, 1] | Base WM weight in adaptive formula |
| WM capacity | K | [1, 7] | Capacity for adaptive weighting |
| Perseveration strength | κ | [0, 1] | Global choice persistence (inherited from M3) |
| Drift rate scaling | v_scale | [0.1, 20] | Scales hybrid policy into drift rates |
| Max start point | A | [0.001, 2] | Uniform start-point variability (seconds) |
| Threshold gap | δ | [0.001, 2] | b - A gap; b = A + δ (decoded in objectives) |
| Non-decision time | t₀ | [0.05, 0.3] | Motor/encoding time (seconds) |

**Code reference:** `WMRL_M4_PARAMS` in `mle_utils.py` — 10 parameters: `['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa', 'v_scale', 'A', 'delta', 't0']`

> **Reference:** Brown, S. D., & Heathcote, A. (2008). The simplest complete model of choice response time: Linear ballistic accumulation. *Cognitive Psychology*, 57(3), 153-178.

---

## 4. Comparison to Senta et al. (2025)

This implementation is based on Senta et al. (2025), "Dual process impairments in reinforcement learning and working memory systems underlie learning deficits in physiological anxiety."

### 4.1 Key Alignments

| Component | Senta et al. (2025) | Our Implementation | Match |
|-----------|---------------------|-------------------|-------|
| WM Update | WM(s,a) ← r (immediate overwrite) | WM(s,a) ← r | ✓ |
| WM Decay | (1-φ)WM + φ·baseline | (1-φ)WM + φ·WM₀ | ✓ |
| Adaptive Weight | ω = ρ · min(1, K/ns) | ω = ρ · min(1, K/Ns) | ✓ |
| Hybrid Policy | ω·p_WM + (1-ω)·p_RL | ω·p_WM + (1-ω)·p_RL | ✓ |
| Asymmetric Learning | α₊ for δ>0, α₋ for δ≤0 | Same | ✓ |
| Fixed β | β = 50 during learning | β = 50 | ✓ |
| Epsilon Noise | ε/nA + (1-ε)·p | ε/nA + (1-ε)·p | ✓ |

### 4.2 Simplifications and Extensions

Our implementation extends the basic WM-RL model with perseveration (M3, M5, M6a, M6b) and joint choice+RT modeling (M4). The following features from Senta et al. remain unimplemented:

| Feature | Senta et al. | Our Implementation | Reason |
|---------|--------------|-------------------|--------|
| Perseveration (choice kernel) | Global choice repetition bias | **Implemented** as M3 (κ), M6a (κ_s), M6b (dual) | Core extension for dissociating perseveration from learning |
| RL forgetting | Q-value decay toward baseline | **Implemented** as M5 (φ_rl) | Tests whether forgetting improves fit |
| Information sharing (i) | RL receives scaled WM info | Not included | Adds complexity; not needed for current hypotheses |
| Negative feedback bias (η) | Scales α₋ for WM-initiated | Not included | Advanced variant |
| Split WM confidence (ρ_low, ρ_high) | Separate ρ for low/high load | Single ρ | Start with basic model |
| Testing phase | Separate β_test | Not included | Task has no testing phase |
| Lapse rate | η for WM errors | Epsilon covers this | Simplified |

### 4.3 References

**Primary Reference:**
> Senta, D., et al. (2025). Dual process impairments in reinforcement learning and working memory systems underlie learning deficits in physiological anxiety.

**Model Equations** (from paper, pages 17-19):
- Equation 1: WM update with decay
- Equation 2: Adaptive weighting
- Equation 3: Hybrid policy
- Equation 4: Epsilon-noisy choice

### 4.4 Implementation Comparison: Senta et al. vs Ours

This section details the methodological differences between our implementation and the original Senta et al. (2025) approach across four key areas.

#### 4.4.1 Model Fitting

| Aspect | Senta et al. (2025) | Our Implementation |
|--------|---------------------|-------------------|
| **Framework** | Maximum Likelihood Estimation (MLE) | Hierarchical Bayesian Inference |
| **Software** | MATLAB `fmincon` | NumPyro/JAX with NUTS sampler |
| **Optimization** | Point estimate via gradient-free search | Full posterior distribution via MCMC |
| **Starting points** | 20 random starting points per participant | Not applicable (MCMC explores full space) |
| **Regularization** | None (pure MLE) | Informative priors on parameters |
| **Hierarchical** | Individual fits, no pooling | Hierarchical with partial pooling |
| **Output** | Single best-fit parameter per participant | Posterior distribution per participant |
| **Uncertainty** | Bootstrap or Hessian-based SEs | Posterior credible intervals |

**Why Hierarchical Bayesian?**
- **Partial pooling**: Shrinks extreme individual estimates toward group mean, reducing overfitting
- **Principled uncertainty**: Credible intervals reflect parameter uncertainty without additional bootstrapping
- **Better for small N**: Borrowing strength across participants helps with limited data
- **Priors encode knowledge**: Domain knowledge (e.g., learning rates likely 0.1-0.5) regularizes estimates

**Trade-offs:**
- Bayesian: More computationally expensive, requires prior specification
- MLE: Faster, simpler, but point estimates can be noisy with limited data

#### 4.4.2 Model Comparison

| Aspect | Senta et al. (2025) | Our Implementation |
|--------|---------------------|-------------------|
| **Metric** | AIC (Akaike Information Criterion) | LOO-CV (Leave-One-Out Cross-Validation) and WAIC |
| **Computation** | AIC = 2k - 2·log(L) | LOO via Pareto-smoothed importance sampling (PSIS) |
| **Penalty** | Fixed penalty (2 per parameter) | Effective number of parameters (p_eff) |
| **Aggregation** | Sum AIC across participants | Compute per-participant, sum elpd |
| **Selection rule** | Lower AIC wins | Higher elpd (lower LOO) wins |

**AIC Formula (Senta):**
```
AIC = 2k - 2·ln(L_max)

where:
  k = number of free parameters
  L_max = maximum likelihood value
```

**LOO-CV Formula (Ours):**
```
elpd_loo = Σᵢ log p(yᵢ | y₋ᵢ)

Estimated via PSIS-LOO without refitting:
  elpd_loo ≈ Σᵢ log( Σₛ wᵢₛ · p(yᵢ|θₛ) )

where:
  wᵢₛ = Pareto-smoothed importance weights
  θₛ = posterior samples
```

**Why LOO over AIC?**
- **Better for Bayesian**: Uses full posterior, not just point estimate
- **More robust**: Less sensitive to model misspecification
- **Diagnostics**: Pareto-k diagnostic flags problematic observations
- **Predictive focus**: Directly estimates out-of-sample prediction accuracy

#### 4.4.3 Parameter Recovery

| Aspect | Senta et al. (2025) | Our Implementation |
|--------|---------------------|-------------------|
| **Approach** | Simulate → Fit → Correlate | Simulate → Fit → Compare posteriors |
| **Data generation** | Use fit parameters as "true" values | Specify known ground-truth parameters |
| **N simulated** | Same as empirical sample size | Flexible (typically 20-100 synthetic participants) |
| **Success criterion** | r ≥ 0.80 between true and recovered | Posterior contains true value (coverage) |
| **Metrics** | Pearson correlation | Correlation + bias + coverage + RMSE |

**Senta et al. Procedure:**
```
1. Fit model to real data → θ̂ᵢ for each participant
2. Simulate synthetic data using θ̂ᵢ as generative parameters
3. Fit model to synthetic data → θ̂'ᵢ (recovered)
4. Compute correlation: r = cor(θ̂, θ̂')
5. Accept if r ≥ 0.80
```

**Our Procedure:**
```
1. Define ground-truth parameters θ_true (either fixed or sampled from prior)
2. Simulate synthetic datasets using generative model
3. Fit model to synthetic data → posterior p(θ|data)
4. Compute recovery metrics:
   - Correlation: cor(θ_true, E[θ|data])
   - Bias: mean(E[θ|data] - θ_true)
   - Coverage: proportion of 95% CIs containing θ_true
   - RMSE: sqrt(mean((E[θ|data] - θ_true)²))
5. Accept if correlation ≥ 0.80 AND coverage ≈ 0.95
```

**Key Difference:** We check both correlation (identifiability) and coverage (calibration). A model can have high correlation but poor coverage if uncertainty is systematically under/overestimated.

#### 4.4.4 Model Validation

| Aspect | Senta et al. (2025) | Our Implementation |
|--------|---------------------|-------------------|
| **Approach** | Posterior predictive checks | Same conceptual approach |
| **Focus** | Learning curves by set size | Learning curves + behavioral patterns |
| **Aggregation** | Group-level patterns | Individual + group |
| **Key patterns** | (1) Learning curves (2) Low vs high load difference | Same |

**Senta et al. Validation Criteria:**
```
The winning model should replicate:
1. Learning curves: Accuracy increases over trials within blocks
2. Set size effect: Performance higher for low (2,3) vs high (5,6) set sizes
3. Low-high difference during learning: Larger gap early, convergence late
```

**Our Validation Approach:**
```python
# 1. Generate posterior predictive samples
for each posterior sample θ:
    simulated_data = simulate_experiment(θ)

# 2. Compute summary statistics
for simulated_data in posterior_predictive:
    learning_curve_sim = compute_learning_curve(simulated_data)
    set_size_effect_sim = compute_set_size_effect(simulated_data)

# 3. Compare to observed data
plot_posterior_predictive_check(observed, simulated)
```

**Shared Philosophy:** Both approaches use generative model simulations to verify the model captures key behavioral patterns, not just fits well numerically.

#### 4.4.5 Summary Comparison Table

| Component | Senta et al. (2025) | Our Implementation | Rationale for Difference |
|-----------|---------------------|-------------------|--------------------------|
| **Fitting** | MLE (fmincon, 20 starts) | Hierarchical Bayesian (NUTS) | Better uncertainty quantification |
| **Priors** | None | Informative (Beta, TruncNorm) | Regularization + domain knowledge |
| **Pooling** | None (individual fits) | Partial (hierarchical) | Reduces overfitting |
| **Model comparison** | AIC | LOO-CV (PSIS) | Better for Bayesian, more robust |
| **Recovery metric** | Correlation ≥ 0.80 | Correlation + Coverage | Checks calibration too |
| **Validation** | Posterior predictive | Same | — |
| **Testing phase** | Yes (β_test fitted) | No | Task doesn't have testing |
| **Software** | MATLAB | Python (JAX/NumPyro) | Open source, GPU-accelerated |

---

## 5. Fitting Implementation

### 5.1 Approach: Hierarchical Bayesian Inference

We use NumPyro with the NUTS sampler for gradient-based Bayesian inference.

**Hierarchical Structure:**
```
Group-level: μ_θ, σ_θ (population mean/SD for each parameter θ)
    ↓
Individual-level: θ_i ~ Normal(μ_θ, σ_θ) (participant parameters)
    ↓
Likelihood: actions_i ~ Softmax(Q-values; θ_i)
```

**Non-centered Parameterization** (for better sampling):
```python
z_θ_i ~ Normal(0, 1)
θ_i = transform(μ_θ + σ_θ * z_θ_i)
```

### 5.2 Key Files

| File | Purpose |
|------|---------|
| `scripts/fitting/jax_likelihoods.py` | Pure JAX likelihood functions |
| `scripts/fitting/numpyro_models.py` | Hierarchical Bayesian models |
| `scripts/fitting/fit_with_jax.py` | Main fitting script |
| `scripts/fitting/fit_both_models.py` | Fit and compare both models |

### 5.3 JAX Likelihood Functions

The core likelihoods use `jax.lax.scan()` for efficient sequential operations:

**Q-Learning Block Likelihood:**
```python
from scripts.fitting.jax_likelihoods import q_learning_block_likelihood

log_lik = q_learning_block_likelihood(
    stimuli=jnp.array([0, 1, 0, 2, 1]),
    actions=jnp.array([0, 1, 0, 2, 1]),
    rewards=jnp.array([1.0, 0.0, 1.0, 1.0, 0.0]),
    alpha_pos=0.3,
    alpha_neg=0.1,
    beta=50.0,
    epsilon=0.05
)
```

**WM-RL Block Likelihood:**
```python
from scripts.fitting.jax_likelihoods import wmrl_block_likelihood

log_lik = wmrl_block_likelihood(
    stimuli=stim_array,
    actions=act_array,
    rewards=rew_array,
    set_sizes=set_array,
    alpha_pos=0.3,
    alpha_neg=0.1,
    beta=50.0,  # Fixed
    phi=0.1,
    rho=0.7,
    capacity=4.0,
    epsilon=0.05
)
```

### 5.4 Running Fitting

**Basic Usage:**
```bash
# Fit Q-learning model
python scripts/fitting/fit_with_jax.py \
    --model qlearning \
    --data output/task_trials_long.csv \
    --chains 4 \
    --warmup 1000 \
    --samples 2000

# Fit WM-RL model
python scripts/fitting/fit_with_jax.py \
    --model wmrl \
    --data output/task_trials_long.csv \
    --chains 4 \
    --warmup 1000 \
    --samples 2000
```

**Fit Both Models:**
```bash
python scripts/fitting/fit_both_models.py \
    --data output/task_trials_long.csv \
    --chains 4 \
    --output output/v1/
```

### 5.5 Programmatic Usage

```python
from scripts.fitting.numpyro_models import (
    qlearning_hierarchical_model,
    prepare_data_for_numpyro,
    run_inference
)
import pandas as pd

# Load and prepare data
data = pd.read_csv('data/processed/task_trials_long.csv')
participant_data = prepare_data_for_numpyro(data)

# Run MCMC inference
mcmc = run_inference(
    model=qlearning_hierarchical_model,
    model_args={'participant_data': participant_data},
    num_warmup=1000,
    num_samples=2000,
    num_chains=4,
    seed=42
)

# Get samples
samples = mcmc.get_samples()
print(f"μ_α₊: {samples['mu_alpha_pos'].mean():.3f}")
print(f"μ_α₋: {samples['mu_alpha_neg'].mean():.3f}")
print(f"μ_ε: {samples['mu_epsilon'].mean():.3f}")

# Save results
import arviz as az
idata = az.from_numpyro(mcmc)
idata.to_netcdf('models/bayesian/posterior.nc')
```

### 5.6 Prior Specifications

**Q-Learning Model Priors:**
```python
# Group-level
mu_alpha_pos ~ Beta(3, 2)        # Mean ~ 0.6
mu_alpha_neg ~ Beta(2, 3)        # Mean ~ 0.4
mu_epsilon ~ Beta(1, 19)         # Mean ~ 0.05
sigma_alpha_pos ~ HalfNormal(0.3)
sigma_alpha_neg ~ HalfNormal(0.3)
sigma_epsilon ~ HalfNormal(0.1)

# Individual-level (non-centered)
z_i ~ Normal(0, 1)
alpha_pos_i = logit⁻¹(logit(mu_alpha_pos) + sigma_alpha_pos * z_i)
```

**WM-RL Additional Priors:**
```python
mu_phi ~ Beta(2, 8)              # Mean ~ 0.2 (slow decay)
mu_rho ~ Beta(5, 2)              # Mean ~ 0.7 (WM preference)
mu_capacity ~ TruncNorm(4, 1.5, low=1, high=7)
```

---

## 6. Model Comparison

### 6.1 Information Criteria

- **AIC** (Akaike Information Criterion) — used for MLE fitting
- **BIC** (Bayesian Information Criterion) — used for MLE fitting
- **WAIC** (Watanabe-Akaike Information Criterion) — used for Bayesian fitting
- **LOO** (Leave-One-Out Cross-Validation via PSIS) — used for Bayesian fitting

Lower values indicate better predictive performance.

### 6.1.1 Two Comparison Tracks

**Choice-only models (M1, M2, M3, M5, M6a, M6b):** Compared by AIC/BIC (MLE) or LOO/WAIC (Bayesian). These models share the same likelihood domain (choice probabilities only) and are directly comparable.

**Joint choice+RT model (M4):** Reported separately. M4's likelihood operates on a different domain (joint probability of choice AND response time), making its AIC values incommensurable with choice-only models. M4 is evaluated by its own parameter recovery and predictive checks, not by direct AIC comparison with M1-M6.

### 6.2 Running Comparison

```python
import arviz as az

# Load posteriors
qlearning_idata = az.from_netcdf('models/bayesian/qlearning_posterior.nc')
wmrl_idata = az.from_netcdf('models/bayesian/wmrl_posterior.nc')

# Compare models
comparison = az.compare({
    'qlearning': qlearning_idata,
    'wmrl': wmrl_idata
}, ic='loo')

print(comparison)
```

---

## 7. Parameter Recovery

Validate that the fitting procedure can recover known parameters:

```python
from simulations.generate_data import generate_dataset
from scripts.fitting.numpyro_models import qlearning_hierarchical_model

# Generate synthetic data with known parameters
true_params = {'alpha_pos': 0.3, 'alpha_neg': 0.1, 'epsilon': 0.05}
synthetic_data = generate_dataset(n_participants=50, model='qlearning', params=true_params)

# Fit model
mcmc = run_inference(qlearning_hierarchical_model, {'participant_data': synthetic_data})

# Compare recovered vs true
samples = mcmc.get_samples()
recovered_alpha_pos = samples['mu_alpha_pos'].mean()
print(f"True α₊: {true_params['alpha_pos']:.3f}, Recovered: {recovered_alpha_pos:.3f}")
```

---

## 8. Trauma-Specific Hypotheses

Based on the literature, we predict the following parameter differences in trauma-affected individuals:

| Parameter | Hypothesis | Mechanism |
|-----------|------------|-----------|
| α₊ | Decreased | Reduced learning from positive outcomes (anhedonia) |
| α₋ | Increased | Heightened sensitivity to negative outcomes (hypervigilance) |
| K (capacity) | Decreased | WM impairments associated with PTSD |
| ρ (WM reliance) | Decreased | Less confidence in WM, more reliance on slow RL |
| φ (decay) | Increased | Faster forgetting, consolidation deficits |
| ε (noise) | Increased | Attentional lapses, concentration difficulties |

---

## 9. Code Structure

```
scripts/fitting/
├── jax_likelihoods.py      # Pure JAX likelihood functions (choice-only models M1-M3, M5, M6a, M6b)
│   ├── softmax_policy()
│   ├── q_learning_block_likelihood()
│   ├── q_learning_multiblock_likelihood()
│   ├── wmrl_block_likelihood()
│   ├── wmrl_multiblock_likelihood()
│   ├── wmrl_m3_block_likelihood()
│   ├── wmrl_m5_block_likelihood()
│   ├── wmrl_m6a_block_likelihood()
│   └── wmrl_m6b_block_likelihood()
│
├── lba_likelihood.py       # LBA density functions for M4 joint choice+RT (float64)
│   ├── lba_pdf()
│   ├── lba_cdf()
│   └── lba_joint_log_lik()
│
├── mle_utils.py            # MLE utilities (transforms, bounds, info criteria)
│   ├── WMRL_M*_PARAMS      # Parameter name constants for all models
│   ├── WMRL_M*_BOUNDS      # Parameter bounds for all models
│   ├── transform_*()       # Bounded <-> unbounded transforms
│   └── compute_aic/bic()
│
├── fit_mle.py              # MLE fitting implementation (all 7 models)
├── numpyro_models.py       # Hierarchical Bayesian models
│   ├── qlearning_hierarchical_model()
│   ├── wmrl_hierarchical_model()
│   ├── prepare_data_for_numpyro()
│   └── run_inference()
│
├── fit_with_jax.py         # CLI for fitting single model
├── fit_both_models.py      # CLI for fitting both models
└── pymc_models.py          # Alternative PyMC implementation
```

---

## 10. Testing

Run likelihood tests:

```bash
python scripts/fitting/jax_likelihoods.py
```

Expected output:
```
JAX Q-LEARNING LIKELIHOOD TESTS
✓ Single block log-likelihood: -XX.XX
✓ JIT-compiled result matches: True

JAX WM-RL LIKELIHOOD TESTS
✓ WM-RL single block log-likelihood: -XX.XX
✓ JIT-compiled result matches: True

ALL TESTS PASSED!
```

---

## 11. Hierarchical Bayesian Pipeline (v4.0)

This section documents the hierarchical Bayesian inference infrastructure
built in Phases 13-17 of v4.0. It replaces the post-hoc FDR-corrected
regression approach with a single joint posterior that simultaneously
estimates individual parameters and trauma associations.

### 11.1 Non-Centered Parameterization

All bounded parameters use the hBayesDM non-centered convention
(Ahn et al., 2017) implemented in `scripts/fitting/numpyro_helpers.py`:

```
mu_pr     ~ Normal(mu_prior_loc, 1)
sigma_pr  ~ HalfNormal(0.2)
z_i       ~ Normal(0, 1)        for i = 1..N_participants
theta_i   = lower + (upper - lower) * Phi_approx(mu_pr + sigma_pr * z_i)
```

where `Phi_approx = jax.scipy.stats.norm.cdf` (probit link). This is
implemented by `sample_bounded_param()` in `numpyro_helpers.py`.

The prior location `mu_prior_loc` is parameter-specific (see
`PARAM_PRIOR_DEFAULTS` in `numpyro_helpers.py`):

| Parameter | mu_prior_loc | Bounds | Rationale |
|-----------|-------------|--------|-----------|
| alpha_pos, alpha_neg | 0.0 | [0, 1] | Centered on 0.5 on probability scale |
| phi | -0.8 | [0, 1] | Prior expectation of moderate forgetting |
| rho | 0.5 | [0, 1] | Prior toward higher WM reliability |
| capacity (K) | 0.0 | [2, 6] | Centered; see [K Parameterization](#k-parameterization) |
| kappa | -2.0 | [0, 1] | Prior toward low perseveration |
| epsilon | -2.0 | [0, 1] | Prior toward low noise |
| kappa_total | -2.0 | [0, 1] | Same as kappa |
| kappa_share | 0.0 | [0, 1] | No a priori preference for global vs stimulus |
| phi_rl | -0.8 | [0, 1] | Same as phi |
| kappa_s | -2.0 | [0, 1] | Same as kappa |

**K parameterization:** K uses bounds [2, 6] based on Collins (2012, 2014)
structural identifiability analysis. See [K Parameterization](#k-parameterization) (section 12)
for full rationale and literature review.

### 11.2 Level-2 Regression Structure

Trauma associations are estimated as Level-2 shifts on the unconstrained
(probit) scale. The 4-predictor design matrix is locked:

```
X = [lec_total, iesr_total, iesr_intr_resid, iesr_avd_resid]
```

where `iesr_intr_resid` and `iesr_avd_resid` are Gram-Schmidt
residualized against `iesr_total` (removes collinearity). The
hyperarousal residual is excluded because the three IES-R subscales
sum exactly to the total (rank deficiency). Condition number of the
design matrix: 11.3 (well below the 30 threshold).

The Level-2 shift modifies the non-centered parameterization:

```
shifted_mu_pr = mu_pr + sum(beta_k * X_k)
theta_i = lower + (upper - lower) * Phi_approx(shifted_mu_pr + sigma_pr * z_i)
```

Beta coefficients use Normal(0, 1) priors. The `build_level2_design_matrix`
function in `scripts/fitting/level2_design.py` is the single source of truth
for predictor construction. `COVARIATE_NAMES` list is authoritative.

**N=160 participants** have complete IES-R + LEC-5 data for Level-2.

### 11.3 NumPyro Factor Pattern and Pointwise Log-Likelihood

**Likelihood evaluation:** Each participant's contribution is computed by
the model-specific stacked likelihood function (e.g.,
`wmrl_m6b_multiblock_likelihood_stacked`) and attached to the model via
`numpyro.factor(f"obs_{participant_id}", -nll_i)`.

This pattern avoids numpyro.sample with observed data (which would require
a custom distribution class) while enabling correct posterior sampling.

**Pointwise log-likelihood for LOO/WAIC:** The `compute_pointwise_log_lik()`
function in `scripts/fitting/bayesian_diagnostics.py` re-evaluates the
likelihood post-hoc using `jax.vmap` over (chains, samples_per_chain) to
produce shape `(chains, samples, participants, n_blocks * max_trials)`.

Padded trial positions have `log_prob = 0.0` from the mask. These are
filtered by `filter_padding_from_loglik()` before constructing the
`log_likelihood` group in InferenceData.

### 11.4 WAIC/LOO Workflow

**Primary metric:** LOO-CV via `az.loo(idata, pointwise=True)` with
Pareto smoothed importance sampling.

**Secondary metric:** WAIC via `az.waic(idata)`.

**Model comparison:** `az.compare(compare_dict, ic='loo', method='stacking')`
produces stacking weights across the 6 choice-only models (M1, M2, M3, M5,
M6a, M6b).

**M4 separate track:** M4 (joint choice+RT via LBA) is NOT included in the
choice-only `az.compare` table because its likelihood domain differs. M4
gets its own LOO with Pareto-k gating:
- If < 5% observations have Pareto-k > 0.7: report M4 LOO ELPD separately.
- If >= 5%: fall back to choice-only marginal log-likelihood.
- See `scripts/06_fit_analyses/01_compare_models.py --bayesian-comparison` for implementation.

**Convergence gate:** All hierarchical fits must pass before LOO/WAIC:
`max_rhat < 1.01 AND min_ess_bulk > 400 AND num_divergences == 0`.

### 11.5 Schema-Parity CSV

`scripts/fitting/bayesian_summary_writer.py` writes individual-level
Bayesian summaries to `output/bayesian/{model}_individual_fits.csv` with
column names identical to the MLE CSVs plus:

- `{param}_hdi_low`, `{param}_hdi_high`, `{param}_sd` for each parameter
- `max_rhat`, `min_ess_bulk`, `num_divergences`
- `converged` (bool), `parameterization_version`

This schema parity enables `scripts/06_fit_analyses/04_analyze_mle_by_trauma.py`,
`scripts/06_fit_analyses/05_regress_parameters_on_scales.py`, and
`scripts/06_fit_analyses/06_analyze_winner_heterogeneity.py` to operate on either MLE or
Bayesian fits via a `--source mle|bayesian` flag with zero analysis-logic
changes.

---

## 12. K Parameterization

*Merged from docs/K_PARAMETERIZATION.md on 2026-04-22 (Phase 29 Plan 02).*

**Status:** v4.0 canonical reference (supersedes any K conventions used in v1.0–v3.0)
**Requirement:** K-01

### TL;DR

K is the working-memory capacity parameter, constrained to the continuous interval **[2, 6]**
via a non-centered normal CDF (Phi_approx) transform.  This matches the convention of the
project's reference paper: Senta, Bishop, Collins (2025) PLOS Comp Biol 21(9):e1012872, p. 20.
The canonical version string for all v4.0 fits using this convention is `"v4.0-K[2,6]-phiapprox"`.

### The WM Weight Formula

All Collins-lab RLWM papers use the same weight formula (Senta 2025 eq. 5; Collins 2014 eq. 1):

```
w(s, a) = rho * min(1, K / ns)
```

where `ns` is the number of stimulus-action pairs in the current block (the "set size"),
`rho` is the WM reliance parameter, and `K` is the capacity.

**Interpretation of K as a crossover point:**

- When `ns <= K`: `min(1, K/ns) = 1`; WM operates at full reliance (`rho`).
- When `ns > K`: `min(1, K/ns) = K/ns < 1`; WM contribution scales down proportionally.

K therefore marks the set-size threshold at which the WM system begins to
be capacity-limited.  A participant with K=3 shows full WM contribution in
2- and 3-item blocks but degraded WM in 5- and 6-item blocks.

This project's task uses set sizes `{2, 3, 5, 6}`.

### Non-Centered Hierarchical Transform

The non-centered parameterization follows the hBayesDM convention
(Ahn, Haines, Zhang 2017; Senta 2025 uses per-participant MLE, but the
transform structure below matches the hBayesDM template for NUTS compatibility):

```
# Group-level priors (unconstrained)
mu_K_pr    ~ Normal(0, 1)
sigma_K_pr ~ HalfNormal(0.2)

# Individual-level offsets
z_K_i      ~ Normal(0, 1)         # one per participant

# Individual capacity (constrained to [2, 6])
K_i = 2.0 + 4.0 * Phi_approx(mu_K_pr + sigma_K_pr * z_K_i)
```

`Phi_approx` is the standard normal CDF.  In NumPyro/JAX use:

```python
import jax.scipy.stats as jss
K_i = 2.0 + 4.0 * jss.norm.cdf(mu_K_pr + sigma_K_pr * z_K_i)
```

**Why Phi_approx (not sigmoid):**
The standard normal CDF gives a prior on the unconstrained scale that is Normal(0,1),
matching the group-mean prior exactly.  The sigmoid would give a logistic-shaped prior,
which is slightly heavier-tailed.  hBayesDM uses Phi_approx for this reason throughout
its Stan RLWM models, and the NumPyro port follows suit.

**Prior implied on K_i:**
`mu_K_pr = 0, sigma_K_pr = 0` gives `K_i = 2.0 + 4.0 * 0.5 = 4.0` (midpoint of [2, 6]).
The HalfNormal(0.2) prior on `sigma_K_pr` strongly regularizes individual variation,
consistent with a population where most participants are near K=3–5.

### `parameterization_version`

All Bayesian output CSVs written by v4.0 must include a `parameterization_version`
column with the value:

```
v4.0-K[2,6]-phiapprox
```

Downstream scripts (`scripts/06_fit_analyses/04_analyze_mle_by_trauma.py`, `scripts/06_fit_analyses/05_regress_parameters_on_scales.py`)
validate this column on load and raise a `ValueError` if the string does not match,
preventing accidental mixing of v3.0 MLE fits (which use K in [1, 7]) with v4.0 fits.

### Historical Collins-Lab K Conventions

| Paper | Year | K Type | Bounds | Fitting Method | Set Sizes | Notes |
|-------|------|--------|--------|----------------|-----------|-------|
| Collins & Frank (Eur J Neurosci 35:1024) | 2012 | Discrete integer | {0, 1, 2, 3, 4, 5, 6} | MLE fmincon, iterated over all K values | 2, 3, 4, 5, 6 | Founding RLWM paper; established `w = rho * min(1, K/ns)` |
| Collins, Brown, Gold, Waltz, Frank (J Neurosci 34:13747) | 2014 | Discrete integer | {0, 1, 2, 3, 4, 5, 6} | Iterated fmincon with 50 random starts per K | 2, 3, 4, 5, 6 | PMC4188972; patients median K=2, controls median K=3 |
| McDougle & Collins (Psychon Bull Rev 28:1205) | 2021 | Continuous | [2, 5] | MLE fmincon, 40 iterations | Instrumental (not RLWM) | PMC7854965; first Collins-lab continuous K; used symbol C |
| Senta, Bishop, Collins (PLOS Comp Biol 21:e1012872) | 2025 | Continuous | **[2, 6]** | MLE fmincon, 20 random starts | 2, 3, 4, 5, 6 | **Project reference paper**; K constrained to [2,6] per p. 20 |

### Why Lower Bound = 2

**1. Senta 2025 convention.**
The project's reference paper explicitly constrains K to [2, 6] (Senta 2025, p. 20).
Matching this bound aligns the project with the most recent Collins-lab canonical standard.

**2. Scientific interpretation of K < 2.**
The smallest set size in this task is `ns = 2`.  At K=2, WM weight in a 2-item block is:

```
rho * min(1, 2/2) = rho    # full WM reliance at ns=2
```

At K=1 (below the lower bound), WM weight in the same 2-item block is:

```
rho * min(1, 1/2) = 0.5 * rho    # half WM reliance at ns=2
```

This 0.5 scaling factor is fully confounded with `rho` itself — any reduction in K below 2
is geometrically absorbed by `rho`, making K < 2 non-identifiable.  The lower bound of 2
is therefore not merely a convention but a structural identifiability requirement given this
task's minimum set size.

**3. Breaking change acknowledgment.**
The v3.0 MLE pipeline used K in [1, 7].  This IS a breaking change.
Phase 14 (requirements K-02, K-03) refits all models with the new [2, 6] bounds so that
the v4.0 MLE and Bayesian pipelines share the same convention.  The
`parameterization_version` column enforces this at runtime.

### Why Upper Bound = 6

The task's maximum set size is `ns = 6`.  For any K > 6:

```
min(1, K/ns) = min(1, K/6) = 1    for all ns in {2, 3, 5, 6}
```

K above 6 is therefore structurally indistinguishable from K = 6 — it adds no degrees of
freedom to the likelihood.  Capping at 6 removes this non-identified region of parameter
space and keeps the support finite.  Senta 2025 uses the same upper bound for the same reason.

### BIC Rejection Rationale

Senta 2025 (p. 22, verbatim): "Previous research has shown that Bayesian model selection
criteria such as the Bayesian Information Criteria (BIC) tend to over-penalize models in the
RLWM class [Collins & Frank 2018].  To confirm this in the current data and support our use
of AIC as a measure of model fit, we performed a parallel model recovery analysis for the
selected RLWM models using BIC.  The confusion matrix for this analysis... confirms that
data generated from more complex underlying processes tends to be (incorrectly) best-fit by
simpler models when BIC is used."

**v4.0 policy:** BIC is retained in all output CSVs for v3.0 MLE back-compatibility.
It is NOT used as a model-selection criterion.  The primary comparison criterion in Phase 18
(requirement CMP-03) is WAIC/LOO from posterior predictive evaluation.

### References

Senta JD, Bishop SJ, Collins AGE (2025).
Dual process impairments in reinforcement learning and working memory systems underlie
learning deficits in physiological anxiety.
*PLoS Computational Biology* 21(9): e1012872.
DOI: 10.1371/journal.pcbi.1012872. Data: https://osf.io/w8ch2/

Collins AGE, Brown JK, Gold JM, Waltz JA, Frank MJ (2014).
Working memory contributions to reinforcement learning impairments in schizophrenia.
*Journal of Neuroscience* 34(41): 13747–56.
PMC4188972.

McDougle SD, Collins AGE (2021).
Modeling the influence of working memory, reinforcement learning, and action uncertainty
on reaction time and choice during instrumental learning.
*Psychonomic Bulletin & Review* 28(4): 1205–18.
PMC7854965.

Collins AGE, Frank MJ (2012).
How much of reinforcement learning is working memory, not reinforcement learning?
A behavioral, computational, and neuroimaging analysis.
*European Journal of Neuroscience* 35(7): 1024–35.

*Historical source: see [../legacy/K_PARAMETERIZATION.md](../legacy/K_PARAMETERIZATION.md) for the original standalone version.*

---

## 13. See Also

- **Task/Environment**: `docs/TASK_AND_ENVIRONMENT.md`
- **Configuration**: `config.py`
- **Agent Classes**: `src/rlwm/models/q_learning.py`, `src/rlwm/models/wm_rl_hybrid.py`
- **K Bounds Rationale**: `docs/03_methods_reference/MODEL_REFERENCE.md#k-parameterization` (this document, section 12)
