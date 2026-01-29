# Phase 1: Core Implementation - Research

**Researched:** 2026-01-29
**Domain:** JAX-based likelihood functions and agent extension for perseveration parameter
**Confidence:** HIGH

## Summary

Phase 1 extends the existing WM-RL hybrid model (M2) with a perseveration parameter (κ) to create M3. The implementation involves:
1. Adding κ·Rep(a) term to JAX likelihood functions in `scripts/fitting/jax_likelihoods.py`
2. Extending `WMRLHybridAgent` class in `models/wm_rl_hybrid.py` with optional kappa parameter
3. Tracking global action repetition with block-boundary resets

The existing codebase provides a strong foundation: JAX likelihood functions with block-aware processing, agent architecture with hybrid value computation, and established parameter transformation patterns. The perseveration parameter captures outcome-insensitive action repetition (motor perseveration) separate from reduced negative learning rate (outcome insensitivity).

Recent computational psychiatry research confirms that perseveration parameters are standard practice for dissociating action stickiness from value-based learning, and failure to include perseveration can bias learning rate estimates.

**Primary recommendation:** Extend existing functions rather than duplicate code. Follow established patterns for block-aware processing and parameter handling.

## Standard Stack

### Core (Already in Place)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| JAX | Latest | JIT-compiled likelihoods | Fast gradients, functional programming, established in project |
| NumPy | Latest | Array operations | Universal standard for numerical computing |
| SciPy | Latest | Optimization (L-BFGS-B) | Standard for MLE fitting with bounded parameters |

### Supporting (Already in Place)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| jax.lax.scan | - | Sequential operations | Processing trial sequences functionally |
| functools.partial | - | Parameter binding | Creating closures for objective functions |

**Installation:** Already installed (no new dependencies)

## Architecture Patterns

### Recommended Implementation Structure

Phase 1 modifies existing files only:
```
scripts/fitting/
├── jax_likelihoods.py    # ADD: wmrl_m3_block_likelihood()
│                         #      wmrl_m3_multiblock_likelihood()
models/
├── wm_rl_hybrid.py       # MODIFY: WMRLHybridAgent.__init__() add kappa parameter
                          #         WMRLHybridAgent.get_hybrid_probs() include κ·Rep(a)
                          #         Add self.last_action tracking
```

### Pattern 1: Action Repetition Tracking

**What:** Track previous action to compute Rep_t(a) = I[a = a_{t-1}]

**When to use:** Every trial in likelihood function; reset at block boundaries

**Example from research:**
```python
# In JAX likelihood (functional, stateless)
def step_fn(carry, inputs):
    Q_table, WM_table, WM_baseline, log_lik_accum, last_action = carry
    stimulus, action, reward, set_size = inputs

    # Compute Rep(a) indicator: 1 if action == last_action, else 0
    rep_indicator = jnp.where(
        last_action >= 0,  # Valid last action exists
        jnp.where(action == last_action, 1.0, 0.0),
        0.0  # First trial or post-reset
    )

    # Add κ·Rep(a) to hybrid values BEFORE softmax
    hybrid_vals = omega * wm_vals + (1 - omega) * q_vals
    hybrid_vals_with_persev = hybrid_vals.at[action].add(kappa * rep_indicator)

    # Softmax and compute log probability
    probs = softmax_policy(hybrid_vals_with_persev, FIXED_BETA)

    # Update last_action for next trial
    new_carry = (Q_updated, WM_updated, WM_baseline, log_lik_new, action)
    return new_carry, log_prob

# Initial carry includes last_action = -1 (invalid)
init_carry = (Q_init, WM_init, WM_0, 0.0, -1)
```

**Key insight:** Use -1 sentinel for "no previous action" (first trial, block boundary). JAX functional style requires passing last_action through carry.

### Pattern 2: Block-Aware Processing with Reset

**What:** Reset last_action at start of each block (no carry-over between blocks)

**When to use:** Multi-block wrapper function

**Example:**
```python
def wmrl_m3_multiblock_likelihood(...):
    total_log_lik = 0.0

    for block_idx, (stim, act, rew, sets) in enumerate(zip(...)):
        # Each block starts fresh - last_action resets to -1 inside block_likelihood
        block_log_lik = wmrl_m3_block_likelihood(
            stimuli=stim,
            actions=act,
            rewards=rew,
            set_sizes=sets,
            # ... parameters including kappa
        )
        total_log_lik += block_log_lik

    return total_log_lik
```

**Key insight:** Block independence mirrors existing Q/WM reset pattern. Each block initializes its own last_action = -1.

### Pattern 3: Backward Compatibility via Default Parameter

**What:** kappa=0 parameter makes M3 identical to M2

**When to use:** Agent initialization, testing, optional parameter handling

**Example:**
```python
class WMRLHybridAgent:
    def __init__(
        self,
        # ... existing parameters
        kappa: float = 0.0,  # Default 0 = M2 behavior
    ):
        self.kappa = kappa
        self.last_action = None  # Track for perseveration

    def get_hybrid_probs(self, stimulus: int, set_size: int) -> Dict[str, Any]:
        # Compute base hybrid values
        hybrid_probs = omega * wm_probs + (1 - omega) * rl_probs

        # Add perseveration if kappa > 0 and last_action exists
        if self.kappa > 0 and self.last_action is not None:
            # Convert probs to values (inverse softmax)
            # Add κ to last_action's value
            # Re-apply softmax
            # (Detailed implementation in agent class)

        return {'probs': hybrid_probs, ...}

    def update(self, stimulus, action, reward, next_stimulus):
        # ... existing updates
        self.last_action = action  # Track for next trial

    def reset(self):
        # ... existing resets
        self.last_action = None  # Reset at block boundary
```

**Key insight:** Optional parameter with sensible default enables gradual adoption and testing.

### Pattern 4: Additive Term in Softmax

**What:** κ·Rep(a) adds to value BEFORE softmax, not after

**When to use:** Computing action probabilities with perseveration

**Correct:**
```python
# Add to values (log-space equivalent)
V_persev = V_hybrid + kappa * rep_indicator
probs = softmax(beta * V_persev)
```

**Incorrect:**
```python
# DON'T add to probabilities directly
probs_hybrid = softmax(beta * V_hybrid)
probs_persev = probs_hybrid + kappa * rep_indicator  # WRONG
```

**Key insight:** Perseveration operates at the value level, not probability level. This ensures proper normalization and interaction with beta parameter.

### Anti-Patterns to Avoid

- **Stimulus-specific perseveration:** Track global action (motor level), not stimulus-action pairs. Research shows motor perseveration is the relevant construct post-reversal.
- **Carry-over between blocks:** Do NOT pass last_action from block N to block N+1. Blocks are independent (Q/WM reset).
- **Separate beta for perseveration:** Use same β=50 for all terms. Adding another temperature parameter hurts identifiability.
- **Modifying probabilities post-softmax:** Add κ·Rep(a) to values before softmax, not to probabilities after.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Sequential updates | Custom for-loop over trials | jax.lax.scan() | JIT compilation requires functional style; scan is JAX-native |
| Parameter transformations | Custom logit/bounds | Existing mle_utils functions | Already handles bounded→unconstrained→bounded roundtrip |
| Softmax with numerical stability | np.exp(x) / sum | Existing softmax_policy() helper | Max-subtraction prevents overflow |
| Block data preparation | Custom parsing | Existing prepare_block_data() | Already handles participant→block→trial structure |

**Key insight:** Existing codebase has solved all infrastructure problems. Phase 1 focuses purely on model extension logic.

## Common Pitfalls

### Pitfall 1: Forgetting First Trial

**What goes wrong:** First trial of each block has no previous action, causing NaN or incorrect Rep(a)

**Why it happens:** Forgetting edge case when t=0

**How to avoid:** Use sentinel value (last_action = -1 for "invalid") and conditional logic

**Warning signs:** NaN in likelihoods, incorrect log-likelihood on first trial of each block

**Solution:**
```python
# In carry initialization
init_carry = (Q_init, WM_init, WM_0, 0.0, -1)  # last_action = -1

# In step function
rep_indicator = jnp.where(
    last_action >= 0,  # Check if valid action exists
    jnp.where(action == last_action, 1.0, 0.0),
    0.0  # No previous action = no perseveration bonus
)
```

### Pitfall 2: Block Boundary Contamination

**What goes wrong:** Last action from block N leaks into block N+1, artificially inflating perseveration

**Why it happens:** Forgetting to reset last_action between blocks

**How to avoid:** Each call to wmrl_m3_block_likelihood() initializes last_action = -1

**Warning signs:** Unrealistically high κ estimates, first trial of each block shows spurious perseveration

**Solution:** Block-level function always starts with fresh last_action = -1 in init_carry

### Pitfall 3: Parameter Ordering Mismatch

**What goes wrong:** Unconstrained parameter array has wrong order, causing α to be interpreted as κ

**Why it happens:** WMRL_M3_PARAMS list doesn't match parameter extraction order

**How to avoid:** Define canonical order early, document clearly, test round-trip transformation

**Warning signs:** Nonsensical parameter values, optimization converging to weird boundaries

**Solution:** Use consistent ordering everywhere:
```python
WMRL_M3_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'epsilon', 'kappa']
# Always extract/insert in this exact order
```

### Pitfall 4: Agent vs Likelihood Inconsistency

**What goes wrong:** Agent class implements perseveration differently than likelihood function

**Why it happens:** Implementing same logic twice without coordination

**How to avoid:**
- Implement likelihood first (source of truth for fitting)
- Test likelihood thoroughly
- Match agent implementation to likelihood exactly
- Use same Rep(a) computation, same κ bounds, same reset logic

**Warning signs:** Agent simulations don't match likelihood predictions, parameter recovery fails

**Solution:** Write unit test that compares agent-generated data likelihood against agent's own parameters

### Pitfall 5: Ignoring Beta Scaling

**What goes wrong:** Treating κ and β as independent, causing non-identifiability

**Why it happens:** Forgetting that both scale the softmax input

**How to avoid:** Remember β is FIXED at 50. κ is in same units as value function. Don't try to estimate both.

**Warning signs:** κ parameter hitting bounds, strange correlation with other parameters

**Solution:** Document clearly that β=50 applies to entire argument: β·(V + κ·Rep)

## Code Examples

Verified patterns from existing codebase:

### Block-Aware Likelihood Function Pattern
```python
# Source: scripts/fitting/jax_likelihoods.py (lines 521-663)
# Pattern: Single block processing with lax.scan

def wmrl_block_likelihood(
    stimuli: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    set_sizes: jnp.ndarray,
    alpha_pos: float,
    alpha_neg: float,
    phi: float,
    rho: float,
    capacity: float,
    epsilon: float = DEFAULT_EPSILON,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0
) -> float:
    # Initialize Q-table and WM matrix
    Q_init = jnp.ones((num_stimuli, num_actions)) * q_init
    WM_init = jnp.ones((num_stimuli, num_actions)) * wm_init
    WM_0 = jnp.ones((num_stimuli, num_actions)) * wm_init

    # Initial carry: (Q, WM, WM_0, log_likelihood)
    init_carry = (Q_init, WM_init, WM_0, 0.0)

    # Prepare inputs
    scan_inputs = (stimuli, actions, rewards, set_sizes)

    def step_fn(carry, inputs):
        Q_table, WM_table, WM_baseline, log_lik_accum = carry
        stimulus, action, reward, set_size = inputs

        # 1. Decay WM
        WM_decayed = (1 - phi) * WM_table + phi * WM_baseline

        # 2. Compute hybrid policy
        q_vals = Q_table[stimulus]
        rl_probs = softmax_policy(q_vals, FIXED_BETA)

        wm_vals = WM_decayed[stimulus]
        wm_probs = softmax_policy(wm_vals, FIXED_BETA)

        omega = rho * jnp.minimum(1.0, capacity / set_size)
        hybrid_probs = omega * wm_probs + (1 - omega) * rl_probs
        hybrid_probs = hybrid_probs / jnp.sum(hybrid_probs)

        # 3. Apply epsilon noise
        noisy_probs = apply_epsilon_noise(hybrid_probs, epsilon, num_actions)
        log_prob = jnp.log(noisy_probs[action] + 1e-8)

        # 4. Update WM
        WM_updated = WM_decayed.at[stimulus, action].set(reward)

        # 5. Update Q-table
        q_current = Q_table[stimulus, action]
        delta = reward - q_current
        alpha = jnp.where(delta > 0, alpha_pos, alpha_neg)
        q_updated = q_current + alpha * delta
        Q_updated = Q_table.at[stimulus, action].set(q_updated)

        log_lik_new = log_lik_accum + log_prob

        return (Q_updated, WM_updated, WM_baseline, log_lik_new), log_prob

    # Run scan over trials
    (Q_final, WM_final, _, log_lik_total), log_probs = lax.scan(step_fn, init_carry, scan_inputs)

    return log_lik_total
```

**Adaptation for M3:** Add last_action to carry, compute rep_indicator, add κ·Rep(a) to hybrid values before softmax.

### Multi-Block Wrapper Pattern
```python
# Source: scripts/fitting/jax_likelihoods.py (lines 666-770)
# Pattern: Independent block processing with summation

def wmrl_multiblock_likelihood(
    stimuli_blocks: list,
    actions_blocks: list,
    rewards_blocks: list,
    set_sizes_blocks: list,
    alpha_pos: float,
    alpha_neg: float,
    phi: float,
    rho: float,
    capacity: float,
    epsilon: float = DEFAULT_EPSILON,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
    wm_init: float = 1.0 / 3.0,
    verbose: bool = False,
    participant_id: str = None
) -> float:
    total_log_lik = 0.0
    num_blocks = len(stimuli_blocks)

    for block_idx, (stim_block, act_block, rew_block, set_block) in enumerate(
        zip(stimuli_blocks, actions_blocks, rewards_blocks, set_sizes_blocks)
    ):
        block_log_lik = wmrl_block_likelihood(
            stimuli=stim_block,
            actions=act_block,
            rewards=rew_block,
            set_sizes=set_block,
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            phi=phi,
            rho=rho,
            capacity=capacity,
            epsilon=epsilon,
            num_stimuli=num_stimuli,
            num_actions=num_actions,
            q_init=q_init,
            wm_init=wm_init
        )
        total_log_lik += block_log_lik

    return total_log_lik
```

**Adaptation for M3:** Add kappa parameter, pass to block likelihood. Block independence is automatic.

### Agent Extension Pattern
```python
# Source: models/wm_rl_hybrid.py (lines 59-187)
# Pattern: Optional parameter with default value

class WMRLHybridAgent:
    def __init__(
        self,
        num_stimuli: int = TaskParams.MAX_STIMULI,
        num_actions: int = TaskParams.NUM_ACTIONS,
        alpha_pos: float = ModelParams.ALPHA_POS_DEFAULT,
        alpha_neg: float = ModelParams.ALPHA_NEG_DEFAULT,
        beta: float = ModelParams.BETA_DEFAULT,
        beta_wm: float = ModelParams.BETA_WM_DEFAULT,
        gamma: float = 0.0,
        capacity: int = ModelParams.WM_CAPACITY_DEFAULT,
        phi: float = ModelParams.PHI_DEFAULT,
        rho: float = ModelParams.RHO_DEFAULT,
        q_init: float = ModelParams.Q_INIT_VALUE,
        wm_init: float = ModelParams.WM_INIT_VALUE,
        seed: Optional[int] = None,
    ):
        # Store all parameters
        self.alpha_pos = alpha_pos
        # ... etc

        # Initialize state
        self.Q = np.full((num_stimuli, num_actions), q_init, dtype=np.float64)
        self.WM = np.full((num_stimuli, num_actions), wm_init, dtype=np.float64)

    def reset(self, q_init: Optional[float] = None, wm_init: Optional[float] = None):
        """Reset the agent to initial state."""
        # Reset Q-table and WM matrix
        self.Q.fill(self.q_init if q_init is None else q_init)
        self.WM.fill(self.wm_init if wm_init is None else wm_init)
```

**Adaptation for M3:** Add kappa parameter, add self.last_action tracking, reset last_action in reset().

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Single perseveration term | Multiple types (motor vs stimulus) | ~2024 research | Distinguish motor stickiness from stimulus preference |
| Ignoring perseveration in RL models | Including perseveration parameter | ~2023-2025 | Reduces bias in learning rate estimates |
| Same β for all value terms | Separate β for perseveration | Some models | Increases complexity; fixed β=50 is standard in this project |

**Deprecated/outdated:**
- PyTensor-based likelihoods: Project switched to JAX in 2026-01-20
- Estimation of beta parameter: Fixed at 50 following Senta et al. (2025)

**Current best practice (2025-2026):**
- Include perseveration parameter to avoid biasing learning rate estimates
- Use additive term in softmax argument: P(a) ∝ exp(V(a) + κ·Rep(a))
- Reset perseveration at natural task boundaries (blocks)
- Global action repetition (motor level) for reversal learning tasks

## Open Questions

1. **Optimal κ bounds**
   - What we know: Senta et al. use [0.001, 0.999] for rate parameters
   - What's unclear: Should κ use same bounds, or [0, 1] without margin, or different scale?
   - Recommendation: Use [0.001, 0.999] to match other parameters; test sensitivity

2. **Numerical stability with large κ**
   - What we know: κ·Rep(a) adds to hybrid value before softmax with β=50
   - What's unclear: If κ approaches 1 and β=50, can cause large exp() arguments
   - Recommendation: Test with κ=0.99, ensure softmax_policy() max-subtraction handles it

3. **Agent implementation order**
   - What we know: Need to modify get_hybrid_probs() to include perseveration
   - What's unclear: Add κ·Rep before or after computing omega-weighted hybrid?
   - Recommendation: Add after hybrid combination: V_final = V_hybrid + κ·Rep(a), then softmax

## Sources

### Primary (HIGH confidence)
- Existing codebase: scripts/fitting/jax_likelihoods.py - WM-RL likelihood implementation
- Existing codebase: models/wm_rl_hybrid.py - Agent architecture and hybrid value computation
- Existing codebase: scripts/fitting/mle_utils.py - Parameter transformation patterns
- Existing codebase: config.py - Model parameter defaults and bounds
- [Signatures of Perseveration in Two-Step Sequential Decision Task](https://cpsyjournal.org/articles/10.5334/cpsy.101) - Recent 2025 study on perseveration in RL models
- [Choice-confirmation bias and gradual perseveration in human reinforcement learning](https://pubmed.ncbi.nlm.nih.gov/36395020/) - Meta-analysis showing perseveration is robust feature
- [Dissociation between asymmetric value updating and perseverance](https://www.nature.com/articles/s41598-020-80593-7) - Shows failure to include perseveration biases learning rate estimates

### Secondary (MEDIUM confidence)
- [Evolving choice hysteresis in reinforcement learning](https://www.pnas.org/doi/10.1073/pnas.2422144122) - 2025 evolutionary study on perseveration emergence
- [Active reinforcement learning versus action bias and hysteresis](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011950) - Action repetition bias patterns

### Tertiary (LOW confidence)
- None - all key claims verified with primary sources

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - JAX/NumPy/SciPy already in use, no new dependencies
- Architecture: HIGH - Existing patterns directly apply, minimal adaptation needed
- Pitfalls: HIGH - Based on careful reading of existing code and recent literature

**Research date:** 2026-01-29
**Valid until:** ~60 days (stable technical domain, core JAX API unlikely to change)

**Key findings:**
1. Perseveration parameter is standard practice in computational psychiatry (2023-2025 research)
2. Existing codebase provides all necessary infrastructure (JAX, block-aware processing, parameter handling)
3. Implementation requires minimal changes: add carry element, compute Rep(a), add κ·Rep(a) before softmax
4. Block independence and reset logic match existing Q/WM reset pattern
5. Backward compatibility via kappa=0 default ensures M2 behavior preserved

**Critical for planning:**
- No new dependencies required
- Extend existing functions, don't duplicate
- Follow established block-aware processing pattern
- Test backward compatibility (κ=0 → M2 results)
- Agent implementation should match likelihood exactly
