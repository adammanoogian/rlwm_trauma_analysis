# Phase 1 Plan Verification

**Phase:** 01-core-implementation
**Date:** 2026-01-29
**Plans Verified:** 2 (01-01-PLAN.md, 01-02-PLAN.md)

## Summary

**PASS** - Both plans comprehensively address all Phase 1 requirements. The plans are well-structured with clear task definitions, proper verification steps, and backward compatibility testing. Minor concerns exist around exact numerical equivalence for backward compatibility, but these are addressed with appropriate test cases.

## Plan Analysis

### 01-01-PLAN.md (JAX Likelihood Functions)

**Requirements covered:** LIK-01, LIK-02, LIK-03, LIK-04, LIK-05

**Completeness:** HIGH

| Aspect | Assessment |
|--------|------------|
| Task structure | 3 tasks, all have files/action/verify/done |
| Files modified | 1 file (jax_likelihoods.py) - appropriate scope |
| Dependencies | None (wave 1) - correct |
| must_haves | Well-defined truths, artifacts, key_links |

**Task breakdown:**
1. Task 1: wmrl_m3_block_likelihood() - Implements kappa perseveration with Rep(a) indicator
2. Task 2: wmrl_m3_multiblock_likelihood() - Wrapper summing across blocks
3. Task 3: Tests and backward compatibility verification

**Strengths:**
- Detailed code examples showing exact implementation
- Correct use of sentinel value (-1) for first trial handling
- Proper placement of kappa*Rep(a) in value space before softmax
- Explicit backward compatibility test (kappa=0 == M2)
- Uses jnp.eye(num_actions)[last_action] for one-hot encoding - efficient JAX pattern

**Concerns:**
1. **Minor:** Task 1 action shows rep_indicators computed for ALL actions but the key_link pattern suggests checking after softmax. The implementation is actually correct (compute rep for all actions, add kappa only to matching one before softmax), but the description could be clearer.

2. **Minor:** The plan mentions add kappa*Rep(a) to hybrid VALUES (not probs) before final softmax which is correct for M3, but the existing M2 wmrl_block_likelihood() computes hybrid_probs = omega * wm_probs + (1-omega) * rl_probs (probability space, not value space). The M3 implementation uses value-space computation which is mathematically different. This is acceptable because:
   - When kappa=0, the value-space vs prob-space difference is tested by the backward compatibility test
   - The test will catch if they do not match numerically

### 01-02-PLAN.md (Agent Extension)

**Requirements covered:** AGT-01

**Completeness:** HIGH

| Aspect | Assessment |
|--------|------------|
| Task structure | 6 tasks, all have files/action/verify/done |
| Files modified | 1 file (wm_rl_hybrid.py) - appropriate scope |
| Dependencies | None (wave 1) - correct |
| must_haves | Well-defined truths, artifacts, key_links |

**Task breakdown:**
1. Task 1: Add kappa parameter and last_action tracking to __init__
2. Task 2: Update reset() to clear last_action
3. Task 3: Update get_hybrid_probs() with perseveration logic
4. Task 4: Update update() to track last_action
5. Task 5: Update parameter getter/setter and factory function
6. Task 6: Add M3 test to verify perseveration behavior

**Strengths:**
- Comprehensive coverage of all agent methods that need modification
- Explicit handling of last_action = None for block boundaries
- Conditional path for backward compatibility (kappa=0 uses original M2 behavior)
- Test verifies: perseveration effect, reset clears last_action, M3 without last_action matches M2
- Factory function updated for easy agent creation

**Concerns:**
1. **Minor:** Task 3 shows two different implementation approaches - one working with values directly, one with conditional path for M2 backward compatibility. The plan notes If backward compatibility is critical suggesting the conditional path. This is the safer approach and should be used.

2. **Minor:** 6 tasks is at the upper threshold (guideline suggests 2-3 tasks per plan). However, the tasks are simple modifications to different methods, and keeping them in one plan maintains logical cohesion for the agent class.

## Success Criteria Check

| # | Criterion | Plan | Status | Notes |
|---|-----------|------|--------|-------|
| SC1 | wmrl_m3_block_likelihood() computes log-likelihood with kappa*Rep(a) | 01-01, Task 1 | COVERED | Explicit code showing kappa term in hybrid_vals |
| SC2 | Rep(a) = I[a = a_t-1] tracks global action repetition | 01-01, Task 1 | COVERED | jnp.eye(num_actions)[last_action] creates one-hot for last action |
| SC3 | Last action resets at block start | 01-01, Task 1 | COVERED | init_carry = (..., -1) sentinel value |
| SC4 | wmrl_m3_multiblock_likelihood() sums across blocks | 01-01, Task 2 | COVERED | Loop calling block likelihood, fresh last_action each block |
| SC5 | WMRLHybridAgent extended with optional kappa (default 0) | 01-02, Tasks 1,5 | COVERED | kappa: float = 0.0 in __init__ and factory |
| SC6 | get_hybrid_probs() includes kappa*Rep(a) when kappa > 0 | 01-02, Task 3 | COVERED | Conditional perseveration bonus added |

## Requirement Coverage Matrix

| Requirement | Description | Plans | Tasks | Status |
|-------------|-------------|-------|-------|--------|
| LIK-01 | M3 block likelihood with kappa | 01-01 | 1 | COVERED |
| LIK-02 | Global Rep(a) tracking | 01-01 | 1 | COVERED |
| LIK-03 | Block boundary reset | 01-01 | 1,2 | COVERED |
| LIK-04 | Multi-block wrapper | 01-01 | 2 | COVERED |
| LIK-05 | kappa=0 backward compatibility | 01-01 | 3 | COVERED |
| AGT-01 | Agent with optional kappa | 01-02 | 1-6 | COVERED |

## Dependency Analysis

Plan 01-01-PLAN.md (Wave 1) depends_on: []

Plan 01-02-PLAN.md (Wave 1) depends_on: []

**Analysis:** Both plans are in Wave 1 with no dependencies. This is correct - they modify independent files (jax_likelihoods.py vs wm_rl_hybrid.py) and can execute in parallel.

## Key Links Verification

### Plan 01-01 Key Links
| From | To | Via | Planned? |
|------|----|-----|----------|
| wmrl_m3_block_likelihood | softmax_policy | add kappa*rep to hybrid_vals | YES - Task 1 action shows hybrid_vals_persev = hybrid_vals + kappa * rep_indicators then softmax_policy(hybrid_vals_persev, FIXED_BETA) |
| wmrl_m3_multiblock_likelihood | wmrl_m3_block_likelihood | loop with fresh last_action | YES - Task 2 shows loop calling block likelihood |

### Plan 01-02 Key Links
| From | To | Via | Planned? |
|------|----|-----|----------|
| WMRLHybridAgent.get_hybrid_probs | perseveration logic | adds kappa bonus when last_action exists | YES - Task 3 shows conditional if self.kappa > 0 and self.last_action is not None |
| WMRLHybridAgent.reset | last_action | clears for block boundary | YES - Task 2 shows self.last_action = None |

## Scope Assessment

| Plan | Tasks | Files Modified | Assessment |
|------|-------|----------------|------------|
| 01-01 | 3 | 1 | GOOD - Within guidelines |
| 01-02 | 6 | 1 | WARNING - At threshold, but tasks are simple method modifications |

**Total estimated scope:** 9 tasks across 2 files. Manageable for Phase 1.

## Gaps or Concerns

### Minor Issues (should address but not blocking)

1. **Value-space vs probability-space computation:** The M3 likelihood uses value-space combination (hybrid_vals = omega * wm_vals + (1-omega) * q_vals) while M2 uses probability-space (hybrid_probs = omega * wm_probs + (1-omega) * rl_probs). The backward compatibility test should catch any numerical differences, but developers should be aware this is a subtle change.

2. **Agent task count:** Plan 01-02 has 6 tasks which is at the upper limit. Consider that Tasks 1, 2, and 4 could be combined into one Add kappa tracking infrastructure task. However, keeping them separate provides clearer verification points.

3. **Rep(a) documentation:** Both plans correctly implement global Rep(a), but neither explicitly documents WHY it is global (motor-level perseveration vs stimulus-specific). The RESEARCH.md covers this but it should be reflected in code comments.

### No Blockers Found

All critical requirements are covered. Implementation details are well-specified. Verification steps will catch integration issues.

## Verdict

**PASS**

Both plans are ready for execution. They:
- Cover all Phase 1 requirements (LIK-01 through LIK-05, AGT-01)
- Meet all success criteria (SC1-SC6)
- Have complete task definitions with verify/done criteria
- Include backward compatibility tests
- Follow existing codebase patterns (JAX lax.scan, agent method structure)
- Have no dependency conflicts

**Recommendation:** Execute plans in Wave 1 (parallel execution possible since they modify independent files). The backward compatibility tests in both plans will validate that kappa=0 produces M2-equivalent behavior.
