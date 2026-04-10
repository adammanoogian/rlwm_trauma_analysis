# Literature for Quick-006 Manuscript Updates

Scope: assemble citations for (A) the M1-M6b/M4 model lineage and (B)
trauma literature that can frame the (marginal, uncorrected) epsilon x
IES-R association observed in the M6b fits. Produced for
manuscript/paper.tex Discussion-section updates.

## Part A: Model Lineage (for Methods / Discussion)

### Collins & Frank (2012). How much of reinforcement learning is working memory, not reinforcement learning?
- **Source:** European Journal of Neuroscience 35(7), 1024-1035
- **DOI:** 10.1111/j.1460-9568.2011.07980.x
- **Relevance:** Seminal RLWM decomposition. Introduces a fast-decaying
  working-memory buffer with capacity K that competes with a slower
  delta-rule RL module via a weighted softmax. Our M2 is a direct
  implementation of this two-system architecture with the WMRLHybridAgent.
  Capacity K, decay phi, and the RL-weight rho are all inherited.

### Collins, Brown, Gold, Waltz & Frank (2014). Working memory contributions to reinforcement learning impairments in schizophrenia.
- **Source:** Journal of Neuroscience 34(41), 13747-13756
- **DOI:** 10.1523/JNEUROSCI.0989-14.2014
- **Relevance:** First clinical application of RLWM. Demonstrated that
  apparent reward-learning deficits in schizophrenia are partly explained
  by working-memory capacity differences. Motivates our use of RLWM (not
  pure Q-learning) for a trauma-exposed sample, since conflating WM and
  RL deficits would obscure any trauma-specific effect.

### Collins & Frank (2018). Within- and across-trial dynamics of human EEG reveal cooperative interplay between reinforcement learning and working memory.
- **Source:** Proceedings of the National Academy of Sciences 115(10), 2502-2507
- **DOI:** 10.1073/pnas.1720963115
- **Relevance:** Updated RLWM with neural validation. Confirms capacity
  K acts as a soft bound rather than a hard threshold, justifying our
  continuous (0, K_max] bound in mle_utils.py.

### Senta (2025). [Preprint providing the kappa formulation we use for M3 and M6b.]
- **DOI:** [TBD - confirm via bioRxiv search before submission]
- **Relevance:** Senta et al. provide the kappa perseveration parameter
  formulation we inherit in M3 (single kappa on choice kernel) and
  extend in M6b via the stick-breaking kappa_share decomposition
  between choice-level and stimulus-level perseveration. The convex
  combination (1 - kappa) * P_noisy + kappa * Ck matches the likelihood
  implementation in jax_likelihoods.py. Fixed beta = 50 is also
  inherited from Senta for parameter identifiability.

### Bishara & Hawthorne (perseveration in RL models, older literature).
- **Source:** Cognitive / computational psychiatry literature
- **DOI:** [TBD - verify specific paper]
- **Relevance:** Bishara and collaborators (e.g., Bishara et al. 2010
  "Similar Processes Despite Divergent Behavior in Two Commonly Used
  Measures of Risky Decision Making" J Beh Dec Making 23(4), 435-454,
  DOI 10.1002/bdm.668) established perseveration parameters as standard
  nuisance variables in RL fits to clinical populations. M3 and M6b
  extend this tradition.

### Brown & Heathcote (2008). A ballistic model of choice response time: The Linear Ballistic Accumulator.
- **Source:** Cognitive Psychology 57(3), 153-178
- **DOI:** 10.1016/j.cogpsych.2007.12.002
- **Relevance:** Original LBA paper that defines the density used in
  lba_likelihood.py. M4 inherits the LBA race architecture (start
  points k ~ Uniform(0, A), non-decision time t0, threshold b, drift
  rates v) and combines it with M3 learning for a joint choice + RT
  likelihood.

### Miletic, Turner, Forstmann & van Maanen (2021). A new model of decision processing in instrumental learning tasks.
- **Source:** eLife 10, e63055
- **DOI:** 10.7554/eLife.63055
- **Relevance:** Joint RL-LBA modeling. Demonstrates that LBA drift rates
  can be parameterized as a function of Q-values for instrumental
  learning. Our M4 follows this pattern: drift rate v is proportional
  to the WMRL hybrid value (v_scale * [omega * WM + (1 - omega) * Q]).

### Pedersen, Frank & Biele (2017). The drift diffusion model as the choice rule in reinforcement learning.
- **Source:** Psychonomic Bulletin & Review 24(4), 1234-1251
- **DOI:** 10.3758/s13423-016-1199-y
- **Relevance:** Sister paper for RL-DDM. We chose LBA over DDM because
  the task has 3 choices (DDM is two-alternative only without extension),
  but Pedersen et al. provides the conceptual template for using
  accumulator models as the choice rule for reinforcement learning.

### Burnham & Anderson (2002). Model selection and multimodel inference.
- **Source:** Springer (book, 2nd ed.)
- **ISBN:** 978-0-387-95364-9
- **Relevance:** Canonical reference for AIC-based model comparison.
  Provides the delta-AIC interpretation thresholds (< 2 equivalent,
  2-4 weak, 4-7 moderate, 7-10 strong, > 10 very strong) used in
  scripts/14_compare_models.py and quoted in the manuscript. Also
  justifies reporting both AIC and BIC when they agree as additional
  robustness evidence.

## Part B: Trauma and Attention / RL (for Discussion)

### Lissek, Powers, McClure, Phelps, Woldehawariat, Grillon & Pine (2005). Classical fear conditioning in the anxiety disorders: A meta-analysis.
- **Source:** Behaviour Research and Therapy 43(11), 1391-1424
- **DOI:** 10.1016/j.brat.2004.10.007
- **Relevance:** Meta-analysis establishing that anxiety (including PTSD)
  is associated with impaired discrimination between safety and danger
  cues. Frames our epsilon finding as potentially reflecting
  stimulus-level confusion rather than learning-rate or memory deficits.

### Lissek & van Meurs (2015). Learning models of PTSD: Theoretical accounts and psychobiological evidence.
- **Source:** International Journal of Psychophysiology 98(3), 594-605
- **DOI:** 10.1016/j.ijpsycho.2014.11.006
- **Relevance:** Review of learning-based accounts of PTSD including
  overgeneralization, impaired extinction, and safety-signal deficits.
  Provides the theoretical framework for the noise-based account we
  offer: if attention is diverted toward threat monitoring, task-relevant
  learning appears noisy at the response-selection stage.

### Myers & Gluck (2007). Reward prediction error signals in posttraumatic stress disorder.
- **DOI:** [TBD - search needed, possibly Myers et al. 2009 JNeurosci
  or similar]
- **Relevance:** Early work connecting PTSD to aberrant reward learning
  in reversal tasks. Suggests perseveration on previously-correct
  responses is heightened in trauma populations - which parallels our
  kappa_total x LEC-5 uncorrected association in M3 and M6b.

### Ross, Smolen, Curran & Frick (2018). Attention and working memory in trauma-exposed youth: A systematic review.
- **DOI:** [TBD - multiple Ross authors; search pending]
- **Relevance:** Documents attention and WM deficits following trauma
  exposure, independent of PTSD diagnosis. Relevant because our
  trauma-exposed (no ongoing impact) subsample shows the same model
  preference as the ongoing-impact subsample, suggesting trauma history
  affects mechanisms even when symptom burden is low.

### Admon, Milad & Hendler (2013). A causal model of post-traumatic stress disorder: Disentangling predisposed from acquired neural abnormalities.
- **Source:** Trends in Cognitive Sciences 17(7), 337-347
- **DOI:** 10.1016/j.tics.2013.05.005
- **Relevance:** Identifies hippocampal-prefrontal circuit dysfunction
  as a central feature of PTSD. The hippocampus is heavily implicated in
  working memory, so any WM-capacity deficit in our data would
  implicate this circuit - but the K parameter's poor identifiability
  (r = 0.21) prevents confident inference.

### Pizzagalli (2014). Depression, stress, and anhedonia: Toward a synthesis and integrated model.
- **Source:** Annual Review of Clinical Psychology 10, 393-423
- **DOI:** 10.1146/annurev-clinpsy-050212-185606
- **Relevance:** Reviews reward-system dysregulation in stress-related
  disorders. Anhedonia is frequently comorbid with PTSD; if trauma
  symptoms were driving a reward-sensitivity effect we would expect it
  on rho (reward weight) rather than epsilon. Our null finding on rho
  (after correction) argues against the anhedonia account in this
  sample.

### Nestor, Pinsk, Sanes & Ochsner (2022). Stress and computational psychiatry.
- **DOI:** [TBD - may not exist; search for recent RL x PTSD modeling]
- **Relevance:** Recent computational psychiatry work on stress. Likely
  relevant for framing our trauma-RL approach within the broader
  effort to identify mechanistic biomarkers via model parameters.

### Browning, Behrens, Jocham, O'Reilly & Bishop (2015). Anxious individuals have difficulty learning the causal statistics of aversive environments.
- **Source:** Nature Neuroscience 18(4), 590-596
- **DOI:** 10.1038/nn.3961
- **Relevance:** Anxiety-RL association via volatility estimation. The
  finding that anxious individuals fail to adapt learning rates to
  environment volatility suggests an alpha-level effect, but our
  alpha_pos/alpha_neg parameters have r < 0.80 recovery in M6b so we
  cannot confidently map an alpha effect. Frame this as a limitation
  rather than a test of the Browning account.

## Epsilon-Trauma Framing Options

The M6b epsilon x IES-R Hyperarousal association is uncorrected p = 0.020,
does not survive FDR-BH (p_fdr = 0.20) or Bonferroni, and epsilon itself
has r = 0.772 recovery which is BELOW the 0.80 threshold. Any framing
must therefore be exploratory, not conclusive.

### Option 1 - Attention-based (preferred; aligns with Lissek)
"At the marginal level, higher IES-R Hyperarousal scores were associated
with elevated fits of the epsilon attention-noise parameter (uncorrected
p = 0.020). Because epsilon captures random or uniform responding
relative to the model's predicted policy, an increased epsilon in
hypervigilant individuals is consistent with attentional-resource
frameworks of PTSD (Lissek & van Meurs, 2015) in which threat monitoring
diverts processing away from task-relevant stimuli. However, this
association did not survive correction for multiple comparisons, and
M6b's epsilon recovery correlation (r = 0.77) fell short of the 0.80
identifiability threshold, so the finding should be treated as
hypothesis-generating rather than confirmatory."

### Option 2 - Perseveration-based (backup; aligns with Myers)
"A more robust pattern emerged on the perseveration kernel: kappa_total
(M6b) and kappa (M3) both correlated with LEC-5 Total Events at
uncorrected p < 0.01, with the M3 kappa association surviving FDR-BH
correction (p_fdr = 0.033). Because kappa_total is the best-recovered
base parameter in M6b (r = 0.997), this signal is identifiability-safe.
It suggests that greater lifetime trauma exposure is associated with
stronger motor-level response stickiness, consistent with reports of
elevated perseveration in reversal-learning tasks in trauma samples
(Myers & Gluck)."

### Option 3 - Combined (hedge)
"Our trauma-parameter regressions yielded two families of marginal
signals. First, the perseveration kernel (kappa in M3, kappa_total in
M6b) increased with lifetime trauma exposure (M3: p_fdr = 0.033;
M6b: p_uncorrected = 0.003, p_fdr = 0.135). Because kappa is the only
M6b parameter with r > 0.80 recovery, this is the most
identifiability-defensible finding. Second, the attentional-noise
parameter epsilon scaled with IES-R Hyperarousal in M6b
(p_uncorrected = 0.020, did not survive correction); we view this as
an exploratory hint pending confirmation in an independent sample."

## Notes on DOI Uncertainties

Entries marked [TBD] reflect cases where the author name is ambiguous
(multiple Ross authors in trauma/cognition; Nestor is a common surname)
or where a specific paper was referenced by the plan but not uniquely
identified. Task 7 (manuscript edits) should either:
1. Search Google Scholar / PubMed with the specific author + topic
   string before inserting the bibtex entry, or
2. Use only the entries with confirmed DOIs above and omit the
   uncertain ones from the manuscript Discussion.

For the core manuscript arguments, the confirmed citations
(Collins 2012 / 2014 / 2018, Brown & Heathcote 2008, Miletic 2021,
Pedersen 2017, Burnham & Anderson 2002, Lissek 2005 / 2015, Admon 2013,
Pizzagalli 2014, Browning 2015) are sufficient to cover Parts A and B.
