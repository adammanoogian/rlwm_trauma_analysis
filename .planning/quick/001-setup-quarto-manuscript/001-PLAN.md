---
phase: quick
plan: 001
type: execute
wave: 1
depends_on: []
files_modified:
  - manuscript/_quarto.yml
  - manuscript/paper.qmd
  - manuscript/arxiv_template.tex
  - manuscript/references.bib
  - manuscript/figures/plot_utils.py
  - .gitignore
autonomous: true

must_haves:
  truths:
    - "quarto render manuscript/paper.qmd produces valid PDF output"
    - "Setup cell loads real MLE output CSVs from ../output/mle/ without error"
    - "Paper sections match RLWM trauma analysis content (not template schizotypy content)"
    - "Matplotlib style in manuscript matches project plotting_config.py conventions"
  artifacts:
    - path: "manuscript/_quarto.yml"
      provides: "Quarto project config for Python/rlwm env"
    - path: "manuscript/paper.qmd"
      provides: "Main manuscript with RLWM-specific sections and code cells"
    - path: "manuscript/arxiv_template.tex"
      provides: "LaTeX preamble for arXiv-style PDF"
    - path: "manuscript/references.bib"
      provides: "Bibliography with Senta 2025, Collins & Frank, etc."
    - path: "manuscript/figures/plot_utils.py"
      provides: "Consistent matplotlib style for manuscript figures"
  key_links:
    - from: "manuscript/paper.qmd"
      to: "../output/mle/"
      via: "pd.read_csv in setup cell"
      pattern: "read_csv.*output/mle"
    - from: "manuscript/figures/plot_utils.py"
      to: "../../plotting_config.py"
      via: "Mirrors PlotConfig color/font settings for publication"
      pattern: "GROUP_COLORS|font.family"
---

<objective>
Set up a Quarto scientific manuscript project in `manuscript/` for the RLWM trauma analysis paper.

Purpose: Create the scaffolding for a reproducible manuscript where figures and inline statistics are generated directly from MLE fitting outputs, ensuring the paper stays in sync with analysis results.

Output: A complete `manuscript/` directory with Quarto config, paper.qmd adapted for RLWM trauma content (7 models, 154 participants, trauma group analyses), LaTeX template, bibliography, and consistent plotting utilities.
</objective>

<execution_context>
@C:\Users\aman0087\.claude/get-shit-done/workflows/execute-plan.md
@C:\Users\aman0087\.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@CLAUDE.md
@config.py
@plotting_config.py

Template source files (adapt, do not copy verbatim):
- C:\Users\aman0087\AppData\Local\Temp\quarto_skill\quarto-scientific\assets\starter_project\_quarto.yml
- C:\Users\aman0087\AppData\Local\Temp\quarto_skill\quarto-scientific\assets\starter_project\paper.qmd
- C:\Users\aman0087\AppData\Local\Temp\quarto_skill\quarto-scientific\assets\starter_project\arxiv_template.tex
- C:\Users\aman0087\AppData\Local\Temp\quarto_skill\quarto-scientific\assets\starter_project\references.bib
- C:\Users\aman0087\AppData\Local\Temp\quarto_skill\quarto-scientific\assets\starter_project\.gitignore
</context>

<tasks>

<task type="auto">
  <name>Task 1: Create manuscript infrastructure files</name>
  <files>
    manuscript/_quarto.yml
    manuscript/arxiv_template.tex
    manuscript/references.bib
    manuscript/figures/plot_utils.py
    .gitignore
  </files>
  <action>
Create `manuscript/` directory and `manuscript/figures/` subdirectory.

**manuscript/_quarto.yml** — Adapt from template with these changes:
- `output-dir: _output` (keep as-is)
- `execute.cache: true` (keep)
- Remove `csl: apa.csl` line (we do not have the file yet; natbib handles citation style in PDF)
- PDF format: keep `template: arxiv_template.tex`, `cite-method: natbib`, `number-sections: true`, `colorlinks: true`, `fig-pos: 'H'`, `geometry: margin=1in`, `keep-tex: true`
- HTML format: keep as template (cosmo theme, toc, embed-resources)
- LaTeX format: keep as template

**manuscript/arxiv_template.tex** — Copy verbatim from template. This is a Pandoc template with `$variable$` syntax; do not modify it.

**manuscript/references.bib** — Replace template entries with entries relevant to RLWM trauma analysis. Include these references (use reasonable placeholder metadata where exact pages/volumes are unknown):
1. `senta2025` — Senta, Rmus, Hartley, Collins (2025). Working memory and reinforcement learning paper (the primary methodological reference).
2. `collins2014working` — Collins & Frank (2014). Working memory contributions to reinforcement learning. Cognitive, Affective, & Behavioral Neuroscience.
3. `collins2012how` — Collins & Frank (2012). How much of reinforcement learning is working memory, not reinforcement learning? A behavioral, computational, and neurogenetic analysis. European Journal of Neuroscience, 35(7), 1024-1035.
4. `daw2011model` — Daw (2011). Trial-by-trial data analysis using computational models. In Decision Making, Affect, and Learning (Oxford UP).
5. `weathers2013life` — Weathers et al. (2013). The Life Events Checklist for DSM-5 (LEC-5). National Center for PTSD.
6. `weiss2007impact` — Weiss (2007). The Impact of Event Scale-Revised. In Assessing psychological trauma and PTSD.
7. `lissek2005classical` — Lissek et al. (2005). Classical fear conditioning in functional neuroimaging. Review of fear conditioning and trauma/PTSD.
8. `homan2019neural` — Homan et al. (2019). Neural computations of threat in the aftermath of combat trauma. Nature Neuroscience.
9. `browning2015anxious` — Browning, Behrens, Jocham, O'Reilly, Bishop (2015). Anxious individuals have difficulty learning the causal statistics of aversive environments. Nature Neuroscience.
10. `gillan2016characterizing` — Gillan, Kosinski, Whelan, Phelps, Daw (2016). Characterizing a psychiatric symptom dimension related to deficits in goal-directed control. eLife.
11. `millner2018pavlovian` — Millner, Gershman, Nock, den Ouden (2018). Pavlovian control of escape and avoidance. Journal of Cognitive Neuroscience.

**manuscript/figures/plot_utils.py** — Create a publication-quality matplotlib style module that mirrors `plotting_config.py` but tuned for manuscript figures:
- Import `from __future__ import annotations` at top
- Define `MANUSCRIPT_STYLE` dict with rcParams for publication: `font.family: serif`, `font.serif: ['Computer Modern Roman', 'Times New Roman']`, `font.size: 9`, `axes.labelsize: 10`, `axes.titlesize: 11`, `legend.fontsize: 8`, `figure.dpi: 300`, `savefig.dpi: 300`, `axes.linewidth: 0.8`, `lines.linewidth: 1.5`
- Copy `GROUP_COLORS` dict from `plotting_config.py` exactly (control=#06A77D, exposed=#F18F01, symptomatic=#D62246)
- Define `MODEL_DISPLAY_NAMES` dict mapping internal names to display names: `{'qlearning': 'M1: Q-Learning', 'wmrl': 'M2: WM-RL', 'wmrl_m3': 'M3: WM-RL+kappa', 'wmrl_m4': 'M4: RLWM-LBA', 'wmrl_m5': 'M5: WM-RL+phi_rl', 'wmrl_m6a': 'M6a: WM-RL+kappa_s', 'wmrl_m6b': 'M6b: WM-RL+dual'}`
- Define `PARAM_DISPLAY_NAMES` dict mapping parameter column names to LaTeX-formatted display names: `{'alpha_pos': r'$\alpha_+$', 'alpha_neg': r'$\alpha_-$', 'phi': r'$\phi$', 'rho': r'$\rho$', 'K': r'$K$', 'kappa': r'$\kappa$', 'kappa_s': r'$\kappa_s$', 'kappa_total': r'$\kappa_{\mathrm{total}}$', 'kappa_share': r'$\kappa_{\mathrm{share}}$', 'phi_rl': r'$\phi_{\mathrm{RL}}$', 'epsilon': r'$\varepsilon$'}`
- Define `apply_manuscript_style()` function that calls `plt.rcParams.update(MANUSCRIPT_STYLE)`
- Define `COLUMN_WIDTH = 3.5` and `TEXT_WIDTH = 7.0` constants (inches, standard for two-column journals)
- Add NumPy-style docstrings to everything

**Append to project .gitignore** — Add a `# Quarto manuscript build artifacts` section at the end with:
```
manuscript/_output/
manuscript/_cache/
manuscript/*.aux
manuscript/*.log
manuscript/*.out
manuscript/*.toc
manuscript/*.synctex.gz
manuscript/*.bbl
manuscript/*.blg
```
Do NOT remove any existing gitignore entries.
  </action>
  <verify>
- `ls manuscript/_quarto.yml manuscript/arxiv_template.tex manuscript/references.bib manuscript/figures/plot_utils.py` all exist
- `python -c "import sys; sys.path.insert(0, 'manuscript/figures'); import plot_utils; plot_utils.apply_manuscript_style(); print('OK')"` runs without error
- `grep 'Quarto manuscript' .gitignore` finds the new section
- `grep 'senta2025' manuscript/references.bib` finds the primary reference
  </verify>
  <done>
All infrastructure files exist. plot_utils.py imports cleanly. Project .gitignore updated with Quarto artifacts. references.bib contains 11 relevant entries.
  </done>
</task>

<task type="auto">
  <name>Task 2: Create paper.qmd with RLWM trauma analysis content</name>
  <files>
    manuscript/paper.qmd
  </files>
  <action>
Create `manuscript/paper.qmd` adapted from the template but with content specific to this project. The template's schizotypy/physics content must be completely replaced. Use the template's YAML frontmatter structure and code cell patterns as a guide.

**YAML frontmatter:**
```yaml
title: "Dissociating Perseverative Responding from Learning-Rate Effects in Trauma: A Computational Modeling Approach"
author:
  - name: Adam Manoogian
    orcid: 0009-0002-8002-3191
    email: adam.manoogian@monash.edu
    affiliations:
      - id: mc3s
        name: Monash Centre for Consciousness and Contemplative Studies, Monash University
      - id: mbi
        name: Monash Biomedical Imaging, Monash University
      - id: turner
        name: Turner Institute for Brain and Mental Health, Monash University
  - name: Coauthor Name
    affiliations:
      - ref: mbi
abstract: |
  ABSTRACT PLACEHOLDER. Trauma exposure alters reinforcement learning and working memory processes, but the computational mechanisms remain unclear. We fitted seven computational models (M1--M6b) to data from 154 participants completing a reinforcement learning working memory task, and examined how model parameters relate to trauma exposure and post-traumatic stress symptoms. [To be completed.]
keywords: [reinforcement learning, working memory, trauma, perseveration, computational psychiatry, PTSD]
bibliography: references.bib
execute:
  echo: false
  warning: false
  message: false
format:
  pdf:
    template: arxiv_template.tex
    keep-tex: true
    cite-method: natbib
    number-sections: true
    colorlinks: true
    fig-pos: 'H'
```

**Setup cell** (label: setup, no echo):
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
import sys

# Add manuscript figures module to path
sys.path.insert(0, str(Path("figures")))
from plot_utils import (
    apply_manuscript_style,
    GROUP_COLORS,
    MODEL_DISPLAY_NAMES,
    PARAM_DISPLAY_NAMES,
    COLUMN_WIDTH,
    TEXT_WIDTH,
)

apply_manuscript_style()

# ── Paths (relative to manuscript/ directory) ────────────────────────────
MLE_DIR = Path("../output/mle")
FIGURES_DIR = Path("../figures")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Load MLE results ─────────────────────────────────────────────────────
models = ["qlearning", "wmrl", "wmrl_m3", "wmrl_m5", "wmrl_m6a", "wmrl_m6b", "wmrl_m4"]
fits = {}
for model in models:
    fpath = MLE_DIR / f"{model}_individual_fits.csv"
    if fpath.exists():
        fits[model] = pd.read_csv(fpath)

# ── Load performance summaries ───────────────────────────────────────────
perf = {}
for model in models:
    fpath = MLE_DIR / f"{model}_performance_summary.json"
    if fpath.exists():
        with open(fpath) as f:
            perf[model] = json.load(f)

# ── Key numbers for inline text ──────────────────────────────────────────
n_participants = len(fits.get("wmrl_m5", pd.DataFrame()))
n_models = len(fits)
winning_model = "wmrl_m5"  # M5 wins by dAIC=435.6 over M3
```

**Sections — write as prose-ready placeholders with real section cross-refs and citation keys. Each section should have 2-4 sentences of real content direction (not lorem ipsum) so the author knows what goes where. Include code cells where figures/tables will go.**

1. `## Introduction {#sec-intro}` — Frame the paper: trauma alters RL [@lissek2005classical; @homan2019neural; @browning2015anxious], WM contributions to RL [@collins2012how; @collins2014working], computational psychiatry of trauma [@gillan2016characterizing; @millner2018pavlovian], gap: perseveration vs learning rates not dissociated in trauma. Cite Senta et al. [@senta2025] as the modeling framework.

2. `## Methods {#sec-methods}` with subsections:
   - `### Participants {#sec-participants}` — Use inline code: `` `{python} n_participants` `` participants. Trauma screening via LEC-5 [@weathers2013life] and IES-R [@weiss2007impact]. Three groups: control, exposed, symptomatic.
   - `### Task {#sec-task}` — RLWM task from Senta et al. [@senta2025]. Set sizes 2, 3, 5, 6. Three response options. Reversals. Describe briefly.
   - `### Computational Models {#sec-models}` — Describe all 7 models in a table. Include the model equation for WM-RL (the mixture: `p(a|s) = w * WM(s,a) + (1-w) * Q(s,a)` with `w = K/(K + phi*n_s)`). Reference @sec-model-table. Note M4 is joint choice+RT (LBA) and cannot be compared by AIC with choice-only models.
   - `### Model Fitting {#sec-fitting}` — MLE via scipy.optimize.minimize (L-BFGS-B), 10 random restarts per participant. JAX-accelerated likelihoods. AIC for model comparison [@daw2011model].
   - `### Statistical Analysis {#sec-stats}` — Group comparisons (Kruskal-Wallis, Mann-Whitney U with Bonferroni), Spearman correlations, OLS regressions of parameters on IES-R subscales.

3. `## Results {#sec-results}` with subsections:
   - `### Model Comparison {#sec-model-comparison}` — Include a code cell (label: `tbl-model-comparison`, tbl-cap: "Model comparison by AIC/BIC...") that builds a comparison table from the `perf` dict. Placeholder code that reads from perf summaries and displays a DataFrame with columns: Model, Free Parameters, Mean AIC, Mean BIC, dAIC.
   - `### Winning Model: M5 {#sec-winning-model}` — Describe M5 (WM-RL + phi_rl) as the winning model. Note dAIC=435.6 over M3.
   - `### Parameter-Trauma Group Relationships {#sec-group-results}` — Include a code cell (label: `fig-parameters-by-group`, fig-cap about parameter distributions by trauma group) with placeholder that loads `wmrl_m5_group_summary.csv` and creates violin/box plots colored by GROUP_COLORS.
   - `### Continuous Trauma Associations {#sec-correlations}` — Include a code cell (label: `fig-correlation-heatmap`, fig-cap about Spearman correlations) with placeholder that loads correlation results.
   - `### Regression Analyses {#sec-regressions}` — Include a code cell (label: `tbl-regression`, tbl-cap about OLS regression results) with placeholder for regression table.

4. `## Discussion {#sec-discussion}` — Placeholder structure: (1) summary of findings, (2) perseveration vs learning rates, (3) WM capacity and trauma, (4) comparison to Senta et al., (5) limitations, (6) clinical implications.

5. `## Conclusion {#sec-conclusion}` — One paragraph placeholder.

6. `## References {.unnumbered}` with `::: {#refs} :::` div.

7. `## Appendix {.appendix}` with:
   - `### Model Parameters {#sec-model-table}` — A table listing all 7 models with their free parameters (from CLAUDE.md Parameter Summary table), rendered as a DataFrame.
   - `### Exclusion Criteria {#sec-exclusions}` — Minimum 400 trials threshold (from config.py MIN_TRIALS_THRESHOLD).

For all code cells that produce figures:
- Use `fig-width` and `fig-height` that correspond to COLUMN_WIDTH (3.5) for single-column or TEXT_WIDTH (7.0) for full-width.
- Use `plt.savefig(OUTPUT_DIR / "filename.pdf", bbox_inches='tight')` pattern.
- Include `plt.show()` at the end.

For code cells that are placeholders (data not guaranteed to exist at render time), wrap the loading/plotting in a try/except that prints a clear message: "Data not yet available. Run scripts/15_analyze_mle_by_trauma.py first."
  </action>
  <verify>
- `ls manuscript/paper.qmd` exists
- `grep 'Dissociating Perseverative' manuscript/paper.qmd` finds the title
- `grep 'output/mle' manuscript/paper.qmd` finds data loading paths
- `grep 'sec-intro' manuscript/paper.qmd` and `grep 'sec-methods' manuscript/paper.qmd` find section cross-refs
- `grep 'senta2025' manuscript/paper.qmd` finds citations to the primary reference
- `grep 'plot_utils' manuscript/paper.qmd` confirms the style module is imported
- Count of `{python}` code cells is at least 5 (setup + model comparison table + parameters figure + correlation figure + regression table)
  </verify>
  <done>
paper.qmd contains: RLWM-specific title/abstract/keywords, all 7 sections (Intro through Appendix), setup cell loading from ../output/mle/, at least 5 code cells for figures/tables, inline Python for participant count, citations using correct bibtex keys, cross-references between sections. No schizotypy/physics template content remains.
  </done>
</task>

</tasks>

<verification>
1. All 6 files exist: `manuscript/_quarto.yml`, `manuscript/paper.qmd`, `manuscript/arxiv_template.tex`, `manuscript/references.bib`, `manuscript/figures/plot_utils.py`, updated `.gitignore`
2. `python -c "import sys; sys.path.insert(0, 'manuscript/figures'); import plot_utils; print('OK')"` passes
3. `grep -c '@' manuscript/paper.qmd` shows multiple citation references
4. `grep -c 'python' manuscript/paper.qmd` shows multiple code cells
5. No template schizotypy/physics content remains: `grep -i 'schizotypy\|physics\|collision\|egress' manuscript/paper.qmd` returns nothing
</verification>

<success_criteria>
- manuscript/ directory is a self-contained Quarto project that could render (modulo Quarto installation and data availability)
- paper.qmd sections match the RLWM trauma analysis paper structure
- Setup cell correctly references ../output/mle/ paths for all 7 models
- Plotting style is publication-quality and consistent with project conventions
- .gitignore prevents Quarto build artifacts from being committed
- references.bib contains at least 10 relevant computational psychiatry / RL / trauma references
</success_criteria>

<output>
After completion, create `.planning/quick/001-setup-quarto-manuscript/001-SUMMARY.md`
</output>
