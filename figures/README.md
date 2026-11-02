# Figures

Two sets of figures exist in this project and they should not be confused.

## Analysis figures

Sixteen, embedded in `notebooks/consumer_interest_pipeline.ipynb` and visible in
`docs/executed_notebook.html`. The largest group is the eight paired plots from §13.2: relative
topic prevalence in the upper panel, relative search interest in the lower, with the Google
Trends baseline period shaded so that it is obvious the baseline precedes the Twitter observation
window. These are evidence, and they live inside the notebook that produced them.

## Report figures

Seven, committed here as PNG. They are drawn from `outputs/*.csv` for the written report, with
report-specific styling, captions and sizing, and are not produced by the notebook.

| File | Shows |
| --- | --- |
| `fig1_coverage.png` | Collection coverage by date — which of the 15 calendar dates were sampled, and for how many hours |
| `fig2_models.png` | Model comparison across configurations and seeds: topics, coherence, diversity, outlier share |
| `fig2b_stability.png` | Seed stability of the selected configuration |
| `fig3_prevalence.png` | Relative topic prevalence over the observation window for the candidate topics |
| `fig4_disagreement.png` | Inter-coder disagreement by category boundary and by topic |
| `fig5_paired.png` | Topic prevalence against external search interest for the qualifying topic |
| `fig6_detectability.png` | Expanding-window detectability: the lag before each candidate becomes formable |

## Conventions

| | |
| --- | --- |
| Input | `outputs/*.csv` only — never a hard-coded number |
| Output | PNG at 300 dpi |
| Size | 10 × 5 inches for paired panels, 8 × 4.5 for single |
| Colour | one accent (`#2563eb`) against grey; no colour carries meaning alone |
| Missing data | gaps left as gaps. A collection gap is not a zero, and a figure must not imply it is |

That last line is the one that matters. The defect that produced a false emergence for every topic
in an earlier version of the analysis was exactly this mistake made in code rather than in a plot,
and it is recorded in `docs/decisions.md`.

## The generation script

Not committed. The figures above were produced from `outputs/*.csv` by a script that has not been
reconciled with the final section numbering. Adding it as `figures/make_figures.py` would make
this folder fully reproducible, and until then the PNGs are the artefact.