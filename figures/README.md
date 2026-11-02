# Figures

Two sets of figures exist in this project and they should not be confused.

## Figures produced by the notebook

Sixteen, embedded in `notebooks/consumer_interest_pipeline.ipynb` and visible in
`docs/executed_notebook.html`. The largest group is the eight paired plots from §13.2: relative
topic prevalence in the upper panel, relative search interest in the lower, with the Google
Trends baseline period shaded so that it is obvious the baseline precedes the Twitter observation
window. These are analysis figures. They are evidence and they are committed inside the notebook.

## Figures prepared for the report

Six, drawn from `outputs/*.csv` for the Results chapter of the report. They are not
produced by the notebook, because they are presentation artefacts with report-specific styling,
captions and sizing — including one figure in §4.2 that carries a zero below-figure skip to stop
an orphaned line holding a page open.

This folder is where their generation script belongs. It is **not committed yet**: the script was
written against the report draft and has not been reconciled with the final §4 numbering. Add it
as `figures/make_figures.py`, reading only from `outputs/` so that the figures cannot drift from
the artefacts, and writing PNG at 300 dpi for Overleaf.

A script that invents its inputs is worse than no script, which is why nothing is committed here
in the meantime.

## Conventions, when the script is added

| | |
| --- | --- |
| Input | `outputs/*.csv` only — never a hard-coded number |
| Output | `figures/fig_<n>_<slug>.png`, 300 dpi |
| Size | 10 × 5 inches for paired panels, 8 × 4.5 for single |
| Colour | one accent (`#2563eb`) against grey; no colour carries meaning alone |
| Missing data | gaps left as gaps. A collection gap is not a zero, and a figure must not imply it is |

The last line is the one that matters. The defect that produced a false emergence for every topic
in an earlier version of the analysis was exactly this mistake made in code rather than in a
plot, and it is recorded in `docs/decisions.md`.
