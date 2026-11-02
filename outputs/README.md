# Outputs

The twelve CSV artefacts written by the notebook **are committed**. They are a few hundred
kilobytes in total, they are the evidence behind every number in the report, and a marker should
be able to open `summary.csv` without installing BERTopic or waiting an hour for a rerun.

The one exception is `coding_sheet.csv`, which is excluded — see below.

> **Before the first push:** copy your real `outputs/` folder from `the project outputs folder`
> into this folder, so that the committed artefacts are the ones your run produced. Leave
> `coding_sheet.csv` behind; `.gitignore` excludes it, but do not rely on that alone.

## Manifest

| File | Written by | Contents | Evidence for |
| --- | --- | --- | --- |
| `model_comparison.csv` | §7.1 | Three configurations and an NMF baseline across three seeds: topics, UMass coherence, diversity, ARI seed consistency, outlier share | RQ1 internal quality; the choice of `min_topic_size=15` |
| `topic_screen.csv` | §9 | All 73 topics with proxy share, matched proxy groups and representative terms | Candidate nomination; the 0.08 proxy-share cut |
| `prevalence.csv` | §10 | Relative prevalence per topic per observed date, `NaN` where uncollected or under-sampled | RQ1 temporal structure; the collection gap |
| `expanding_window.csv` | §11 | Per-candidate concentration of original members in the cut-point model | Topic formability as documents accumulate |
| `detectability.csv` | §11 | First cut-point at which each candidate becomes formable, and the lag to it | Validation; mean lag 1.0 days |
| `onset.csv` | §12 | First usable value, observed maximum, and onset date or `indeterminate` per topic | RQ3; 7 of 8 onsets unobservable |
| `trends_specification.csv` | §13 | Query string or entity ID, type, geography, language, aggregation, baseline and criterion per topic | Reproducibility of the external series |
| `query_validity.csv` | §13 | Per-query conceptual validity (valid / ambiguous / invalid) with a stated reason | Which external comparisons are admissible at all |
| `lead_or_lag.csv` | §14 | Both onsets, the lead in days where estimable, and the interpretation otherwise | RQ3 — no valid estimate |
| `bertrend_signals.csv` | §15.1 | Per-slice BERTrend topics with popularity and noise / weak / strong classification | Published baseline behaviour |
| `bertrend_comparison.csv` | §15.2 | Candidate-to-BERTrend matching by centroid similarity, and the difference in flagging dates | Baseline recovers 8/8 within 0–5 days |
| `qualification.csv` | §16.1 | Per-candidate agreed-label counts, the consumer interest proportion and the qualification decision | RQ1 — one qualifying topic |
| `summary.csv` | §18 | Every reported metric mapped to the research question it addresses | The summary table in the report |

## Excluded, and not regenerable

`coding_sheet.csv` holds two coders' independent labels over 150 documents. It is excluded for
two separate reasons, and either alone would be sufficient:

1. It contains full tweet text, so redistributing it is inconsistent with the dataset's terms.
2. It is the only artefact in the project that **cannot be reproduced by running code**. Every
   other file in this folder is a deterministic function of the corpus and the seed.

The cell that creates the blank instrument is deliberately commented out in §16 for that second
reason: running it regenerates an empty sheet over the labels. Keep a backup outside the
repository folder. `docs/coding_instructions.txt` is the instrument as issued, and the aggregate
results computed from the sheet are in `qualification.csv` and `docs/run_log.md`, so nothing in
the report depends on the sheet itself being published.

If your supervisor wants the labels visible, a version with the `text` column replaced by a
SHA-256 hash of the text — preserving every label, dropping the redistributable content — is the
usual compromise.

## Regenerating

Run the notebook top to bottom with `Olympics_Tokyo_tweets.csv` in place. §16 and §16.3 need a
labelled `coding_sheet.csv`; without one they print `Coding not yet complete.` and the remaining
sections run unaffected, so a reviewer without the sheet still reproduces §1–§15 and §17.
