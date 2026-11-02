# Early Detection of Consumer Interest Shifts in Olympic Gymnastics Discourse

![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![BERTopic 0.17.4](https://img.shields.io/badge/BERTopic-0.17.4-informational)
![Licence MIT](https://img.shields.io/badge/licence-MIT-green)

Topic-model detection of consumer interest signals in Twitter discourse surrounding the Tokyo
2020 Olympic Games, with manual two-coder qualification, comparison against an external
search-interest series, and comparison against a published emerging-trend detection method.

The notebook is committed with its outputs intact, so every reported number can be read without
running anything.

## Research questions

| | Question | Answered? |
| --- | --- | --- |
| **RQ1** | Which consumer interest topics can be identified in event-associated social media discourse, and with what internal quality? | Yes — one qualifying topic |
| **RQ2** | Do identified topics correspond to an independent external interest series? | Yes — 4/4 valid queries |
| **RQ3** | Where both series admit an onset, does the social signal lead or lag the external one? | **No valid estimate** |

## Result

| Stage | Outcome |
| --- | --- |
| Cleaned corpus | 261,467 documents from 312,791 records |
| Unrestricted discovery | 291 topics, 39.5% outliers |
| Gymnastics discourse set | 7,129 documents (2.73% of the corpus) from 5,982 accounts |
| Topic discovery | 73 topics, UMass coherence −2.097, diversity 0.844, seed consistency 0.633 |
| Behavioural screen | 8 candidate topics nominated |
| Two-coder qualification | **1 qualifying topic** — broadcast and streaming access |
| Inter-coder agreement | raw 87.3%, linearly weighted κ = 0.461, n = 150 |
| Screen evaluation | precision 0.05, recall 1.00, against 131 agreed labels |
| BERTrend baseline | recovers 8/8 candidates, within 0–5 days |

**The principal finding is a negative one.** The observation window opens on 25 July 2021, after
the discourse of interest was already underway. Seven of eight candidate topics are already
elevated at the first usable observation, so their onsets cannot be located, and every
conceptually valid external query returns an onset of 25 July — the first date of the window — at
every detection threshold tested between 1.0 and 2.5 standard deviations. RQ3 therefore yields no
valid lead-or-lag estimate. No analytical procedure resolves this; it requires a corpus whose
collection begins before the period of interest.

## Where to start

| If you want | Open |
| --- | --- |
| The result, in two minutes | `outputs/summary.csv` |
| The analysis with its outputs and explanations | `notebooks/consumer_interest_pipeline.ipynb` |
| The same thing without a notebook viewer | `docs/executed_notebook.html` |
| Every console figure as plain text | `docs/run_log.md` |
| Why each threshold is what it is, and what was fixed | `docs/decisions.md` |
| Readable code diffs | `notebooks/consumer_interest_pipeline.py` |

## Layout

```
notebooks/   the executed analysis, plus a jupytext export for readable diffs
outputs/     the twelve CSV artefacts, committed as evidence
cache/       cached Google Trends responses, so the reported numbers reproduce
docs/        run log, rendered run, design decisions, the coders' instrument
figures/     figures prepared for the written report
data/        dataset provenance; the corpus itself is not redistributed
scripts/     validation helpers used by make and by CI
```

There is no `src/` package, deliberately. The notebook is the artefact that was executed and
whose outputs are the evidence. Refactoring it into modules that were never run would put code in
the repository whose results nothing in `outputs/` supports — the structure would look more
rigorous while the evidence got weaker. `notebooks/consumer_interest_pipeline.py` is the
compromise: a mechanical jupytext export, with CI asserting the two never drift apart. It is
generated, never hand-edited.

## Data

| | |
| --- | --- |
| Dataset | *Olympics (Tokyo 2021) Tweets, 300,000+*, by Amit Pratap Singh |
| Source | <https://www.kaggle.com/datasets/amritpal333/olympics-tokyo-2021-tweets> |
| Records | 312,791, spanning 25 July – 8 August 2021 |
| Dates actually collected | 9 of 15 — 2–7 August are absent |
| External series | Google Trends via `pytrends`, worldwide, `en-US`, daily |

The corpus is not committed; `data/README.md` explains why and how to obtain it. The collection
gap is the most consequential property of the data and is handled explicitly rather than
interpolated: uncollected dates are `NaN`, and dates with fewer than 30 gymnastics documents are
also `NaN`, because too few observations cannot establish absence. An earlier version treated
25 July (2 documents) as a zero, which manufactured an apparent emergence on 26 July for every
topic — see `docs/decisions.md`.

## Pipeline

Sections correspond one-to-one to the notebook.

| § | Stage |
| --- | --- |
| 1–3 | Environment and pinned versions; UTC parsing and a collection-coverage audit; cleaning, with normalisation before deduplication |
| 4 | **Unrestricted topic discovery** — BERTopic on the full corpus, no domain restriction |
| 5–6 | Domain vocabulary derived from three indisputable seed terms; filtering on unambiguous, ambiguous and context term classes, with both decision outcomes inspected |
| 7–8 | Model selection across three configurations and three seeds, plus NMF at matched topic count; discovery with the selected configuration |
| 9 | Consumer interest screening — eight behavioural proxy groups nominate candidates; nomination is not classification |
| 10–12 | Relative prevalence P(k,d) = n(k,d)/N(d); expanding-window validation by document membership; onset assessment |
| 13–14 | External comparison, with Trends entities where a search term would be ambiguous; lead or lag only where both onsets are observable |
| 15 | **Published baseline** — BERTrend reimplemented and compared |
| 16 | Two-coder qualification: weighted κ, screen evaluation with Wilson intervals |
| 17–18 | Sensitivity of every threshold adopted without precedent; summary by research question |

Discovery precedes selection throughout. The gymnastics domain is **selected from the evidence**
in §4–§5, not assumed.

## Reproducing

Python 3.11 is required. BERTopic's dependency set does not resolve cleanly on 3.13, and 3.9 is
below the floor for the pinned `sentence-transformers`.

```bash
py -3.11 -m venv venv                # Windows
.\venv\Scripts\Activate.ps1
# python3.11 -m venv venv            # macOS / Linux
# source venv/bin/activate

pip install -r requirements.txt
python -m ipykernel install --user --name venv
```

If `hdbscan` or `umap-learn` fail to build on Windows, use `conda env create -f environment.yml`
instead. Place `Olympics_Tokyo_tweets.csv` in the repository root and run all cells.

A CUDA build of PyTorch is optional and gives roughly a 4× speed-up on the embedding steps.
Install it explicitly — the default PyPI wheel is CPU-only, so
`torch.cuda.is_available() == False` after a plain `pip install torch` means the CPU wheel is
installed, not that the GPU is unavailable:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Runtime is roughly 40–60 minutes on CPU, dominated by the embedding passes in §4 and §15, with
peak memory around 6 GB. The `del` and `gc.collect()` calls in the model-selection sweep are
deliberate: it fits seventeen models and exhausted memory without them.
`os.environ["HF_HUB_OFFLINE"] = "1"` is set immediately after `import os`, before any Hugging
Face import, so the sentence-transformer loads from cache; set after those imports it has no
effect, which is a silent failure rather than an error.

### What cannot be re-executed

**The coding sheet.** §16 reads `outputs/coding_sheet.csv`, holding two coders' independent
labels. It is excluded from version control, contains full tweet text, and is the only artefact
that cannot be regenerated by running code — so the cell that creates the blank instrument is
deliberately commented out. §16 prints `Coding not yet complete.` and continues when the sheet is
absent, so §1–§15 and §17 reproduce without it.

**Google Trends.** Responses are cached under `cache/`, which is committed for exactly this
reason: Trends normalises each series to the maximum of the *requested* window, so the same query
run today over a different window returns different values. Without the cache the reported
numbers cannot be reproduced.

## BERTrend baseline, and where this implementation deviates

BERTrend (Boutaleb, Picault & Grosjean, 2024, *arXiv:2411.05930*) is reimplemented from
Algorithm 1 rather than taken from the authors' release, which expects a different corpus layout.
These deviations are deliberate and recorded rather than presented as a faithful reproduction:

| Aspect | Published | Here | Reason |
| --- | --- | --- | --- |
| Slice granularity | configurable | daily | matches the collection granularity |
| Merge threshold, retained embedding, decay | 0.7 cosine; first occurrence; `p *= exp(−λ·gap²)`, λ = 0.01 | unchanged | — |
| Signal classification | rolling percentiles | `<P10` noise, `>P50` strong, between = weak only if an OLS slope over the window is positive | the paper leaves the weak-signal rule to the implementer |
| Window | configurable | 7 days | the shortest the nine observed dates support |
| Minimum slice cluster | configurable | 5 | daily slices are small |

Topics are matched by **centroid cosine similarity**, appropriate here because both sides are
full-corpus representations at comparable granularity — unlike §11, where the same measure was
tried and rejected for comparing a broad model to a fine one. BERTrend recovers all eight
candidates at similarities of 0.78–0.997 and flags them within 0–5 days, establishing that
candidate nomination is not an artefact of this framework's own clustering.

## Qualification instrument

The screen in §9 **nominates** candidates; it does not classify them. Qualification is manual, by
two coders working independently over 150 documents — 15 from each of the 8 candidate topics and
3 from each of 10 randomly drawn non-candidate topics, shuffled, with the screen decision
withheld.

| Label | Meaning |
| --- | --- |
| 2 | Consumer interest — the **author** expresses their own searching, wanting, buying, using, price or brand interest, viewing, or participation |
| 1 | Product mention — a product or service is present, but the author expresses no consumer engagement |
| 0 | Description or reporting — an event, athlete behaviour, result or news item |

The full instrument is `docs/coding_instructions.txt`. The 2-versus-0 distinction is whether the
author engages as a consumer or reports what someone else did; a headline describing what athletes
wore is 0 however much product vocabulary it carries. An earlier wording was read as "mentions a
product" and produced 118 of 150 documents at label 1 with none at label 2, and a misleadingly
high κ of 0.852; it was rewritten and the round rerun.

κ uses **linear weighting**, since the categories are ordered. Raw agreement of 87.3% against
weighted κ of 0.461 is the kappa paradox (Feinstein & Cicchetti, 1990): agreement concentrates in
the majority category. Coders agreed on 87% of documents where either used label 0, but 33% where
either used label 2 and 29% for label 1. Both figures are reported, because either alone misleads.

A topic qualifies where at least 25% of its agreed-label documents carry label 2. §16.3 varies
this: the observed proportions are 0.00 and 0.38, so no topic falls between them and the outcome
is invariant for any threshold in that interval.

## Negative and weak results

Stated rather than minimised.

- **RQ3 yields no valid estimate.** 7 of 8 candidate onsets are unobservable; the one estimate
  available comes from an ambiguous query and is excluded from the finding.
- **Screen precision is 0.05** (95% Wilson interval [0.02, 0.12]). Suitable for nomination, not
  for classification — which is why qualification is manual.
- **Recall is 1.00** but rests on five positive cases, interval [0.57, 1.00]. One missed sixth
  case would have taken it to 0.83.
- **Weighted κ is 0.461** — moderate. Disagreements are systematic, not dispersed: nine on the
  0–2 boundary, concentrated on retrospective or hypothetical participation and on shared news
  content carrying added commentary.
- **Two of the four valid external queries are duplicates**, so they are not four independent
  observations.
- **Seed consistency is 0.633** (ARI across three seeds). HDBSCAN cluster selection is not
  deterministic at this granularity, and the figure is reported rather than hidden by presenting
  a single seed.
- **Qualification is not invariant to disagreement resolution** for one of the eight candidates.

## References

Boutaleb, A., Picault, J., & Grosjean, G. (2024). BERTrend: Neural topic modeling for emerging
trends detection. *arXiv:2411.05930*.

Feinstein, A. R., & Cicchetti, D. V. (1990). High agreement but low kappa: I. The problems of two
paradoxes. *Journal of Clinical Epidemiology*, 43(6), 543–549.

Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure.
*arXiv:2203.05794*.

McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform manifold approximation and
projection for dimension reduction. *arXiv:1802.03426*.

Singh, A. (2021). *Olympics (Tokyo 2021) tweets 300,000+* [Data set]. Kaggle.

Full bibliography: `docs/references.bib`.

## Licence

MIT, for the code in this repository. The source corpus is not redistributed and remains subject
to the terms of its Kaggle listing and of the X/Twitter Developer Agreement.
