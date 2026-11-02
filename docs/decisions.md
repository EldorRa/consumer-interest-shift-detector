# Decisions, defects and corrections

A record of the choices in the notebook that are not self-evident from the code, and of the
defects found during implementation and what was done about them. Each entry names the section
it applies to.

The point of the file is that an examiner can see which numbers are the product of a considered
choice, which are the product of a correction, and which would change if the choice were made
differently. Where a choice is arbitrary, §17 and §16.3 vary it and the result is reported.

---

## Part 1 — Defects found and corrected

These were present in earlier versions of the analysis and are fixed in the committed notebook.
They are listed because the corrected numbers differ materially from the uncorrected ones, and
because two of them would have produced a publishable-looking but false result.

### Prefix matching inflated every term count (§3.1)

The term matcher anchored a word boundary at the start of the search term only:

```python
re.search(r"\b" + term, text)        # wrong
```

`cost` then matched *Costa*, `app` matched *apparently*, `ring` matched *ringing*. The effect was
not marginal: `cost` was inflated by roughly 27×. The consequence was that the consumer-interest
proxy screen nominated topics on the strength of matches that were not the term at all.

Corrected to require a boundary at both ends:

```python
def has_exact_word(text, words):
    return any(re.search(r"\b" + re.escape(w) + r"\b", text) for w in words)
```

`re.escape` matters too, since several terms are multi-word and one contains an apostrophe.

### A three-document day manufactured an emergence for every topic (§10)

25 July 2021 is a stub: three hours collected, two gymnastics documents after cleaning. Treated
as a real observation it gave every topic a prevalence of 0.000 on that date, and therefore an
apparent onset on 26 July — for all eight candidates, uniformly. That uniformity was the clue:
a genuine emergence pattern does not align perfectly across unrelated topics.

The fix is to distinguish three states rather than two. A date can be uncollected, collected but
too sparse to establish absence, or usable:

```python
if day not in observed_days:
    values[day] = np.nan          # never sampled: unknown, not zero
elif len(day_docs) < min_documents_per_day:
    values[day] = np.nan          # too few to establish absence
else:
    values[day] = (day_docs["topic"] == topic_id).sum() / len(day_docs)
```

`min_documents_per_day = 30`. §17.1 varies it over 10, 20, 30 and 50; all four exclude exactly
25 July and nothing else, because no other observed date falls below 200 gymnastics documents.
The exclusion is therefore determined by the collection, not by the threshold.

### Expanding-window topic alignment — two wrong approaches before the right one (§11)

The problem: a model fitted on documents up to a cut-point produces its own topic numbering, and
its topics must be matched to the topics of the full-corpus model to say when a candidate first
becomes formable. Three alignment methods were tried.

**Term overlap, rejected.** Requiring two shared representative terms matched none of the eight
candidates at any cut-point. c-TF-IDF selects terms *relative to the rest of the corpus*, so a
smaller document pool selects different terms for the same cluster. The method measures a
property of the vocabulary selection, not of the cluster.

**Centroid similarity, rejected.** Averaging the embeddings of a topic's documents and comparing
by cosine matched all eight candidates at 0.81–0.93 — against only four distinct clusters. Mean
embeddings of document sets drawn from one corpus all sit close to the corpus mean, so the
similarities are high and nearly uninformative. Raising the threshold from 0.55 to 0.80 did not
separate them, which is the diagnostic that the measure itself was wrong.

**Document membership, adopted.** A topic is formable at a cut-point if the documents that the
full-corpus model assigned to it are concentrated in a single cluster of the cut-point model:

```python
members = set(gym[gym["topic"] == t]["clean"])
present = cut_frame[cut_frame["clean"].isin(members)]
if len(present) < minimum_documents:
    continue
assigned = present[present["cut_topic"] != -1]
largest = assigned["cut_topic"].value_counts().iloc[0]
share = largest / len(present)
if share >= concentration_threshold:          # 0.5
    visible.append(t)
```

This asks the question directly — do the same documents group together — and is invariant to
topic numbering and to term selection. It produced differentiated lags of 0–5 days.

Centroid similarity *is* used in §15.2, where both sides are full-corpus representations at
comparable granularity. The measure is not wrong in general; it was wrong for comparing a broad
model to a fine one.

### `HF_HUB_OFFLINE` set too late (§1)

The flag was set after the Hugging Face imports, where it has no effect. This is a silent
failure: the embedder still loads, but over the network, so an offline run fails at an unrelated
point. It is now set immediately after `import os`, before any other import. The ordering of the
first three lines of the notebook is load-bearing and is commented as such.

### Unseeded UMAP inside the model-selection sweep (§7)

The sweep let BERTopic construct its own UMAP, which does not receive `random_state`. The same
configuration therefore reported 36.3% outliers in the sweep and 37.7% in the comparison table.
Both numbers were correct for what they measured and neither was reproducible. A seeded `UMAP` is
now passed explicitly everywhere a model is fitted.

### Smaller fixes

| § | Defect | Fix |
| --- | --- | --- |
| 4 | `fit_transform` returns a list, so `labels == -1` evaluates to a single `bool` and `.sum()` raises | wrap in `np.array(...)` |
| 1 | `hdbscan.__version__` does not exist | `importlib.metadata.version()`, using *install* names (`scikit-learn`, `umap-learn`) |
| 16.1 | `KeyError: 'screen_flagged'` — the column was dropped when preparing the coders' copy and the copy was saved back over the master | derive it: `coded["topic"].isin(candidates).astype(int)` |
| 16 | A `Restart & Run All` regenerated the blank instrument over the labelled sheet | generator cell disabled; labelled sheet backed up outside the repository |
| 18 | `len(qualified) if "qualified" in dir()` fell through silently, so the summary reported 0 qualifying topics while §16.1 reported `[40]` | guard removed; the variable is always bound |
| 7 | Repeated kernel deaths fitting ~17 BERTopic models | explicit `del` and `gc.collect()` between fits; reduced scope samples |

---

## Part 2 — Choices, and what they cost

### The domain is selected from the evidence, not assumed (§4–§6)

An earlier version filtered to gymnastics first and then ran topic discovery, which assumes the
answer to part of RQ1. The committed order is: discovery on the full 261,467-document corpus
(§4, 291 topics), then derivation of the domain vocabulary from three indisputable seed terms
(§5), then filtering (§6). The gymnastics subset is 7,129 documents, 2.73% of the corpus. §4's
output is what justifies selecting it, and it is reported whether or not it flatters the choice.

### Three term classes, because six terms are genuinely ambiguous (§6)

`vault`, `bars`, `beam`, `ring`, `rings` and `floor` carry common non-gymnastics meanings — a
bank vault, a drinks bar, a beam of light, a wedding ring, a basketball floor. They are admitted
only alongside a context term. §6.3 inspects both decision outcomes: 45 documents accepted on an
ambiguous term plus context, 625 rejected on an ambiguous term without it, each printed so the
rule can be judged on cases rather than asserted.

`team`, `olympic` and `final` were originally in the context list and admitted 466 false
matches — boxing, basketball, pole vault. Removing them cut acceptances from 466 to 45.

### Relative prevalence rather than counts (§10)

Hours collected per date vary from 3 to 16, so absolute daily counts measure collection effort
as much as discourse. P(k,d) = n(k,d)/N(d) divides that out. The cost is that prevalence falls
when the overall volume of a date rises for unrelated reasons, which is stated in §10 rather than
hidden.

### Onset requires absence first (§12)

An onset is only meaningful if the topic was not already there. The criterion is 25% of the
topic's own observed maximum, as a fraction rather than an absolute value so that it does not
favour large topics:

```python
threshold = series.max() * 0.25
if series.iloc[0] >= threshold:
    onset[t] = None               # already elevated: onset precedes the window
```

§17.2 varies the fraction over 0.10, 0.25, 0.40 and 0.50. The same single candidate returns an
observable onset at every value. The finding — 7 of 8 unobservable — does not depend on the
number.

### Google Trends entities where a search term would be ambiguous (§13)

Topic 40 concerns the Peacock streaming service. The search term "peacock" is dominated by the
bird, so the topic is specified as the Trends entity `/g/11h582cpmh`. Entity IDs are opaque, so
`query_validity.csv` records for every query whether it denotes the topic's concept at all:

| Topic | Query | Validity | Reason |
| --- | --- | --- | --- |
| 71 | `leotards` | valid | denotes competitive gymnastics apparel |
| 22 | `unitards` | valid | denotes the specific garment discussed in the topic |
| 24 | `unitards` | valid | as above; duplicates the query for topic 22, so not an independent observation |
| 40 | `/g/11h582cpmh` | valid | Trends entity for the streaming service |
| 66 | `wearing` | invalid | generic verb, no consumer concept |
| 27 | `suits` | invalid | denotes business attire and legal proceedings |
| 50 | `trampoline` | ambiguous | denotes both the apparatus as a consumer product and the Olympic event |
| 18 | `career` | invalid | generic term, unrelated to any consumer concept |

Correspondence is reported for the four valid queries only. A series exceeding a threshold is not
evidence of correspondence where the query does not denote the topic. Two of the four are the
same query, so they are not four independent observations — stated in the report rather than
counted as four.

Responses are cached by query, window and geography. Trends normalises to the maximum of the
requested window, so an uncached rerun over a different window returns different values and the
reported numbers would not reproduce.

### Weighted kappa, and why raw agreement is reported next to it (§16.1)

The three labels are ordered, so `weights="linear"`: a 1–2 disagreement is less serious than a
0–2 one. Raw agreement 87.3% against weighted κ 0.461 is the kappa paradox (Feinstein &
Cicchetti, 1990) — agreement concentrated in the majority category. Agreement was 87% on
documents where either coder used label 0, 33% where either used label 2, and 29% for label 1.
Both figures are reported because either alone misleads.

### Wilson intervals on precision and recall (§16.4)

Five positive cases. A Wald interval on five successes is not usable, so
`proportion_confint(..., method="wilson")` is used. Recall 1.00, 95% interval [0.57, 1.00];
precision 0.05, interval [0.02, 0.12]. The intervals are the point: the screen is a nomination
device, and the report says so.

### The one place the result is *not* invariant (§16.3)

Every other sensitivity analysis returns "insensitive". This one does not, and is reported as it
stands:

| Resolution of the 19 disagreements | Qualifying topics |
| --- | --- |
| Rater 1's labels only | `[40]` |
| Rater 2's labels only | `[40, 18]` |
| Agreed labels only (adopted) | `[40]` |
| Optimistic (max of the two) | `[40, 18]` |
| Conservative (min of the two) | `[40]` |

Topic 40 qualifies under every resolution. Topic 18 qualifies under two of five, and its
qualification is therefore contingent on how disagreements are resolved. Disagreements are also
concentrated rather than dispersed: topic 18 lost 40% of its sampled documents to disagreement
and topic 24 lost 33%, while topics 27 and 50 lost none, so excluding disagreements shrinks the
denominators unequally. Topic 18's proportion rests on nine documents.

### BERTrend deviations (§15)

Reimplemented from Algorithm 1 rather than taken from the authors' release, which expects a
different corpus layout. Daily slices, merge at cosine ≥ 0.7 retaining the first-occurrence
embedding, popularity decayed as `p *= np.exp(-decay_lambda * gap**2)` with λ = 0.01. The weak-
signal rule is left to the implementer by the paper and is specified here as: below the 10th
percentile is noise, above the median is a strong signal, and between the two is a weak signal
only where an OLS slope over the window is positive. The window is 7 days — the shortest the nine
observed dates support — and the minimum slice cluster is 5, because daily slices are small. All
of this is in the comparison table in the README so that the baseline is not presented as a
faithful reproduction when it is a documented reimplementation.

---

## Part 3 — What was not done, and why

**The model did not label the documents.** Having the language model assign the 150 labels was
considered and declined. The screening rule in §9 was designed with the model's involvement, so
model labels would evaluate the screen against its own author, and a weighted κ between a coder
and the designer of the rule measures something other than inter-rater reliability. Both coders
are project members who did not design the screen.

**Disagreements were not resolved by discussion.** Adjudicating them would have produced a
single consensus label set and no measurable disagreement, which is the input to §16.3. The
agreed-label subset is used for qualification and the disagreements are analysed as data about
where the scheme is underspecified.

**The collection gap was not interpolated.** Filling 2–7 August would have produced a complete
series and an estimable onset for every topic. Every such number would have been manufactured.

**RQ3 was not answered.** Seven of eight onsets are unobservable and the single estimate
available comes from an ambiguous query. Reporting that estimate as the answer would have meant
presenting a −5 day lead derived from a query denoting both a consumer product and an Olympic
event. It is recorded in `lead_or_lag.csv` and excluded from the finding.
