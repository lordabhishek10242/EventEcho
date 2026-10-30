# Event headline classifier (NLP)

## Purpose

The problem statement asks to "categorize event headlines using NLP or a basic text classifier". EventEcho does this as a **cross-check** on the hand-assigned event categories. The event study still uses the hand-assigned categories; the classifier never changes them.

## Classes

Six classes, matching how the registry categories are grouped (`src/config.py`, `EVENT_CLASS_BY_CATEGORY`):

| Class | Registry categories |
|---|---|
| `monetary_policy` | Monetary Policy |
| `corporate_earnings` | Corporate Earnings |
| `government_policy` | Government Policy, Budget, Tax Reform, Public Health / Government Policy |
| `economic_data` | Economic Release |
| `election_politics` | Election / Political Event |
| `corporate_development` | Major Industry Event (deals, regulator actions, insolvency, allegations) |

## Training data

`results/headline_corpus.csv` holds **374 real headlines**. Each one has:

- a source URL;
- the web search that found it (49 search groups, 5 to 10 per class);
- the date it was collected (2026-10-29);
- a label, assigned by reading the headline.

The headlines are the titles of web search results. Junk titles (site names, ads, URL slugs) were left out, and exact duplicates were removed. Two URLs had broken tracking text at the end, which was trimmed. Headlines about the same event as a registry event are tagged in `registry_event_overlap`.

The corpus does **not** contain headlines written for this project, and no headline was invented.

## Model and baselines

- **Model:** TF-IDF features on words (1–2 grams) and characters (3–5 grams), then logistic regression with balanced class weights. All settings were fixed before testing and were not tuned on either test.
- **Keyword rules:** a hand-written list of words per class. **These rules were written after reading the corpus**, so their scores are optimistic and are not a blind test. Words specific to registry events (for example a company or report name) were removed to limit this bias.
- **Majority class:** always predict the most common class. Any useful model must beat it.

## How it is tested

1. **Grouped cross-validation (5 folds):** whole search groups are held out at once. A group is mostly about one event family (for example "RBI cuts in 2019", or "Yes Bank moratorium"), so this measures how the model does on event families it has not seen.
2. **External test on the 21 registry headlines:** these headlines are short and formal, unlike news headlines, and are never used for training. Corpus headlines about any registry event are removed from training for this test. The 95% interval is a bootstrap over the 21 headlines.

## Results (run 40915721578; unchanged in run 40924836126)

| Test | Model | Accuracy | 95% CI | Macro F1 |
|---|---|---:|---:|---:|
| Grouped CV (374 headlines, 49 groups) | TF-IDF + logistic regression | 75% | 71–80% | 0.75 |
| Grouped CV | Keyword rules (not blind) | 89% | 86–92% | 0.92 |
| Grouped CV | Majority class | 23% | 19–27% | 0.06 |
| Registry headlines (21) | TF-IDF + logistic regression | 81% | 62–95% | 0.79 |
| Registry headlines | Keyword rules (not blind) | 86% | 71–100% | 0.79 |
| Registry headlines | Majority class | 29% | 10–48% | 0.07 |

The model disagrees with the hand-assigned class on 4 registry headlines: demonetisation (EVT001), the COVID lockdown (EVT002), the GST rollout (EVT015) and the HDFC Bank results (EVT012). Its probability is low on all four (0.30–0.46). The weakest class is `government_policy` (F1 0.56): one-off policy shocks such as demonetisation or a lockdown use words that are rare in the training headlines.

## What this does and does not show

- A simple, transparent text model labels unseen event headlines far better than guessing, with honest uncertainty.
- Hand-written keyword rules are competitive on this small, clean task. That is a useful, honest finding, not a weakness to hide.
- The classifier does not affect any return, CAR or significance result.
- Search-result titles are not a random sample of all news. Some sources appear more than once, and the labels were assigned by one person, with no second annotator.

## Reproduce

The classifier runs inside `scripts/run_analysis.py`. Its outputs are `headline_classifier_metrics.csv`, `headline_classifier_per_class.csv`, `headline_classifier_confusion.csv` and `event_headline_classification.csv`. The corpus checksum is recorded in the run manifest (`headline_corpus_sha256`).
