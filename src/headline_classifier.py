"""Event-headline classifier: keyword baseline vs TF-IDF + logistic regression.

The classifier answers one narrow question: given only a headline, which of six
event classes does it belong to? It is trained on ``results/headline_corpus.csv``
(real headlines collected from web search results, each with its source link)
and evaluated two ways:

1. Grouped cross-validation. Headlines are grouped by the search that found
   them, and a whole group is held out at once. Each group is mostly about one
   event family, so this tests whether the model works on events it has not
   seen, not just on near-copies of training headlines.
2. An external test on the project's own 21 registry headlines. These are
   written in a different style and are never used for training. Corpus
   headlines that describe one of the registry events are removed from the
   training set for this test, so the model cannot just recognise the event.

Hyperparameters are fixed in advance and are not tuned on either test. The
classifier is a cross-check on the hand-assigned categories; it does not
replace them in the event study.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .config import EVENT_CLASS_BY_CATEGORY

CLASSES = (
    "monetary_policy",
    "corporate_earnings",
    "government_policy",
    "economic_data",
    "election_politics",
    "corporate_development",
)

# Transparent baseline: count keyword hits per class. Order breaks ties.
# These rules were written by hand after reading the corpus, so their scores
# are optimistic: they are not a blind test. Words specific to the registry
# events (for example a company or report name) are deliberately excluded.
KEYWORD_RULES: dict[str, tuple[str, ...]] = {
    "monetary_policy": (
        r"\brepo\b", r"\bmpc\b", r"monetary policy", r"\bcrr\b", r"rate cut", r"rate hike",
        r"\bbps\b", r"basis points?", r"accommodat", r"\bstance\b", r"policy rate", r"interest rates?",
    ),
    "corporate_earnings": (
        r"\bq[1-4]\b", r"\bq[1-4]fy", r"quarter", r"net profit", r"\bprofit\b", r"\bresults?\b",
        r"earnings", r"revenue", r"guidance", r"\bebitda\b",
    ),
    "government_policy": (
        r"budget", r"\btax", r"\bgst\b", r"fiscal", r"cabinet", r"scheme", r"\bpli\b", r"stimulus",
        r"package", r"\bduty\b", r"slabs?", r"\bgovt\b", r"government", r"policy",
    ),
    "economic_data": (
        r"\bgdp\b", r"inflation", r"\bcpi\b", r"\bwpi\b", r"\biip\b", r"industrial production",
        r"\bpmi\b", r"trade deficit", r"exports", r"imports", r"unemployment", r"\bplfs\b",
        r"collections", r"\bdata\b", r"estimates?",
    ),
    "election_politics": (
        r"election", r"\bpolls?\b", r"exit poll", r"lok sabha", r"assembly", r"\bbjp\b", r"congress",
        r"\bnda\b", r"\bvote", r"counting", r"sworn", r"\boath\b", r"mandate",
    ),
    "corporate_development": (
        r"merger", r"merge", r"acquir", r"acquisition", r"\bstake\b", r"\bsebi\b", r"insolvency",
        r"bankrupt", r"\bnclt\b", r"fraud", r"briber", r"indict", r"moratorium",
        r"verdict", r"\bbars?\b", r"crash", r"plunge", r"tank",
    ),
}

RANDOM_SEED = 2026
BOOTSTRAP_DRAWS = 5000


def keyword_classify(headlines: pd.Series) -> pd.Series:
    """Return the class with the most keyword hits, or 'unclassified'."""
    compiled = {label: [re.compile(p, re.IGNORECASE) for p in patterns] for label, patterns in KEYWORD_RULES.items()}

    def one(text: str) -> str:
        scores = {label: sum(bool(p.search(text)) for p in patterns) for label, patterns in compiled.items()}
        best = max(scores.values())
        if best == 0:
            return "unclassified"
        return next(label for label in CLASSES if scores[label] == best)

    return headlines.fillna("").astype(str).map(one)


def build_model():
    """TF-IDF on words (1-2 grams) and characters (3-5 grams) + logistic regression."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline

    features = FeatureUnion([
        ("words", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
        ("chars", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)),
    ])
    classifier = LogisticRegression(C=4.0, class_weight="balanced", max_iter=5000)
    return Pipeline([("features", features), ("classifier", classifier)])


def _scores(true: pd.Series, predicted: pd.Series) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    labels = sorted(set(true))
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "macro_f1": float(f1_score(true, predicted, labels=labels, average="macro", zero_division=0)),
    }


def _bootstrap_accuracy(correct: np.ndarray, seed: int) -> tuple[float, float]:
    if len(correct) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(correct.astype(float), size=(BOOTSTRAP_DRAWS, len(correct)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def validate_corpus(corpus: pd.DataFrame) -> pd.DataFrame:
    required = {"headline_id", "label", "headline", "source_url", "search_group", "registry_event_overlap"}
    missing = required - set(corpus.columns)
    if missing:
        raise ValueError(f"Headline corpus is missing columns: {sorted(missing)}")
    frame = corpus.copy()
    if frame["headline_id"].duplicated().any():
        raise ValueError("Headline IDs must be unique")
    if frame["headline"].str.casefold().duplicated().any():
        raise ValueError("Headline corpus contains duplicate headlines")
    unknown = sorted(set(frame["label"]) - set(CLASSES))
    if unknown:
        raise ValueError(f"Unknown headline labels: {unknown}")
    if frame["source_url"].fillna("").str.strip().eq("").any():
        raise ValueError("Every corpus headline needs a source URL")
    frame["registry_event_overlap"] = frame["registry_event_overlap"].fillna("").astype(str).str.strip()
    return frame


def registry_classes(events: pd.DataFrame) -> pd.Series:
    unknown = sorted(set(events["category"]) - set(EVENT_CLASS_BY_CATEGORY))
    if unknown:
        raise ValueError(f"No event class mapping for registry categories: {unknown}")
    return events["category"].map(EVENT_CLASS_BY_CATEGORY)


def run_headline_classification(corpus: pd.DataFrame, events: pd.DataFrame, n_splits: int = 5) -> dict[str, pd.DataFrame]:
    """Evaluate the baselines and the model; classify the registry headlines."""
    from sklearn.metrics import precision_recall_fscore_support
    from sklearn.model_selection import GroupKFold

    data = validate_corpus(corpus)
    texts, labels, groups = data["headline"], data["label"], data["search_group"]
    majority = labels.value_counts().sort_index().idxmax()

    # 1. Grouped cross-validation: whole search groups are held out.
    out_of_fold = pd.Series(index=data.index, dtype=object)
    for train_idx, test_idx in GroupKFold(n_splits=n_splits).split(texts, labels, groups):
        model = build_model().fit(texts.iloc[train_idx], labels.iloc[train_idx])
        out_of_fold.iloc[test_idx] = model.predict(texts.iloc[test_idx])
    keyword_cv = keyword_classify(texts)
    majority_cv = pd.Series(majority, index=data.index)

    # 2. External test on the registry, training without same-event headlines.
    truth = registry_classes(events).reset_index(drop=True)
    registry_ids = set(events["event_id"].astype(str))
    train = data[~data["registry_event_overlap"].isin(registry_ids)]
    final_model = build_model().fit(train["headline"], train["label"])
    registry_text = events["headline"].astype(str).reset_index(drop=True)
    probabilities = final_model.predict_proba(registry_text)
    predicted = pd.Series(final_model.classes_[probabilities.argmax(axis=1)])
    keyword_registry = keyword_classify(registry_text)
    majority_registry = pd.Series(train["label"].value_counts().sort_index().idxmax(), index=truth.index)

    metrics: list[dict] = []
    for evaluation, true, rows, n_groups in (
        ("grouped_cross_validation_on_corpus", labels, (("tfidf_logistic_regression", out_of_fold), ("keyword_rules", keyword_cv), ("majority_class", majority_cv)), groups.nunique()),
        ("external_test_on_registry_headlines", truth, (("tfidf_logistic_regression", predicted), ("keyword_rules", keyword_registry), ("majority_class", majority_registry)), len(truth)),
    ):
        for offset, (model_name, prediction) in enumerate(rows):
            correct = (prediction.to_numpy() == true.to_numpy())
            lower, upper = _bootstrap_accuracy(correct, RANDOM_SEED + offset)
            metrics.append({
                "evaluation": evaluation,
                "fair_blind_test": model_name != "keyword_rules",
                "model": model_name,
                "headlines": int(len(true)),
                "held_out_units": int(n_groups),
                "training_headlines": int(len(train)) if evaluation.startswith("external") else int(len(data)),
                **_scores(true, prediction),
                "accuracy_ci95_lower": lower,
                "accuracy_ci95_upper": upper,
                "note": (
                    "Keyword rules were written after reading the corpus, so this score is optimistic. "
                    if model_name == "keyword_rules" else ""
                ) + (
                    "Whole search groups held out (GroupKFold); groups are event families."
                    if evaluation.startswith("grouped") else
                    "Trained on the corpus minus headlines about any registry event; registry headlines never used in training. CI by bootstrap over the 21 headlines."
                ),
            })

    precision, recall, f1, support = precision_recall_fscore_support(labels, out_of_fold, labels=list(CLASSES), zero_division=0)
    per_class = pd.DataFrame({
        "evaluation": "grouped_cross_validation_on_corpus",
        "event_class": CLASSES,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support.astype(int),
    })
    confusion = (
        pd.DataFrame({"true_class": labels.to_numpy(), "predicted_class": out_of_fold.to_numpy()})
        .value_counts().rename("count").reset_index()
        .assign(evaluation="grouped_cross_validation_on_corpus")
        .sort_values(["true_class", "predicted_class"]).reset_index(drop=True)
    )
    event_predictions = pd.DataFrame({
        "event_id": events["event_id"].astype(str).to_numpy(),
        "headline": registry_text,
        "registry_category": events["category"].to_numpy(),
        "assigned_class": truth,
        "predicted_class": predicted,
        "predicted_probability": probabilities.max(axis=1),
        "keyword_rule_class": keyword_registry,
    })
    event_predictions["model_agrees_with_assigned_class"] = event_predictions["predicted_class"] == event_predictions["assigned_class"]
    for index, label in enumerate(final_model.classes_):
        event_predictions[f"probability_{label}"] = probabilities[:, index]
    return {
        "headline_classifier_metrics.csv": pd.DataFrame(metrics),
        "headline_classifier_per_class.csv": per_class,
        "headline_classifier_confusion.csv": confusion[["evaluation", "true_class", "predicted_class", "count"]],
        "event_headline_classification.csv": event_predictions,
    }
