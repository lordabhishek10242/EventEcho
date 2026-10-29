import pandas as pd
import pytest

from src.headline_classifier import CLASSES, keyword_classify, run_headline_classification, validate_corpus

TEMPLATES = {
    "monetary_policy": ["RBI cuts repo rate by {n} bps", "MPC keeps policy rate unchanged at {n}%", "Central bank hikes repo rate {n} basis points"],
    "corporate_earnings": ["Firm{n} Q2 net profit rises", "Company{n} quarterly results beat estimates", "Firm{n} Q4 revenue and profit fall"],
    "government_policy": ["Budget cuts tax slab {n}", "Cabinet approves scheme worth Rs {n} crore", "GST council revises rates on {n} items"],
    "economic_data": ["GDP grows {n}% in quarter", "Retail inflation eases to {n}%", "Trade deficit widens to ${n} billion"],
    "election_politics": ["Election results: party wins {n} seats", "Exit polls predict majority {n}", "Assembly poll counting day {n}"],
    "corporate_development": ["Firm{n} announces merger", "Regulator bars promoters of firm{n}", "Firm{n} files for insolvency"],
}


def _corpus() -> pd.DataFrame:
    rows = []
    for label, templates in TEMPLATES.items():
        for group in range(2):
            for t, template in enumerate(templates):
                n = group * 10 + t
                rows.append({"label": label, "headline": template.format(n=n), "source_url": f"https://example.org/{label}/{n}",
                             "search_group": f"{label}-{group}", "registry_event_overlap": ""})
    frame = pd.DataFrame(rows)
    frame["headline_id"] = [f"H{i:04d}" for i in range(len(frame))]
    return frame


EVENTS = pd.DataFrame({
    "event_id": ["E1", "E2", "E3"],
    "headline": ["RBI raises repo rate by 25 basis points", "Company Q4 earnings release", "GDP estimates released"],
    "category": ["Monetary Policy", "Corporate Earnings", "Economic Release"],
})


def test_corpus_validation_rejects_duplicates_unknown_labels_and_missing_sources():
    corpus = _corpus()
    with pytest.raises(ValueError, match="duplicate"):
        validate_corpus(pd.concat([corpus, corpus.iloc[[0]].assign(headline_id="X")]))
    with pytest.raises(ValueError, match="Unknown"):
        validate_corpus(corpus.assign(label=corpus["label"].where(corpus.index > 0, "other")))
    with pytest.raises(ValueError, match="source URL"):
        validate_corpus(corpus.assign(source_url=corpus["source_url"].where(corpus.index > 0, "")))


def test_keyword_rules_return_unclassified_without_hits():
    result = keyword_classify(pd.Series(["RBI cuts repo rate", "Something unrelated happened"]))
    assert list(result) == ["monetary_policy", "unclassified"]


def test_registry_event_headlines_are_excluded_from_training():
    corpus = _corpus()
    corpus.loc[0, "registry_event_overlap"] = "E1"
    outputs = run_headline_classification(corpus, EVENTS)
    metrics = outputs["headline_classifier_metrics.csv"]
    external = metrics[metrics["evaluation"] == "external_test_on_registry_headlines"]
    assert set(external["training_headlines"]) == {len(corpus) - 1}
    assert set(external["headlines"]) == {3}
    predictions = outputs["event_headline_classification.csv"]
    assert list(predictions["event_id"]) == ["E1", "E2", "E3"]
    assert set(predictions["predicted_class"]) <= set(CLASSES)
    assert predictions.filter(like="probability_").sum(axis=1).round(6).eq(1).all()


def test_committed_corpus_is_sourced_balanced_and_grouped():
    corpus = validate_corpus(pd.read_csv("results/headline_corpus.csv"))
    counts = corpus["label"].value_counts()
    assert set(counts.index) == set(CLASSES)
    assert counts.min() >= 40
    assert corpus.groupby("label")["search_group"].nunique().min() >= 5
    assert corpus["source_url"].str.startswith("http").all()
