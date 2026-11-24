"""Hand-written result tables (README, model card) must agree with reports/metrics.json."""

import json

import pytest

from satelliteguard import PROJECT_ROOT

M = json.loads((PROJECT_ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
README = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
CARD = (PROJECT_ROOT / "docs" / "model_card.md").read_text(encoding="utf-8")


def row(doc: str, label: str) -> str:
    """The first Markdown table row whose first cell contains `label`."""
    for line in doc.splitlines():
        if line.startswith("|") and label in line.split("|")[1]:
            return line
    raise AssertionError(f"no table row for {label!r}")


def cv(m):
    f = M["cv"][m]["cv_folds"]
    return f"{f['pr_auc_mean']:.3f} ± {f['pr_auc_std']:.3f}"


def tval(m, key):
    return M["test"][m]["overall"][key]


def ci(m, digits):
    lo, hi = M["test"][m]["bootstrap"]["pr_auc_ci95"]
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


README_ROWS = {"Channel only": "channel_only", "Logistic regression": "logreg",
               "Gradient boosting": "hgb", "PCA reconstruction": "pca_recon",
               "Isolation Forest": "iforest"}


@pytest.mark.parametrize("label,m", README_ROWS.items())
def test_readme_results_row_matches_metrics(label, m):
    r = row(README, label)
    for expected in [cv(m), f"{tval(m, 'pr_auc'):.3f} {ci(m, 2)}",
                     f"{tval(m, 'precision'):.2f} / {tval(m, 'recall'):.2f}"]:
        assert expected in r.replace("**", ""), (expected, r)


def test_readme_prevalence_row():
    r = row(README, "Prevalence")
    assert f"{M['cv']['prevalence']['cv_folds']['pr_auc_mean']:.3f}" in r
    assert f"{tval('prevalence', 'pr_auc'):.3f}" in r


CARD_ROWS = {"Gradient boosting": "hgb", "Logistic regression": "logreg",
             "PCA reconstruction": "pca_recon", "Channel-only": "channel_only"}


@pytest.mark.parametrize("label,m", CARD_ROWS.items())
def test_model_card_row_matches_metrics(label, m):
    r = row(CARD, label).replace("**", "")
    expected = [cv(m), f"{tval(m, 'pr_auc'):.3f} {ci(m, 3)}", f"{tval(m, 'precision'):.2f}",
                f"{tval(m, 'recall'):.2f}"]
    if "brier" in M["test"][m]["overall"]:
        expected.append(f"{tval(m, 'brier'):.3f}")
    for e in expected:
        assert e in r, (e, r)


@pytest.mark.parametrize("ch", ["CADC0872", "CADC0873", "CADC0874", "CADC0888", "CADC0892",
                                "CADC0894"])
def test_model_card_per_channel_matches_metrics(ch):
    pc = M["test"]["hgb"]["per_channel"][ch]
    r = row(CARD, ch).replace("**", "")
    for e in [f"| {pc['n']} |", f"{pc['prevalence']:.2f}", f"{pc['pr_auc']:.3f}",
              f"{pc['precision']:.2f}", f"{pc['recall']:.2f}"]:
        assert e in r, (e, r)


def test_model_card_threshold_matches_policy():
    assert f"{M['cv']['hgb']['threshold_policy']['threshold']:.3f}" in CARD
