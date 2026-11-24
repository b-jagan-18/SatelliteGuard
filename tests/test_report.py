"""The technical report is generated from reports/*.json and must not drift from it."""

import pytest

from satelliteguard import report


def test_placeholders_format_and_fail_loudly():
    ctx = {"a": {"b": 0.12345, "ci": [0.1, 0.2]}, "n": 106}
    out = report.render_template("{{a.b:.3f}} [{{ a.ci.0:.2f }}, {{a.ci.1:.2f}}] {{n}} {{a.b:.1%}}",
                                 ctx)
    assert out == "0.123 [0.10, 0.20] 106 12.3%"
    with pytest.raises(KeyError):
        report.render_template("{{a.missing}}", ctx)


def test_report_template_resolves_against_committed_results():
    md = report.render_template(report.TEMPLATE.read_text(encoding="utf-8"), report.context())
    assert "{{" not in md
    test = report.context()["test"]["hgb"]["overall"]
    assert f"{test['pr_auc']:.3f}" in md  # headline number comes from metrics.json
    html = report.to_html(md, "t")
    assert html.count("<figure>") + html.count('<figure class="tall">') >= 20
    assert "<table>" in html
