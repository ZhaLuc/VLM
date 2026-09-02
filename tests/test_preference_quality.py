"""Exact-count tests for preference quality analysis."""

from __future__ import annotations

from pathlib import Path

from magic_vlm.preference_quality import (
    QualitySeverity,
    analyze_preference_file,
    analyze_preference_quality,
    format_quality_report,
    load_preferences_with_malformed,
    write_quality_outputs,
)
from magic_vlm.preferences import build_preference_pair, write_preference_pairs
from magic_vlm.schemas import PreferenceGenerationMeta, Provenance, Split, TaskType


FIXTURE = Path("tests/fixtures/preference_quality/synthetic_quality.jsonl")
FIXTURE_MALFORMED = Path(
    "tests/fixtures/preference_quality/synthetic_quality_with_malformed.jsonl"
)


def _gen() -> PreferenceGenerationMeta:
    return PreferenceGenerationMeta(model_id_a="stub", model_id_b="stub")


def test_synthetic_fixture_counts() -> None:
    pairs, malformed, n_lines = load_preferences_with_malformed(FIXTURE)
    assert malformed == []
    assert n_lines == 13
    assert len(pairs) == 13

    report = analyze_preference_quality(pairs, malformed=malformed, n_lines=n_lines)
    assert report.n_parsed == 13
    assert report.n_malformed == 0
    assert report.winner_counts == {"a": 11, "b": 2}
    assert report.winner_balance["n_a"] == 11
    assert report.winner_balance["n_b"] == 2
    assert report.winner_balance["abs_imbalance"] == 9
    assert report.n_unique_pair_ids == 11
    assert report.n_duplicate_pair_id_groups == 2
    assert report.n_contradictory_pair_ids == 1
    assert report.agreement.n_annotators == 2
    assert report.agreement.n_pairs_with_repeated_judgments == 2
    assert report.agreement.n_pairs_unanimous == 1
    assert report.agreement.n_pairs_contradictory == 1
    assert report.agreement.agreement_rate == 0.5
    assert report.agreement.can_estimate_inter_rater_reliability is True
    assert report.n_exact_response_duplicate_groups >= 1
    assert report.integrity["filtering_applied"] is False
    assert report.integrity["no_records_deleted"] is True
    assert report.integrity["winner_counts_match_parsed"] is True

    codes = {f.code: f for f in report.findings}
    assert "winner_imbalance" in codes
    assert "duplicate_pair_ids" in codes
    assert "contradictory_labels" in codes
    assert "identical_responses_within_pair" in codes
    assert codes["identical_responses_within_pair"].count == 1
    assert "exact_response_duplicates" in codes
    assert "possible_length_bias" in codes
    assert codes["possible_length_bias"].severity is QualitySeverity.BIAS
    assert "possible_confidence_format_bias" in codes
    assert report.clip_counts["c1"] == 2
    assert report.task_counts == {"explanation": 13}
    assert report.trick_counts["cups"] >= 1
    assert report.annotator_counts["ann1"] + report.annotator_counts["ann2"] == 13


def test_malformed_records_counted_not_deleted() -> None:
    pairs, malformed, n_lines = load_preferences_with_malformed(FIXTURE_MALFORMED)
    assert n_lines == 15
    assert len(pairs) == 13
    assert len(malformed) == 2
    report = analyze_preference_quality(pairs, malformed=malformed, n_lines=n_lines)
    assert report.n_malformed == 2
    assert report.integrity["counts_match"] is True
    assert any(f.code == "malformed_records" for f in report.findings)
    assert any(f.severity is QualitySeverity.ERROR for f in report.findings)


def test_balanced_labels_fixture(tmp_path: Path) -> None:
    gen = _gen()
    prov = Provenance(source="bal")
    rows = []
    for i in range(4):
        rows.append(
            build_preference_pair(
                clip_id=f"b{i}",
                instruction="q",
                response_a=f"a{i}",
                response_b=f"b{i}",
                winner="a" if i % 2 == 0 else "b",
                annotator_id="solo",
                timestamp=f"2026-09-02T13:0{i}:00+00:00",
                provenance=prov,
                generation_meta=gen,
                task=TaskType.EXPLANATION,
                split=Split.TRAIN,
            )
        )
    path = tmp_path / "bal.jsonl"
    write_preference_pairs(path, rows)
    report = analyze_preference_file(path)
    assert report.winner_counts == {"a": 2, "b": 2}
    assert report.winner_balance["balanced"] is True
    assert report.agreement.n_annotators == 1
    assert report.agreement.can_estimate_inter_rater_reliability is False
    assert any(f.code == "irr_unavailable" for f in report.findings)
    assert "inter-rater" in report.agreement.caveat.lower() or "IRR" in report.agreement.caveat or "annotator" in report.agreement.caveat.lower()


def test_length_bias_hand_count(tmp_path: Path) -> None:
    gen = _gen()
    prov = Provenance(source="len")
    long = "x" * 100
    short = "y"
    rows = [
        build_preference_pair(
            clip_id=f"l{i}",
            instruction="q",
            response_a=long,
            response_b=short,
            winner="a",
            annotator_id="solo",
            timestamp=f"2026-09-02T14:0{i}:00+00:00",
            provenance=prov,
            generation_meta=gen,
        )
        for i in range(4)
    ]
    path = tmp_path / "len.jsonl"
    write_preference_pairs(path, rows)
    report = analyze_preference_file(path)
    assert report.length.n == 4
    assert report.length.n_preferred_longer == 4
    assert report.length.frac_preferred_longer == 1.0
    assert any(f.code == "possible_length_bias" for f in report.findings)


def test_write_outputs_and_report_language(tmp_path: Path) -> None:
    report = analyze_preference_file(FIXTURE_MALFORMED)
    paths = write_quality_outputs(report, tmp_path)
    assert paths["metrics"].exists()
    text = format_quality_report(report)
    assert "never deletes" in text.lower() or "Caveats" in text
    assert "reward hacking" in text.lower()
    assert "malformed" in text.lower()


def test_cli_analyze_preferences(tmp_path: Path) -> None:
    from magic_vlm.cli import analyze_preferences_main

    code = analyze_preferences_main(
        [
            "--prefs",
            str(FIXTURE_MALFORMED),
            "--out-dir",
            str(tmp_path / "qc"),
        ]
    )
    assert code == 0
    assert (tmp_path / "qc" / "preference_quality.json").exists()
    assert (tmp_path / "qc" / "preference_quality_report.md").exists()


def test_toy_annotated_dataset_findings() -> None:
    report = analyze_preference_file("data/examples/toy_annotated_preferences.jsonl")
    assert report.n_parsed == 2
    assert report.n_malformed == 0
    assert report.winner_counts.get("a") == 2
    assert report.agreement.n_annotators == 1
    assert report.agreement.can_estimate_inter_rater_reliability is False
    assert report.integrity["filtering_applied"] is False
