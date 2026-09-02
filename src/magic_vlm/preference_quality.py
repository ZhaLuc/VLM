"""Preference-data quality analysis (pre-DPO / pre-reward-model QC).

Reports hard errors, quality warnings, and *possible* biases. Does **not**
delete, filter, or rewrite records. A correlation is not evidence of reward
hacking; inter-annotator reliability is only estimated when multiple annotators
exist.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

from magic_vlm.preferences import (
    compute_content_pair_id,
    group_judgments_by_pair,
    load_preference_pairs,
)
from magic_vlm.schemas import PreferencePair, SchemaError
from magic_vlm.utils import write_json, write_jsonl

CONFIDENCE_MARKERS = (
    "certainly",
    "definitely",
    "obviously",
    "clearly",
    "without a doubt",
    "i am sure",
    "i'm sure",
    "confidence alone",
)
MARKDOWNISH_PATTERNS = (
    re.compile(r"(?m)^\s{0,3}#{1,6}\s"),
    re.compile(r"(?m)^\s*[-*+]\s+\S"),
    re.compile(r"(?m)^\s*\d+\.\s+\S"),
    re.compile(r"\*\*[^*]+\*\*"),
    re.compile(r"```"),
)


class QualitySeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    BIAS = "possible_bias"
    INFO = "info"


@dataclass(frozen=True)
class QualityFinding:
    severity: QualitySeverity
    code: str
    message: str
    count: int = 1
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity.value,
            "code": self.code,
            "message": self.message,
            "count": self.count,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class MalformedRecord:
    line_no: int
    error: str
    raw_text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LengthStats:
    n: int
    mean_chars_a: float | None
    mean_chars_b: float | None
    mean_chars_preferred: float | None
    mean_chars_rejected: float | None
    n_preferred_longer: int
    n_preferred_shorter: int
    n_preferred_equal_len: int
    frac_preferred_longer: float | None
    mean_char_delta_preferred_minus_rejected: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgreementStats:
    n_annotators: int
    n_pairs_with_repeated_judgments: int
    n_pairs_unanimous: int
    n_pairs_contradictory: int
    agreement_rate: float | None
    can_estimate_inter_rater_reliability: bool
    caveat: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PreferenceQualityReport:
    """Machine-readable preference QC summary (non-destructive)."""

    n_lines: int
    n_parsed: int
    n_malformed: int
    winner_counts: dict[str, int]
    winner_balance: dict[str, Any]
    n_unique_pair_ids: int
    n_duplicate_pair_id_groups: int
    n_exact_response_duplicate_groups: int
    n_contradictory_pair_ids: int
    agreement: AgreementStats
    length: LengthStats
    formatting: dict[str, Any]
    clip_counts: dict[str, int]
    task_counts: dict[str, int]
    trick_counts: dict[str, int]
    annotator_counts: dict[str, int]
    findings: tuple[QualityFinding, ...]
    malformed: tuple[MalformedRecord, ...]
    integrity: dict[str, Any]
    caveats: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_lines": self.n_lines,
            "n_parsed": self.n_parsed,
            "n_malformed": self.n_malformed,
            "winner_counts": dict(self.winner_counts),
            "winner_balance": dict(self.winner_balance),
            "n_unique_pair_ids": self.n_unique_pair_ids,
            "n_duplicate_pair_id_groups": self.n_duplicate_pair_id_groups,
            "n_exact_response_duplicate_groups": self.n_exact_response_duplicate_groups,
            "n_contradictory_pair_ids": self.n_contradictory_pair_ids,
            "agreement": self.agreement.to_dict(),
            "length": self.length.to_dict(),
            "formatting": dict(self.formatting),
            "clip_counts": dict(self.clip_counts),
            "task_counts": dict(self.task_counts),
            "trick_counts": dict(self.trick_counts),
            "annotator_counts": dict(self.annotator_counts),
            "findings": [f.to_dict() for f in self.findings],
            "malformed": [m.to_dict() for m in self.malformed],
            "integrity": dict(self.integrity),
            "caveats": list(self.caveats),
            "n_errors": sum(1 for f in self.findings if f.severity is QualitySeverity.ERROR),
            "n_warnings": sum(
                1 for f in self.findings if f.severity is QualitySeverity.WARNING
            ),
            "n_possible_biases": sum(
                1 for f in self.findings if f.severity is QualitySeverity.BIAS
            ),
        }

    @property
    def n_errors(self) -> int:
        return sum(1 for f in self.findings if f.severity is QualitySeverity.ERROR)

    @property
    def n_warnings(self) -> int:
        return sum(1 for f in self.findings if f.severity is QualitySeverity.WARNING)


def load_preferences_with_malformed(
    path: str | Path,
) -> tuple[list[PreferencePair], list[MalformedRecord], int]:
    """Load preferences; capture malformed lines without dropping them from the report."""
    pairs: list[PreferencePair] = []
    malformed: list[MalformedRecord] = []
    n_lines = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            n_lines += 1
            try:
                payload = json.loads(text)
                pairs.append(PreferencePair.from_dict(payload))
            except (json.JSONDecodeError, SchemaError, TypeError, ValueError, KeyError) as exc:
                malformed.append(
                    MalformedRecord(line_no=line_no, error=str(exc), raw_text=text)
                )
    return pairs, malformed, n_lines


def _char_len(text: str) -> int:
    return len(text)


def _has_confidence_marker(text: str) -> bool:
    lower = text.lower()
    return any(marker in lower for marker in CONFIDENCE_MARKERS)


def _has_markdownish_format(text: str) -> bool:
    return any(pat.search(text) for pat in MARKDOWNISH_PATTERNS)


def _length_stats(pairs: Sequence[PreferencePair]) -> LengthStats:
    usable = [p for p in pairs if p.winner in {"a", "b"}]
    if not usable:
        return LengthStats(
            n=0,
            mean_chars_a=None,
            mean_chars_b=None,
            mean_chars_preferred=None,
            mean_chars_rejected=None,
            n_preferred_longer=0,
            n_preferred_shorter=0,
            n_preferred_equal_len=0,
            frac_preferred_longer=None,
            mean_char_delta_preferred_minus_rejected=None,
        )
    lens_a = [_char_len(p.response_a) for p in usable]
    lens_b = [_char_len(p.response_b) for p in usable]
    pref_lens: list[int] = []
    rej_lens: list[int] = []
    longer = shorter = equal = 0
    deltas: list[int] = []
    for p in usable:
        chosen, rejected = p.chosen_rejected()
        cl, rl = _char_len(chosen), _char_len(rejected)
        pref_lens.append(cl)
        rej_lens.append(rl)
        deltas.append(cl - rl)
        if cl > rl:
            longer += 1
        elif cl < rl:
            shorter += 1
        else:
            equal += 1
    n = len(usable)
    return LengthStats(
        n=n,
        mean_chars_a=sum(lens_a) / n,
        mean_chars_b=sum(lens_b) / n,
        mean_chars_preferred=sum(pref_lens) / n,
        mean_chars_rejected=sum(rej_lens) / n,
        n_preferred_longer=longer,
        n_preferred_shorter=shorter,
        n_preferred_equal_len=equal,
        frac_preferred_longer=(longer / n) if n else None,
        mean_char_delta_preferred_minus_rejected=(sum(deltas) / n) if n else None,
    )


def _agreement_stats(pairs: Sequence[PreferencePair]) -> AgreementStats:
    annotators = {p.annotator_id for p in pairs}
    grouped = group_judgments_by_pair(pairs)
    multi = {pid: g for pid, g in grouped.items() if len(g) > 1}
    unanimous = 0
    contradictory = 0
    for group in multi.values():
        winners = {g.winner for g in group}
        if len(winners) == 1:
            unanimous += 1
        else:
            contradictory += 1
    n_multi = len(multi)
    rate = (unanimous / n_multi) if n_multi else None
    can_irr = len(annotators) >= 2 and n_multi >= 1
    if len(annotators) < 2:
        caveat = (
            "Only one unique annotator_id is present; inter-rater reliability "
            "cannot be estimated. Repeated judgments by one person are not IRR."
        )
    elif n_multi == 0:
        caveat = (
            "Multiple annotators exist but no content pair has repeated judgments; "
            "agreement is undefined."
        )
    else:
        caveat = (
            "Agreement rate is the fraction of multi-annotated content pairs with "
            "unanimous winners. Disagreement is not automatically annotation error. "
            "This is not a full reliability coefficient."
        )
    return AgreementStats(
        n_annotators=len(annotators),
        n_pairs_with_repeated_judgments=n_multi,
        n_pairs_unanimous=unanimous,
        n_pairs_contradictory=contradictory,
        agreement_rate=rate,
        can_estimate_inter_rater_reliability=can_irr,
        caveat=caveat,
    )


def _formatting_analysis(pairs: Sequence[PreferencePair]) -> dict[str, Any]:
    usable = [p for p in pairs if p.winner in {"a", "b"}]
    n = len(usable)
    if n == 0:
        return {
            "n": 0,
            "n_preferred_has_confidence_marker": 0,
            "n_rejected_has_confidence_marker": 0,
            "n_preferred_has_markdownish": 0,
            "n_rejected_has_markdownish": 0,
            "frac_prefer_confidence_when_only_one_side": None,
            "frac_prefer_markdownish_when_only_one_side": None,
            "note": "No forced-choice judgments for formatting correlations.",
        }
    pref_conf = rej_conf = pref_md = rej_md = 0
    only_one_conf_pref = only_one_conf_total = 0
    only_one_md_pref = only_one_md_total = 0
    for p in usable:
        chosen, rejected = p.chosen_rejected()
        c_conf, r_conf = _has_confidence_marker(chosen), _has_confidence_marker(rejected)
        c_md, r_md = _has_markdownish_format(chosen), _has_markdownish_format(rejected)
        if c_conf:
            pref_conf += 1
        if r_conf:
            rej_conf += 1
        if c_md:
            pref_md += 1
        if r_md:
            rej_md += 1
        if c_conf != r_conf:
            only_one_conf_total += 1
            if c_conf:
                only_one_conf_pref += 1
        if c_md != r_md:
            only_one_md_total += 1
            if c_md:
                only_one_md_pref += 1
    return {
        "n": n,
        "n_preferred_has_confidence_marker": pref_conf,
        "n_rejected_has_confidence_marker": rej_conf,
        "n_preferred_has_markdownish": pref_md,
        "n_rejected_has_markdownish": rej_md,
        "n_pairs_confidence_differs": only_one_conf_total,
        "n_prefer_confidence_when_differs": only_one_conf_pref,
        "frac_prefer_confidence_when_only_one_side": (
            only_one_conf_pref / only_one_conf_total if only_one_conf_total else None
        ),
        "n_pairs_markdownish_differs": only_one_md_total,
        "n_prefer_markdownish_when_differs": only_one_md_pref,
        "frac_prefer_markdownish_when_only_one_side": (
            only_one_md_pref / only_one_md_total if only_one_md_total else None
        ),
        "interpretation": (
            "Formatting/confidence associations are observational correlations only; "
            "not proof of reward hacking or annotator failure."
        ),
    }


def _exact_response_duplicate_groups(
    pairs: Sequence[PreferencePair],
) -> dict[str, list[str]]:
    """Map exact response text -> list of pair_ids that contain it (A or B)."""
    by_text: dict[str, set[str]] = defaultdict(set)
    for p in pairs:
        by_text[p.response_a].add(p.pair_id)
        by_text[p.response_b].add(p.pair_id)
    return {
        text: sorted(pids)
        for text, pids in by_text.items()
        if len(pids) > 1
    }


def analyze_preference_quality(
    pairs: Sequence[PreferencePair],
    *,
    malformed: Sequence[MalformedRecord] = (),
    n_lines: int | None = None,
) -> PreferenceQualityReport:
    """Analyze preference judgments for QC before training (non-destructive)."""
    findings: list[QualityFinding] = []
    n_lines = n_lines if n_lines is not None else len(pairs) + len(malformed)

    if malformed:
        findings.append(
            QualityFinding(
                severity=QualitySeverity.ERROR,
                code="malformed_records",
                message=f"{len(malformed)} malformed JSONL line(s) failed to parse",
                count=len(malformed),
                details={"line_nos": [m.line_no for m in malformed]},
            )
        )

    winner_counts = Counter(p.winner for p in pairs)
    n_a = winner_counts.get("a", 0)
    n_b = winner_counts.get("b", 0)
    n_tie = winner_counts.get("tie", 0)
    forced = n_a + n_b
    balance = {
        "n_a": n_a,
        "n_b": n_b,
        "n_tie": n_tie,
        "frac_a": (n_a / forced) if forced else None,
        "frac_b": (n_b / forced) if forced else None,
        "abs_imbalance": abs(n_a - n_b),
        "balanced": forced > 0 and abs(n_a - n_b) <= max(1, math.ceil(0.1 * forced)),
    }
    if forced > 0 and min(n_a, n_b) == 0:
        findings.append(
            QualityFinding(
                severity=QualitySeverity.WARNING,
                code="winner_one_sided",
                message=f"All forced-choice winners are on one side (a={n_a}, b={n_b})",
                count=forced,
                details=balance,
            )
        )
    elif forced >= 4 and max(n_a / forced, n_b / forced) >= 0.75:
        findings.append(
            QualityFinding(
                severity=QualitySeverity.WARNING,
                code="winner_imbalance",
                message=(
                    f"Strong A/B imbalance: a={n_a}, b={n_b} "
                    f"(frac_a={balance['frac_a']})"
                ),
                count=forced,
                details=balance,
            )
        )
    elif forced > 0 and not balance["balanced"]:
        findings.append(
            QualityFinding(
                severity=QualitySeverity.INFO,
                code="winner_mild_imbalance",
                message=f"Mild A/B imbalance: a={n_a}, b={n_b}",
                count=forced,
                details=balance,
            )
        )

    grouped = group_judgments_by_pair(pairs)
    dup_groups = {pid: g for pid, g in grouped.items() if len(g) > 1}
    if dup_groups:
        findings.append(
            QualityFinding(
                severity=QualitySeverity.WARNING,
                code="duplicate_pair_ids",
                message=(
                    f"{len(dup_groups)} content pair_id(s) appear more than once "
                    "(multi-annotation or accidental duplicates)"
                ),
                count=len(dup_groups),
                details={
                    "pair_ids": sorted(dup_groups),
                    "counts": {pid: len(g) for pid, g in sorted(dup_groups.items())},
                },
            )
        )

    contradictory = {
        pid: sorted({g.winner for g in group})
        for pid, group in dup_groups.items()
        if len({g.winner for g in group}) > 1
    }
    if contradictory:
        findings.append(
            QualityFinding(
                severity=QualitySeverity.WARNING,
                code="contradictory_labels",
                message=(
                    f"{len(contradictory)} pair_id(s) have disagreeing winners "
                    "across judgments (not automatically annotation error)"
                ),
                count=len(contradictory),
                details={"pair_winners": contradictory},
            )
        )

    # Exact identical A/B within a pair
    identical_ab = [p for p in pairs if p.response_a == p.response_b]
    if identical_ab:
        findings.append(
            QualityFinding(
                severity=QualitySeverity.ERROR,
                code="identical_responses_within_pair",
                message=f"{len(identical_ab)} judgment(s) have identical response_a and response_b",
                count=len(identical_ab),
                details={"judgment_ids": [p.judgment_id for p in identical_ab]},
            )
        )

    resp_dups = _exact_response_duplicate_groups(pairs)
    if resp_dups:
        findings.append(
            QualityFinding(
                severity=QualitySeverity.WARNING,
                code="exact_response_duplicates",
                message=(
                    f"{len(resp_dups)} exact response string(s) appear across "
                    "multiple content pairs"
                ),
                count=len(resp_dups),
                details={
                    "examples": [
                        {"text_preview": text[:80], "pair_ids": pids}
                        for text, pids in list(sorted(resp_dups.items(), key=lambda x: -len(x[1])))[
                            :10
                        ]
                    ]
                },
            )
        )

    # pair_id integrity
    mismatch = []
    for p in pairs:
        expected = compute_content_pair_id(
            clip_id=p.clip_id,
            instruction=p.instruction,
            response_a=p.response_a,
            response_b=p.response_b,
            task=p.task,
        )
        if p.pair_id != expected:
            mismatch.append(p.judgment_id)
    if mismatch:
        findings.append(
            QualityFinding(
                severity=QualitySeverity.ERROR,
                code="pair_id_mismatch",
                message=f"{len(mismatch)} judgment(s) have non-canonical pair_id",
                count=len(mismatch),
                details={"judgment_ids": mismatch},
            )
        )

    # Duplicate judgment_ids
    jid_counts = Counter(p.judgment_id for p in pairs)
    dup_jids = [jid for jid, n in jid_counts.items() if n > 1]
    if dup_jids:
        findings.append(
            QualityFinding(
                severity=QualitySeverity.ERROR,
                code="duplicate_judgment_ids",
                message=f"{len(dup_jids)} duplicate judgment_id(s)",
                count=len(dup_jids),
                details={"judgment_ids": dup_jids},
            )
        )

    agreement = _agreement_stats(pairs)
    if not agreement.can_estimate_inter_rater_reliability:
        findings.append(
            QualityFinding(
                severity=QualitySeverity.INFO,
                code="irr_unavailable",
                message=agreement.caveat,
                count=agreement.n_annotators,
            )
        )
    elif agreement.n_pairs_contradictory:
        findings.append(
            QualityFinding(
                severity=QualitySeverity.WARNING,
                code="repeated_judgment_disagreement",
                message=(
                    f"Agreement rate={agreement.agreement_rate} on "
                    f"{agreement.n_pairs_with_repeated_judgments} multi-annotated pairs "
                    f"({agreement.n_pairs_contradictory} contradictory)"
                ),
                count=agreement.n_pairs_contradictory,
                details=agreement.to_dict(),
            )
        )

    length = _length_stats(pairs)
    if (
        length.n >= 4
        and length.frac_preferred_longer is not None
        and length.frac_preferred_longer >= (2 / 3)
    ):
        findings.append(
            QualityFinding(
                severity=QualitySeverity.BIAS,
                code="possible_length_bias",
                message=(
                    f"Preferred response is longer in {length.n_preferred_longer}/{length.n} "
                    f"forced-choice judgments (frac={length.frac_preferred_longer}). "
                    "Correlation only - not proof that longer is better or that "
                    "annotators rewarded verbosity."
                ),
                count=length.n_preferred_longer,
                details=length.to_dict(),
            )
        )

    formatting = _formatting_analysis(pairs)
    conf_frac = formatting.get("frac_prefer_confidence_when_only_one_side")
    if (
        isinstance(conf_frac, float)
        and formatting.get("n_pairs_confidence_differs", 0) >= 3
        and conf_frac >= 0.75
    ):
        findings.append(
            QualityFinding(
                severity=QualitySeverity.BIAS,
                code="possible_confidence_format_bias",
                message=(
                    f"When only one side has confidence markers, preferred that side "
                    f"in {formatting['n_prefer_confidence_when_differs']}/"
                    f"{formatting['n_pairs_confidence_differs']} cases. Observational only."
                ),
                count=int(formatting["n_prefer_confidence_when_differs"]),
                details={
                    k: formatting[k]
                    for k in (
                        "n_pairs_confidence_differs",
                        "n_prefer_confidence_when_differs",
                        "frac_prefer_confidence_when_only_one_side",
                    )
                },
            )
        )
    md_frac = formatting.get("frac_prefer_markdownish_when_only_one_side")
    if (
        isinstance(md_frac, float)
        and formatting.get("n_pairs_markdownish_differs", 0) >= 3
        and md_frac >= 0.75
    ):
        findings.append(
            QualityFinding(
                severity=QualitySeverity.BIAS,
                code="possible_markdown_format_bias",
                message=(
                    f"When only one side looks markdown-formatted, preferred that side "
                    f"in {formatting['n_prefer_markdownish_when_differs']}/"
                    f"{formatting['n_pairs_markdownish_differs']} cases. Observational only."
                ),
                count=int(formatting["n_prefer_markdownish_when_differs"]),
                details={
                    k: formatting[k]
                    for k in (
                        "n_pairs_markdownish_differs",
                        "n_prefer_markdownish_when_differs",
                        "frac_prefer_markdownish_when_only_one_side",
                    )
                },
            )
        )

    clip_counts = dict(sorted(Counter(p.clip_id for p in pairs).items()))
    task_counts = dict(sorted(Counter(p.task.value for p in pairs).items()))
    trick_counts: Counter[str] = Counter()
    for p in pairs:
        trick = None
        if isinstance(p.metadata, dict):
            trick = p.metadata.get("trick_id")
        trick_counts[str(trick) if trick else f"clip:{p.clip_id}"] += 1
    trick_counts_d = dict(sorted(trick_counts.items()))
    annotator_counts = dict(sorted(Counter(p.annotator_id for p in pairs).items()))

    if len(clip_counts) == 1 and len(pairs) >= 4:
        findings.append(
            QualityFinding(
                severity=QualitySeverity.WARNING,
                code="single_clip_concentration",
                message="All judgments share a single clip_id",
                count=len(pairs),
                details={"clip_id": next(iter(clip_counts))},
            )
        )

    integrity = {
        "n_parsed_plus_malformed": len(pairs) + len(malformed),
        "n_lines_nonempty": n_lines,
        "counts_match": len(pairs) + len(malformed) == n_lines,
        "sum_winner_counts": sum(winner_counts.values()),
        "winner_counts_match_parsed": sum(winner_counts.values()) == len(pairs),
        "no_records_deleted": True,
        "filtering_applied": False,
    }

    caveats = (
        "This report never deletes or filters preference records.",
        "Disagreement across annotators is not automatically an annotation error.",
        "Length or formatting correlations are not evidence of reward hacking.",
        "Do not claim inter-rater reliability with only one annotator.",
        "Do not assume longer explanations are better.",
    )

    return PreferenceQualityReport(
        n_lines=n_lines,
        n_parsed=len(pairs),
        n_malformed=len(malformed),
        winner_counts=dict(sorted(winner_counts.items())),
        winner_balance=balance,
        n_unique_pair_ids=len(grouped),
        n_duplicate_pair_id_groups=len(dup_groups),
        n_exact_response_duplicate_groups=len(resp_dups),
        n_contradictory_pair_ids=len(contradictory),
        agreement=agreement,
        length=length,
        formatting=formatting,
        clip_counts=clip_counts,
        task_counts=task_counts,
        trick_counts=trick_counts_d,
        annotator_counts=annotator_counts,
        findings=tuple(findings),
        malformed=tuple(malformed),
        integrity=integrity,
        caveats=caveats,
    )


def analyze_preference_file(path: str | Path) -> PreferenceQualityReport:
    pairs, malformed, n_lines = load_preferences_with_malformed(path)
    return analyze_preference_quality(pairs, malformed=malformed, n_lines=n_lines)


def format_quality_report(report: PreferenceQualityReport) -> str:
    lines: list[str] = [
        "# Preference data quality report",
        "",
        f"- nonempty lines: {report.n_lines}",
        f"- parsed: {report.n_parsed}",
        f"- malformed: {report.n_malformed}",
        f"- unique pair_ids: {report.n_unique_pair_ids}",
        f"- winner counts: `{report.winner_counts}`",
        f"- A/B balance: `{report.winner_balance}`",
        f"- duplicate pair_id groups: {report.n_duplicate_pair_id_groups}",
        f"- contradictory pair_ids: {report.n_contradictory_pair_ids}",
        f"- exact response duplicate groups: {report.n_exact_response_duplicate_groups}",
        "",
        "## Caveats",
        "",
    ]
    for c in report.caveats:
        lines.append(f"- {c}")
    lines.extend(["", "## Agreement", "", f"- {report.agreement.caveat}"])
    lines.append(f"- stats: `{report.agreement.to_dict()}`")
    lines.extend(["", "## Length", "", f"- `{report.length.to_dict()}`"])
    lines.extend(["", "## Formatting correlations", "", f"- `{report.formatting}`"])
    lines.extend(
        [
            "",
            "## Distributions",
            "",
            f"- clips: `{report.clip_counts}`",
            f"- tasks: `{report.task_counts}`",
            f"- tricks/proxies: `{report.trick_counts}`",
            f"- annotators: `{report.annotator_counts}`",
            "",
            "## Findings",
            "",
        ]
    )
    if not report.findings:
        lines.append("_None._")
    else:
        for f in report.findings:
            lines.append(f"- **[{f.severity.value}]** `{f.code}` (n={f.count}): {f.message}")
    if report.malformed:
        lines.extend(["", "## Malformed lines", ""])
        for m in report.malformed:
            preview = m.raw_text[:120].replace("\n", "\\n")
            lines.append(f"- line {m.line_no}: {m.error} :: `{preview}`")
    lines.extend(["", "## Integrity", ""])
    for k, v in report.integrity.items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    return "\n".join(lines)


def write_quality_outputs(
    report: PreferenceQualityReport,
    out_dir: str | Path,
) -> dict[str, Path]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": root / "preference_quality.json",
        "report": root / "preference_quality_report.md",
        "malformed": root / "malformed_records.jsonl",
        "findings": root / "preference_quality_findings.jsonl",
    }
    write_json(paths["metrics"], report.to_dict())
    paths["report"].write_text(format_quality_report(report), encoding="utf-8")
    write_jsonl(paths["malformed"], [m.to_dict() for m in report.malformed])
    write_jsonl(paths["findings"], [f.to_dict() for f in report.findings])
    return paths
