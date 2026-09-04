"""Summarize complete matched official runs from post-selection FineAudit logs."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from audit_gta_pose_likelihood_statistics import absolute_metrics, effect_summary


AUDIT_PATTERN = re.compile(
    r"FineAudit query=(.*?) coarse_m=([\d.eE+-]+) final_m=([\d.eE+-]+) "
    r"fallback=(\d+) hypotheses=(\d+) vop_s=([\d.eE+-]+) matcher_s=([\d.eE+-]+)"
)


def parse_audit(text: str) -> dict:
    rows = {}
    for match in AUDIT_PATTERN.finditer(text):
        name, coarse, final, fallback, hypotheses, vop, matcher = match.groups()
        if name in rows:
            raise ValueError(f"Duplicate query in official log: {name}")
        values = [float(coarse), float(final), float(vop), float(matcher)]
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite audit values for {name}")
        rows[name] = dict(coarse=values[0], final=values[1], fallback=int(fallback),
                          hypotheses=int(hypotheses), vop_s=values[2], matcher_s=values[3])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--expected_queries', type=int, default=3443)
    parser.add_argument('--bootstrap_replicates', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260903)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    variants = ('legacy_top1', 'raw_top5x4', 'adaptive_calibrated')
    records, summaries = {}, {}
    names = None
    for variant in variants:
        log = (run_dir / f'{variant}.log').read_text()
        rows = parse_audit(log)
        if len(rows) != args.expected_queries:
            raise ValueError(f'{variant}: expected {args.expected_queries} queries, got {len(rows)}')
        if names is None:
            names = sorted(rows)
        if set(rows) != set(names):
            raise ValueError(f'{variant}: unmatched query cohort')
        records[variant] = rows
        coarse = np.asarray([rows[n]['coarse'] for n in names])
        errors = np.asarray([rows[n]['final'] for n in names])
        metrics = absolute_metrics(errors, coarse)
        result_lines = [line for line in log.splitlines() if 'Recall@1:' in line and 'Dis@1:' in line]
        if not result_lines:
            raise ValueError(f'{variant}: final official metric line missing')
        logged = {k: float(v) for k, v in re.findall(r'([A-Za-z]+@\d+|mAP): ([\d.eE+-]+)', result_lines[-1])}
        if abs(logged['Dis@1'] - metrics['Dis@1_m']) > 0.0005:
            raise ValueError(f'{variant}: FineAudit mean disagrees with official Dis@1')
        summaries[variant] = {
            **metrics, 'query_count': len(rows),
            'fallback_count': sum(r['fallback'] for r in rows.values()),
            'fallback_pct': 100 * np.mean([r['fallback'] for r in rows.values()]),
            'mean_hypotheses': np.mean([r['hypotheses'] for r in rows.values()]),
            'mean_vop_s': np.mean([r['vop_s'] for r in rows.values()]),
            'mean_matcher_s': np.mean([r['matcher_s'] for r in rows.values()]),
            'mean_vop_match_s': np.mean([r['vop_s'] + r['matcher_s'] for r in rows.values()]),
            'official_retrieval_and_distance_metrics': logged,
            'source_log': str((run_dir / f'{variant}.log').resolve()),
        }
    coarse = np.asarray([records['legacy_top1'][n]['coarse'] for n in names])
    for variant in variants[1:]:
        np.testing.assert_allclose([records[variant][n]['coarse'] for n in names], coarse, rtol=0, atol=1e-6)
    paired = {
        'coarse_error': coarse,
        'legacy_error': np.asarray([records['legacy_top1'][n]['final'] for n in names]),
        'adaptive_error': np.asarray([records['adaptive_calibrated'][n]['final'] for n in names]),
    }
    indices = np.random.default_rng(args.seed).integers(0, len(names), (args.bootstrap_replicates, len(names)))
    effects = effect_summary(paired, indices)
    legacy, adaptive = summaries['legacy_top1'], summaries['adaptive_calibrated']
    gates = {
        'Dis_relative_reduction_ge_5pct': (legacy['Dis@1_m'] - adaptive['Dis@1_m']) / legacy['Dis@1_m'] >= .05,
        'MA20_gain_ge_2pp': adaptive['MA@20_pct'] - legacy['MA@20_pct'] >= 2,
        'worse_than_coarse_nonincrease': adaptive['worse_than_coarse_pct'] <= legacy['worse_than_coarse_pct'],
        'catastrophic_nonincrease': adaptive['catastrophic_50m_pct'] <= legacy['catastrophic_50m_pct'],
    }
    summary = {
        'scope': 'Official matched full same-area; no cross-area validation',
        'variants': summaries, 'paired_effects': effects,
        'bootstrap': {'seed': args.seed, 'replicates': args.bootstrap_replicates, 'unit': 'query'},
        'full_samearea_gates': gates, 'decision': 'KEEP' if all(gates.values()) else 'REJECT',
        'timing_scope': 'VOP + matcher timers, not end-to-end latency',
    }
    output = run_dir / 'official_summary.json'
    output.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
