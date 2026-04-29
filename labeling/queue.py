"""
Generate a per-reviewer labeling batch.

Three phases (see docs/labeling_plan.md):

  0  Calibration only — identical set across reviewers, sourced from
     calibration_ground_truth. Order is deterministic (sorted by
     video_id, track_id) so every reviewer gets the same tasks.

  1  Stratified high-confidence — mean_detection_confidence >= MIN_CONF,
     background not dominant, balanced n/5 per fish species,
     3% calibration injection.

  2  Mixed priority queue —
       priority = w_unc * uncertainty
                + w_tmp * temporal_anomaly  (deferred; w_tmp=0 by default)
                + w_cnf * confusion_pair
                + w_jit * jitter
     3% calibration injection, 7% multi-review re-emits.

This module is read-only against both databases. labels.sqlite is opened
through schema.open_labels_db so missing tables are created idempotently,
but no rows are written here. merge_labels.py / export_labels.py mutate
labels.sqlite later in the pipeline.

Output: a JSON artifact consumed by import_tasks.py (step 5).

Usage:
  python labeling/queue.py
    --tracks-db   PATH/tracks.sqlite
    --labels-db   PATH/labels.sqlite
    --reviewer-id alice
    --phase       {0,1,2}
    --n-tasks     200
    --out         PATH/queue.json
    [--batch-id   batch_20260429_alice_001]
    [--seed       42]
    [--min-conf   0.85]              # phase 1
    [--bg-max     0.50]              # phase 1
    [--w-unc 0.50 --w-tmp 0.0 --w-cnf 0.15 --w-jit 0.05]   # phase 2
    [--calibration-rate 0.03]
    [--multi-review-rate 0.07]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import random
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from schema import open_labels_db

SCHEMA_VERSION: int = 1

# Class indices match inference/schema.py FISH_CLASSES.
FISH_CLASSES: List[str] = ["Chinook", "Coho", "Atlantic", "Rainbow Trout", "Brown Trout"]

# Known-confusable species pairs as frozensets of class indices.
# Source: domain experts. Add pairs here as confusion-matrix evidence emerges.
CONFUSION_PAIRS: frozenset = frozenset({
    frozenset({0, 1}),  # Chinook / Coho
    frozenset({2, 4}),  # Atlantic / Brown
})

# Plan defaults (docs/labeling_plan.md "Queue strategy").
DEFAULT_W_UNCERTAINTY: float = 0.50
DEFAULT_W_TEMPORAL: float = 0.00
DEFAULT_W_CONFUSION: float = 0.15
DEFAULT_W_JITTER: float = 0.05
DEFAULT_CALIBRATION_RATE: float = 0.03
DEFAULT_MULTI_REVIEW_RATE: float = 0.07
DEFAULT_PHASE1_MIN_CONF: float = 0.85
DEFAULT_PHASE1_BG_PROB_MAX: float = 0.50

LOG_5: float = math.log(5)


# ---------------------------------------------------------------------------
# Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class QueueItem:
    video_id: str
    track_id: int
    calibration_task: int           # 0 / 1
    multi_reviewed: int             # 0 / 1
    phase: int                      # 0, 1, 2 (calibration injections keep parent batch phase)
    priority_score: float
    score_components: Dict[str, float]
    reason: str                     # short string for debug / audit


@dataclass
class QueueArtifact:
    schema_version: int
    batch_id: str
    reviewer_id: str
    phase: int
    generated_at: str               # ISO-8601 UTC
    config: Dict
    items: List[QueueItem] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase builders
# ---------------------------------------------------------------------------

def build_phase0(
    conn_labels: sqlite3.Connection,
    n_tasks: int,
) -> List[QueueItem]:
    """Calibration-only batch. Identical set across all reviewers."""
    rows = conn_labels.execute(
        "SELECT video_id, track_id FROM calibration_ground_truth "
        "ORDER BY video_id, track_id"
    ).fetchall()
    if len(rows) < n_tasks:
        raise ValueError(
            f"calibration_ground_truth has {len(rows)} rows; "
            f"need {n_tasks} for phase 0"
        )
    return [
        QueueItem(
            video_id=r["video_id"],
            track_id=r["track_id"],
            calibration_task=1,
            multi_reviewed=0,
            phase=0,
            priority_score=0.0,
            score_components={},
            reason="calibration",
        )
        for r in rows[:n_tasks]
    ]


def build_phase1(
    conn_tracks: sqlite3.Connection,
    conn_labels: sqlite3.Connection,
    reviewer_id: str,
    n_tasks: int,
    min_conf: float,
    bg_prob_max: float,
    calibration_rate: float,
    rng: random.Random,
) -> List[QueueItem]:
    """Species-stratified high-confidence batch with calibration injection."""
    excluded = _excluded_track_ids(conn_labels, reviewer_id)
    candidates = _query_high_conf_tracks(conn_tracks, min_conf, bg_prob_max)
    candidates = [t for t in candidates if (t["video_id"], t["track_id"]) not in excluded]

    n_calibration = int(round(n_tasks * calibration_rate))
    n_main = n_tasks - n_calibration
    n_per_species = max(1, n_main // 5)

    by_class: Dict[int, List[dict]] = {i: [] for i in range(5)}
    for t in candidates:
        cls = t["predicted_class"]
        if cls in by_class:
            by_class[cls].append(t)

    main_items: List[QueueItem] = []
    short_classes: List[str] = []
    for cls_idx in range(5):
        bucket = by_class[cls_idx]
        rng.shuffle(bucket)
        chosen = bucket[:n_per_species]
        if len(chosen) < n_per_species:
            short_classes.append(
                f"{FISH_CLASSES[cls_idx]} ({len(chosen)}/{n_per_species})"
            )
        for t in chosen:
            main_items.append(QueueItem(
                video_id=t["video_id"],
                track_id=t["track_id"],
                calibration_task=0,
                multi_reviewed=0,
                phase=1,
                priority_score=float(t["mean_detection_confidence"]),
                score_components={
                    "mean_detection_confidence": float(t["mean_detection_confidence"]),
                },
                reason=f"phase1_stratified_{FISH_CLASSES[cls_idx]}",
            ))

    if short_classes:
        print(f"WARN: phase1 short on: {', '.join(short_classes)}", file=sys.stderr)

    # Per-species ceil rounding can push the stratified pool above n_main
    # for small n_tasks (e.g. n_main=3 → 5 picks). Trim before injecting
    # calibration so the final batch never exceeds n_tasks.
    rng.shuffle(main_items)
    main_items = main_items[:n_main]

    cal_items = _sample_calibration(conn_labels, reviewer_id, n_calibration, rng, phase=1)

    items = main_items + cal_items
    rng.shuffle(items)
    return items


def build_phase2(
    conn_tracks: sqlite3.Connection,
    conn_labels: sqlite3.Connection,
    reviewer_id: str,
    n_tasks: int,
    weights: Dict[str, float],
    calibration_rate: float,
    multi_review_rate: float,
    rng: random.Random,
) -> List[QueueItem]:
    """Priority-scored mixed batch with calibration and multi-review injections."""
    n_calibration = int(round(n_tasks * calibration_rate))
    n_multi = int(round(n_tasks * multi_review_rate))
    n_main = n_tasks - n_calibration - n_multi
    if n_main < 0:
        raise ValueError(
            f"n_tasks={n_tasks} is too small for "
            f"calibration_rate={calibration_rate} + multi_review_rate={multi_review_rate}"
        )

    excluded = _excluded_track_ids(conn_labels, reviewer_id)
    labeled_anyone = _labeled_by_anyone_track_ids(conn_labels)

    candidates = _query_all_tracks_with_probs(conn_tracks)
    main_pool = [
        t for t in candidates
        if (t["video_id"], t["track_id"]) not in excluded
        and (t["video_id"], t["track_id"]) not in labeled_anyone
    ]

    scored = []
    for t in main_pool:
        probs = t["mean_probs_fish"]
        if any(p is None for p in probs):
            continue
        s_unc = _entropy_normalised(probs)
        top1, top2 = _top2_indices(probs)
        s_cnf = 1.0 if frozenset({top1, top2}) in CONFUSION_PAIRS else 0.0
        s_jit = rng.random()
        s_tmp = 0.0  # temporal_anomaly deferred until historical priors land
        priority = (
            weights["uncertainty"] * s_unc
            + weights["temporal"]   * s_tmp
            + weights["confusion"]  * s_cnf
            + weights["jitter"]     * s_jit
        )
        scored.append((priority, s_unc, s_cnf, s_jit, s_tmp, t))

    scored.sort(key=lambda x: -x[0])
    main_items = [
        QueueItem(
            video_id=t["video_id"],
            track_id=t["track_id"],
            calibration_task=0,
            multi_reviewed=0,
            phase=2,
            priority_score=float(priority),
            score_components={
                "uncertainty": float(s_unc),
                "confusion":   float(s_cnf),
                "jitter":      float(s_jit),
                "temporal":    float(s_tmp),
            },
            reason="priority",
        )
        for priority, s_unc, s_cnf, s_jit, s_tmp, t in scored[:n_main]
    ]
    if len(main_items) < n_main:
        print(
            f"WARN: phase2 main pool short ({len(main_items)}/{n_main})",
            file=sys.stderr,
        )

    multi_pool = _multi_review_candidates(conn_labels, reviewer_id, excluded)
    rng.shuffle(multi_pool)
    multi_items = [
        QueueItem(
            video_id=r["video_id"],
            track_id=r["track_id"],
            calibration_task=0,
            multi_reviewed=1,
            phase=2,
            priority_score=0.0,
            score_components={},
            reason="multi_review",
        )
        for r in multi_pool[:n_multi]
    ]
    if len(multi_items) < n_multi:
        print(
            f"WARN: phase2 multi-review pool short ({len(multi_items)}/{n_multi})",
            file=sys.stderr,
        )

    cal_items = _sample_calibration(conn_labels, reviewer_id, n_calibration, rng, phase=2)

    items = main_items + multi_items + cal_items
    rng.shuffle(items)
    return items


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def _excluded_track_ids(
    conn_labels: sqlite3.Connection,
    reviewer_id: str,
) -> set:
    """Tracks this reviewer has already labeled OR skipped — never re-send."""
    rows = conn_labels.execute(
        "SELECT video_id, track_id FROM labels WHERE reviewer_id=?",
        (reviewer_id,),
    ).fetchall()
    skipped = conn_labels.execute(
        "SELECT video_id, track_id FROM skipped_tasks WHERE reviewer_id=?",
        (reviewer_id,),
    ).fetchall()
    return {(r["video_id"], r["track_id"]) for r in rows} \
         | {(r["video_id"], r["track_id"]) for r in skipped}


def _labeled_by_anyone_track_ids(conn_labels: sqlite3.Connection) -> set:
    rows = conn_labels.execute(
        "SELECT DISTINCT video_id, track_id FROM labels"
    ).fetchall()
    return {(r["video_id"], r["track_id"]) for r in rows}


def _multi_review_candidates(
    conn_labels: sqlite3.Connection,
    reviewer_id: str,
    excluded: set,
) -> List[sqlite3.Row]:
    """Tracks labeled by exactly one OTHER reviewer and not yet flagged
    multi_reviewed. Excludes any track already touched by this reviewer."""
    rows = conn_labels.execute(
        """
        SELECT video_id, track_id
        FROM labels
        WHERE reviewer_id != ?
        GROUP BY video_id, track_id
        HAVING COUNT(*) = 1
           AND SUM(CASE WHEN multi_reviewed=1 THEN 1 ELSE 0 END) = 0
        """,
        (reviewer_id,),
    ).fetchall()
    return [r for r in rows if (r["video_id"], r["track_id"]) not in excluded]


def _sample_calibration(
    conn_labels: sqlite3.Connection,
    reviewer_id: str,
    n: int,
    rng: random.Random,
    phase: int,
) -> List[QueueItem]:
    """Sample n calibration tasks not yet labeled by this reviewer."""
    if n <= 0:
        return []
    rows = conn_labels.execute(
        """
        SELECT cgt.video_id, cgt.track_id
        FROM calibration_ground_truth cgt
        WHERE NOT EXISTS (
            SELECT 1 FROM labels l
            WHERE l.video_id   = cgt.video_id
              AND l.track_id   = cgt.track_id
              AND l.reviewer_id = ?
        )
        """,
        (reviewer_id,),
    ).fetchall()
    if len(rows) < n:
        print(
            f"WARN: calibration pool short ({len(rows)}/{n}) for reviewer "
            f"{reviewer_id}; emitting {len(rows)}",
            file=sys.stderr,
        )
    rng.shuffle(rows)
    return [
        QueueItem(
            video_id=r["video_id"],
            track_id=r["track_id"],
            calibration_task=1,
            multi_reviewed=0,
            phase=phase,
            priority_score=0.0,
            score_components={},
            reason="calibration",
        )
        for r in rows[:n]
    ]


def _query_high_conf_tracks(
    conn_tracks: sqlite3.Connection,
    min_conf: float,
    bg_prob_max: float,
) -> List[dict]:
    rows = conn_tracks.execute(
        """
        SELECT video_id, track_id,
               mean_prob_chinook, mean_prob_coho, mean_prob_atlantic,
               mean_prob_rainbow, mean_prob_brown, mean_prob_background,
               mean_detection_confidence, predicted_class, predicted_class_6
        FROM tracks
        WHERE clip_path IS NOT NULL
          AND mean_detection_confidence >= ?
          AND mean_prob_background       <  ?
          AND predicted_class_6 != 5
        """,
        (min_conf, bg_prob_max),
    ).fetchall()
    return [dict(r) for r in rows]


def _query_all_tracks_with_probs(conn_tracks: sqlite3.Connection) -> List[dict]:
    rows = conn_tracks.execute(
        """
        SELECT video_id, track_id,
               mean_prob_chinook, mean_prob_coho, mean_prob_atlantic,
               mean_prob_rainbow, mean_prob_brown, mean_prob_background,
               mean_detection_confidence, predicted_class, predicted_class_6
        FROM tracks
        WHERE clip_path IS NOT NULL
        """
    ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["mean_probs_fish"] = [
            r["mean_prob_chinook"],
            r["mean_prob_coho"],
            r["mean_prob_atlantic"],
            r["mean_prob_rainbow"],
            r["mean_prob_brown"],
        ]
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _entropy_normalised(probs_fish: Sequence[float]) -> float:
    """Entropy of the renormalised 5-class fish distribution, scaled to [0, 1].

    The mean_prob_* columns are class-marginal probabilities across all 6 DETR
    classes, so they don't sum to 1 over just the 5 fish classes. Renormalise
    before computing entropy so a track with high background prob doesn't get
    a spuriously low fish-only uncertainty.
    """
    s = sum(probs_fish)
    if s <= 0:
        return 0.0
    h = 0.0
    for p in probs_fish:
        if p <= 0:
            continue
        q = p / s
        h -= q * math.log(q)
    return h / LOG_5


def _top2_indices(probs: Sequence[float]) -> Tuple[int, int]:
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    return indexed[0][0], indexed[1][0]


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _artifact_to_dict(a: QueueArtifact) -> dict:
    return {
        "schema_version": a.schema_version,
        "batch_id":       a.batch_id,
        "reviewer_id":    a.reviewer_id,
        "phase":          a.phase,
        "generated_at":   a.generated_at,
        "config":         a.config,
        "items":          [asdict(i) for i in a.items],
    }


def _default_batch_id(reviewer_id: str) -> str:
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"batch_{ts}_{reviewer_id}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    tracks_db = Path(args.tracks_db)
    labels_db = Path(args.labels_db)
    if not tracks_db.is_file():
        sys.exit(f"tracks DB not found: {tracks_db}")
    labels_db.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    conn_tracks = sqlite3.connect(str(tracks_db))
    conn_tracks.row_factory = sqlite3.Row

    conn_labels = open_labels_db(labels_db)
    conn_labels.row_factory = sqlite3.Row

    batch_id = args.batch_id or _default_batch_id(args.reviewer_id)

    if args.phase == 0:
        items = build_phase0(conn_labels, args.n_tasks)
        config_used = {"n_tasks": args.n_tasks}
    elif args.phase == 1:
        items = build_phase1(
            conn_tracks, conn_labels,
            reviewer_id=args.reviewer_id,
            n_tasks=args.n_tasks,
            min_conf=args.min_conf,
            bg_prob_max=args.bg_max,
            calibration_rate=args.calibration_rate,
            rng=rng,
        )
        config_used = {
            "n_tasks":          args.n_tasks,
            "min_conf":         args.min_conf,
            "bg_max":           args.bg_max,
            "calibration_rate": args.calibration_rate,
            "seed":             args.seed,
        }
    elif args.phase == 2:
        weights = {
            "uncertainty": args.w_unc,
            "temporal":    args.w_tmp,
            "confusion":   args.w_cnf,
            "jitter":      args.w_jit,
        }
        items = build_phase2(
            conn_tracks, conn_labels,
            reviewer_id=args.reviewer_id,
            n_tasks=args.n_tasks,
            weights=weights,
            calibration_rate=args.calibration_rate,
            multi_review_rate=args.multi_review_rate,
            rng=rng,
        )
        config_used = {
            "n_tasks":           args.n_tasks,
            "weights":           weights,
            "calibration_rate":  args.calibration_rate,
            "multi_review_rate": args.multi_review_rate,
            "seed":              args.seed,
        }
    else:
        sys.exit(f"unknown phase: {args.phase}")

    artifact = QueueArtifact(
        schema_version=SCHEMA_VERSION,
        batch_id=batch_id,
        reviewer_id=args.reviewer_id,
        phase=args.phase,
        generated_at=dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        config=config_used,
        items=items,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(_artifact_to_dict(artifact), f, indent=2)

    n_cal = sum(1 for i in items if i.calibration_task)
    n_mr  = sum(1 for i in items if i.multi_reviewed)
    print(
        f"Wrote {len(items)} items to {out_path}\n"
        f"  phase={args.phase}  reviewer_id={args.reviewer_id}  batch_id={batch_id}\n"
        f"  calibration={n_cal}  multi_review={n_mr}  "
        f"main={len(items) - n_cal - n_mr}"
    )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a per-reviewer labeling batch JSON.",
    )
    p.add_argument("--tracks-db",   required=True, metavar="PATH",
                   help="Path to tracks.sqlite (post extract_clips.py).")
    p.add_argument("--labels-db",   required=True, metavar="PATH",
                   help="Path to labels.sqlite. Created if missing.")
    p.add_argument("--reviewer-id", required=True, metavar="ID",
                   help="Reviewer identifier for excluding their prior work.")
    p.add_argument("--phase",       required=True, type=int, choices=(0, 1, 2),
                   help="Queue phase (see module docstring).")
    p.add_argument("--n-tasks",     required=True, type=int,
                   help="Total number of tasks to emit.")
    p.add_argument("--out",         required=True, metavar="PATH",
                   help="Output JSON path.")
    p.add_argument("--batch-id",    default=None,
                   help="Batch identifier; auto-generated if omitted.")
    p.add_argument("--seed",        type=int, default=0,
                   help="RNG seed for shuffles, jitter, sampling (default 0).")
    # Phase 1
    p.add_argument("--min-conf",    type=float, default=DEFAULT_PHASE1_MIN_CONF,
                   help="Phase 1: min mean_detection_confidence "
                        f"(default {DEFAULT_PHASE1_MIN_CONF}).")
    p.add_argument("--bg-max",      type=float, default=DEFAULT_PHASE1_BG_PROB_MAX,
                   help="Phase 1: max mean_prob_background "
                        f"(default {DEFAULT_PHASE1_BG_PROB_MAX}).")
    # Phase 2 weights
    p.add_argument("--w-unc",       type=float, default=DEFAULT_W_UNCERTAINTY,
                   help=f"Phase 2: uncertainty weight (default {DEFAULT_W_UNCERTAINTY}).")
    p.add_argument("--w-tmp",       type=float, default=DEFAULT_W_TEMPORAL,
                   help=f"Phase 2: temporal-anomaly weight (default {DEFAULT_W_TEMPORAL}).")
    p.add_argument("--w-cnf",       type=float, default=DEFAULT_W_CONFUSION,
                   help=f"Phase 2: confusion-pair weight (default {DEFAULT_W_CONFUSION}).")
    p.add_argument("--w-jit",       type=float, default=DEFAULT_W_JITTER,
                   help=f"Phase 2: jitter weight (default {DEFAULT_W_JITTER}).")
    # Injection rates
    p.add_argument("--calibration-rate",  type=float, default=DEFAULT_CALIBRATION_RATE,
                   help=f"Phase 1/2: calibration injection rate "
                        f"(default {DEFAULT_CALIBRATION_RATE}).")
    p.add_argument("--multi-review-rate", type=float, default=DEFAULT_MULTI_REVIEW_RATE,
                   help=f"Phase 2: multi-review re-emit rate "
                        f"(default {DEFAULT_MULTI_REVIEW_RATE}).")
    return p.parse_args()


if __name__ == "__main__":
    main()
