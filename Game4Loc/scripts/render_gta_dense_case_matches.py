#!/usr/bin/env python3
"""Render dense DKM correspondence figures for selected GTA-UAV image pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

GAME4LOC = Path(__file__).resolve().parents[1]
if str(GAME4LOC) not in sys.path:
    sys.path.insert(0, str(GAME4LOC))

from game4loc.dataset.gta import get_transforms
from game4loc.matcher.gim_dkm import GimDKM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", nargs="+", required=True)
    parser.add_argument("--galleries", nargs="+", required=True)
    parser.add_argument(
        "--data-root", type=Path, default=GAME4LOC / "data/GTA-UAV-data"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=GAME4LOC / "docs/gta_failure_showcase/dense_match_vis",
    )
    return parser.parse_args()


def load_tensor(path: Path, transform) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return transform(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))["image"]


def render_matches(
    gallery_path: Path,
    query_path: Path,
    debug: dict,
    output: Path,
) -> None:
    gallery = cv2.resize(cv2.imread(str(gallery_path)), (384, 384))
    query = cv2.resize(cv2.imread(str(query_path)), (384, 384))
    canvas = np.concatenate([gallery, query], axis=1)
    mk0 = np.asarray(debug["mk0"])
    mk1 = np.asarray(debug["mk1"])
    inliers = np.asarray(debug["inliers"]).astype(bool)
    # Keep the figure readable while preserving the inlier/outlier balance.
    indices = np.arange(len(mk0))
    if len(indices) > 320:
        indices = np.linspace(0, len(indices) - 1, 320, dtype=int)
    for index in indices:
        p0 = tuple(np.round(mk0[index]).astype(int))
        p1_raw = np.round(mk1[index]).astype(int)
        p1 = (int(p1_raw[0]) + 384, int(p1_raw[1]))
        color = (20, 220, 20) if inliers[index] else (30, 30, 230)
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, color, -1, cv2.LINE_AA)
    text = (
        f"Dense DKM | retained={int(debug['n_kept'])} | "
        f"inliers={int(debug['inliers_count'])} | green=inlier red=outlier"
    )
    cv2.rectangle(canvas, (0, 0), (768, 28), (0, 0, 0), -1)
    cv2.putText(canvas, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(output), canvas)


def main() -> None:
    args = parse_args()
    if len(args.queries) != len(args.galleries):
        raise ValueError("--queries and --galleries must have equal length")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform, _, _ = get_transforms((384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    matcher = GimDKM(device=device, match_mode="dense")
    rows = []
    for query_name, gallery_name in zip(args.queries, args.galleries):
        query_path = args.data_root / "drone/images" / query_name
        gallery_path = args.data_root / "satellite" / gallery_name
        query = load_tensor(query_path, transform)
        gallery = load_tensor(gallery_path, transform)
        matcher.match(
            gallery.to(device)[None, ...] * 0.5 + 0.5,
            query.to(device)[None, ...] * 0.5 + 0.5,
            vis=False,
            rotate=False,
        )
        output = args.output_dir / f"{Path(query_name).stem}_dense_dkm.png"
        debug = matcher.get_last_match_debug()
        if not debug:
            raise RuntimeError(f"Dense matcher did not produce debug correspondences for {query_name}")
        render_matches(gallery_path, query_path, debug, output)
        info = dict(matcher.get_last_match_info() or {})
        rows.append(
            {
                "query": query_name,
                "gallery": gallery_name,
                "mode": "dense DKM, no rotate",
                "retained": int(info.get("n_kept", 0)),
                "inliers": int(info.get("inliers", 0)),
                "inlier_ratio": float(info.get("inlier_ratio", 0.0)),
                "image": output.name,
            }
        )
    (args.output_dir / "dense_match_render_summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
