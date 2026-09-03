#!/usr/bin/env python3
"""Render final VOP+sparse match visualizations for selected GTA-UAV log cases."""

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

from game4loc.dataset.gta import get_transforms, sate2loc
from game4loc.evaluate.gta import _rotate_query_tensor
from game4loc.matcher.gim_dkm import GimDKM
from game4loc.models.model import DesModel
from game4loc.orientation import load_vop_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", nargs="+", required=True)
    parser.add_argument("--galleries", nargs="+", required=True)
    parser.add_argument(
        "--data-root", type=Path, default=GAME4LOC / "data" / "GTA-UAV-data"
    )
    parser.add_argument(
        "--retrieval-checkpoint",
        type=Path,
        default=GAME4LOC / "pretrained/gta/vit_base_eva_gta_same_area.pth",
    )
    parser.add_argument(
        "--orientation-checkpoint",
        type=Path,
        default=GAME4LOC
        / "work_dir/gta_vop_same_area_runs/gta_samearea_fullteacher_exp_c_20260417_125519"
        / "artifacts/gta_samearea_useful5_weight30_e6.pth",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=GAME4LOC / "docs/gta_failure_showcase/match_vis",
    )
    return parser.parse_args()


def tile_geometry(name: str) -> tuple[tuple[float, float], tuple[float, float]]:
    zoom, offset, tile_x, tile_y = map(int, Path(name).stem.split("_"))
    cx, cy, tlx, tly = sate2loc(zoom, offset, tile_x, tile_y)
    return (cx, cy), (tlx, tly)


def load_tensor(path: Path, transform) -> torch.Tensor:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return transform(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB))["image"]


def main() -> None:
    args = parse_args()
    if len(args.queries) != len(args.galleries):
        raise ValueError("--queries and --galleries must have equal length")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    retrieval = DesModel(
        "vit_base_patch16_rope_reg1_gap_256.sbb_in1k",
        pretrained=False,
        img_size=384,
        share_weights=True,
    )
    retrieval.load_state_dict(torch.load(args.retrieval_checkpoint, map_location="cpu"), strict=True)
    retrieval = retrieval.to(device).eval()
    config = retrieval.get_config()
    transform, _, _ = get_transforms((384, 384), mean=config["mean"], std=config["std"])
    vop = load_vop_checkpoint(str(args.orientation_checkpoint), device=device)

    matcher = GimDKM(
        device=device,
        match_mode="sparse",
        sparse_save_final_vis=True,
        sparse_save_final_vis_dir=str(args.output_dir),
        sparse_save_final_vis_max=max(4, len(args.queries) * 2),
    )

    outputs = []
    for query_name, gallery_name in zip(args.queries, args.galleries):
        query = load_tensor(args.data_root / "drone/images" / query_name, transform)
        gallery = load_tensor(args.data_root / "satellite" / gallery_name, transform)
        posterior = vop.predict_posterior(
            retrieval_model=retrieval,
            gallery_img=gallery,
            query_img=query,
            candidate_angles_deg=vop.candidate_angles_deg,
            device=device,
            gallery_branch="img2",
            query_branch="img1",
        )
        top_indices = np.argsort(np.asarray(posterior["probs"], dtype=np.float64))[::-1][:4]
        center, topleft = tile_geometry(gallery_name)
        candidates = []
        for rank, index in enumerate(top_indices):
            angle = float(vop.candidate_angles_deg[int(index)])
            rotated = _rotate_query_tensor(query, angle)
            matcher.est_center(
                gallery,
                rotated,
                center,
                topleft,
                rotate=0.0,
                case_name=f"{Path(query_name).stem}_candidate{rank + 1}_{angle:+.1f}",
                save_final_vis=False,
            )
            info = dict(matcher.get_last_match_info() or {})
            candidates.append((int(info.get("inliers", 0)), float(info.get("inlier_ratio", 0)), angle))
        _, _, selected_angle = max(candidates, key=lambda item: (item[0], item[1]))
        matcher.est_center(
            gallery,
            _rotate_query_tensor(query, selected_angle),
            center,
            topleft,
            rotate=0.0,
            case_name=f"{Path(query_name).stem}_selected_{selected_angle:+.1f}",
            save_final_vis=True,
        )
        info = dict(matcher.get_last_match_info() or {})
        outputs.append(
            {
                "query": query_name,
                "gallery": gallery_name,
                "top4_angles": [float(vop.candidate_angles_deg[int(i)]) for i in top_indices],
                "selected_angle": selected_angle,
                "inliers": int(info.get("inliers", 0)),
                "retained": int(info.get("n_kept", 0)),
                "final_vis_path": info.get("final_vis_path"),
            }
        )
    (args.output_dir / "match_render_summary.json").write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
