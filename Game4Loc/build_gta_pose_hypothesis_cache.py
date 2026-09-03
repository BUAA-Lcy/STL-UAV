"""Build resumable GTA-UAV multi-tile × multi-angle pose hypothesis caches."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import OrderedDict, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from game4loc.dataset.gta import GTADatasetEval, get_transforms
from game4loc.evaluate.gta import predict
from game4loc.evaluate.gta_pose_likelihood import (
    FEATURE_NAMES,
    evaluate_tile_hypotheses,
    prefer_inlier_candidate,
    serializable,
)
from game4loc.matcher.gim_dkm import GimDKM
from game4loc.models.model import DesModel
from game4loc.orientation import load_vop_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="./data/GTA-UAV-data")
    parser.add_argument("--pairs_meta_file", required=True)
    parser.add_argument("--model", default="vit_base_patch16_rope_reg1_gap_256.sbb_in1k")
    parser.add_argument("--checkpoint_start", required=True)
    parser.add_argument("--orientation_checkpoint", required=True)
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--retrieval_topk", type=int, default=5)
    parser.add_argument("--orientation_topk", type=int, default=4)
    parser.add_argument("--query_limit", type=int, default=0)
    parser.add_argument("--query_offset", type=int, default=0)
    parser.add_argument("--sample_mode", choices=("sequential", "stratified"), default="stratified")
    parser.add_argument("--sample_seed", type=int, default=20260903)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="")
    return parser.parse_args()


def _file_identity(path: str | Path) -> Dict[str, object]:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return {"path": str(resolved), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _configuration(args: argparse.Namespace) -> Dict[str, object]:
    payload = {
        "schema_version": 1,
        "data_root": str(Path(args.data_root).resolve()),
        "pairs_meta_file": str(args.pairs_meta_file),
        "model": str(args.model),
        "checkpoint": _file_identity(args.checkpoint_start),
        "orientation_checkpoint": _file_identity(args.orientation_checkpoint),
        "img_size": int(args.img_size),
        "retrieval_topk": int(args.retrieval_topk),
        "orientation_topk": int(args.orientation_topk),
        "sample_mode": str(args.sample_mode),
        "sample_seed": int(args.sample_seed),
        "query_limit": int(args.query_limit),
        "query_offset": int(args.query_offset),
        "feature_names": list(FEATURE_NAMES),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["fingerprint"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _stratified_indices(names: Sequence[str], seed: int) -> List[int]:
    groups: Dict[str, List[int]] = defaultdict(list)
    for index, name in enumerate(names):
        parts = Path(str(name)).name.split("_")
        group = "_".join(parts[:2]) if len(parts) >= 2 else parts[0]
        groups[group].append(index)
    rng = random.Random(int(seed))
    for values in groups.values():
        rng.shuffle(values)
    ordered: List[int] = []
    group_names = sorted(groups)
    while True:
        added = False
        for group in group_names:
            if groups[group]:
                ordered.append(groups[group].pop())
                added = True
        if not added:
            break
    return ordered


def _select_indices(names: Sequence[str], args: argparse.Namespace) -> List[int]:
    if args.sample_mode == "stratified":
        indices = _stratified_indices(names, args.sample_seed)
    else:
        indices = list(range(len(names)))
    offset = max(0, int(args.query_offset))
    indices = indices[offset:]
    if int(args.query_limit) > 0:
        indices = indices[: int(args.query_limit)]
    return indices


def _read_completed(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            completed.add(str(record["query_name"]))
    return completed


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1])))


def _top_retrieval(query_features: torch.Tensor, gallery_features: torch.Tensor, topk: int):
    topk = max(1, min(int(topk), int(gallery_features.shape[0])))
    all_indices: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    for start in range(0, int(query_features.shape[0]), 256):
        scores = query_features[start : start + 256] @ gallery_features.T
        values, indices = torch.topk(scores, k=topk, dim=1, largest=True, sorted=True)
        all_scores.append(values.detach().cpu().numpy())
        all_indices.append(indices.detach().cpu().numpy())
    return np.concatenate(all_indices, axis=0), np.concatenate(all_scores, axis=0)


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(args.output_path)
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config = _configuration(args)

    if args.overwrite:
        output_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"{output_path} exists; pass --resume or --overwrite")
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != config["fingerprint"]:
            raise ValueError("Cache configuration fingerprint mismatch; use a new output or --overwrite")
    else:
        manifest_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    completed = _read_completed(output_path) if args.resume else set()

    model = DesModel(args.model, pretrained=False, img_size=args.img_size, share_weights=True)
    state_dict = torch.load(args.checkpoint_start, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    orientation_model = load_vop_checkpoint(args.orientation_checkpoint, device=device)
    data_config = model.get_config()
    val_transforms, _, _ = get_transforms(
        (args.img_size, args.img_size), mean=data_config["mean"], std=data_config["std"]
    )
    query_dataset = GTADatasetEval(
        data_root=args.data_root,
        pairs_meta_file=args.pairs_meta_file,
        view="drone",
        transforms=val_transforms,
        mode="pos",
        query_mode="D2S",
    )
    gallery_dataset = GTADatasetEval(
        data_root=args.data_root,
        pairs_meta_file=args.pairs_meta_file,
        view="sate",
        transforms=val_transforms,
        sate_img_dir="satellite",
        mode="pos",
        query_mode="D2S",
    )
    selected_indices = _select_indices(query_dataset.images_name, args)
    selected_indices = [
        index for index in selected_indices if str(query_dataset.images_name[index]) not in completed
    ]
    if not selected_indices:
        print(f"Cache already complete: {output_path}")
        return
    selected_dataset = Subset(query_dataset, selected_indices)
    query_loader = DataLoader(
        selected_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    gallery_loader = DataLoader(
        gallery_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    predict_config = SimpleNamespace(device=device, verbose=True, normalize_features=True)
    with torch.no_grad():
        query_features = predict(predict_config, model, query_loader)
        gallery_features = predict(predict_config, model, gallery_loader)
        ranking_indices, ranking_scores = _top_retrieval(
            query_features, gallery_features, max(int(args.retrieval_topk) + 1, 2)
        )

    matcher = GimDKM(device=device, logger=None, match_mode="sparse", sparse_save_final_vis=False)
    gallery_cache: OrderedDict[int, torch.Tensor] = OrderedDict()
    start_time = time.perf_counter()
    written = 0
    with output_path.open("a", encoding="utf-8") as output_handle:
        for local_index, query_index in enumerate(selected_indices):
            query_name = str(query_dataset.images_name[query_index])
            if query_name in completed:
                continue
            query_img = query_dataset[query_index]
            query_xy = query_dataset.images_center_loc_xy[query_index]
            top_indices = ranking_indices[local_index]
            top_scores = ranking_scores[local_index]
            coarse_top1_xy = gallery_dataset.images_center_loc_xy[int(top_indices[0])]
            coarse_top1_error = float(math.hypot(query_xy[0] - coarse_top1_xy[0], query_xy[1] - coarse_top1_xy[1]))

            candidates: List[Dict[str, object]] = []
            vop_time = 0.0
            match_time = 0.0
            for retrieval_position in range(min(int(args.retrieval_topk), len(top_indices))):
                gallery_index = int(top_indices[retrieval_position])
                if gallery_index in gallery_cache:
                    gallery_img = gallery_cache.pop(gallery_index)
                    gallery_cache[gallery_index] = gallery_img
                else:
                    gallery_img = gallery_dataset[gallery_index]
                    gallery_cache[gallery_index] = gallery_img
                    if len(gallery_cache) > 256:
                        gallery_cache.popitem(last=False)
                next_score_index = min(retrieval_position + 1, len(top_scores) - 1)
                tile_candidates, timing = evaluate_tile_hypotheses(
                    retrieval_model=model,
                    orientation_model=orientation_model,
                    matcher=matcher,
                    query_img=query_img,
                    gallery_img=gallery_img,
                    gallery_center_xy=gallery_dataset.images_center_loc_xy[gallery_index],
                    gallery_topleft_xy=gallery_dataset.images_topleft_loc_xy[gallery_index],
                    gallery_index=gallery_index,
                    gallery_name=gallery_dataset.images_name[gallery_index],
                    retrieval_rank=retrieval_position + 1,
                    retrieval_score=float(top_scores[retrieval_position]),
                    retrieval_top1_score=float(top_scores[0]),
                    retrieval_next_score=float(top_scores[next_score_index]),
                    orientation_topk=args.orientation_topk,
                    device=device,
                    case_prefix=Path(query_name).stem,
                )
                vop_time += timing["vop_time_s"]
                match_time += timing["match_time_s"]
                for candidate in tile_candidates:
                    predicted_xy = candidate["predicted_xy"]
                    candidate_error = float(
                        math.hypot(query_xy[0] - predicted_xy[0], query_xy[1] - predicted_xy[1])
                    )
                    candidate["error_m"] = candidate_error
                    candidate["success_20m"] = bool(candidate_error < 20.0)
                    candidate["improves_coarse_top1"] = bool(candidate_error <= coarse_top1_error)
                    candidate["catastrophic_50m"] = bool(candidate_error > coarse_top1_error + 50.0)
                    candidates.append(candidate)

            raw_best = None
            for candidate in candidates:
                if prefer_inlier_candidate(candidate, raw_best):
                    raw_best = candidate
            oracle_best = min(candidates, key=lambda item: float(item["error_m"])) if candidates else None
            record = {
                "query_index": int(query_index),
                "query_name": query_name,
                "query_group": "_".join(Path(query_name).name.split("_")[:2]),
                "query_xy": [float(query_xy[0]), float(query_xy[1])],
                "coarse_top1_xy": [float(coarse_top1_xy[0]), float(coarse_top1_xy[1])],
                "coarse_top1_error_m": coarse_top1_error,
                "retrieval_indices": [int(value) for value in top_indices[: int(args.retrieval_topk)]],
                "retrieval_scores": [float(value) for value in top_scores[: int(args.retrieval_topk)]],
                "vop_time_s": float(vop_time),
                "match_time_s": float(match_time),
                "raw_best_candidate_index": None if raw_best is None else int(candidates.index(raw_best)),
                "oracle_best_candidate_index": None if oracle_best is None else int(candidates.index(oracle_best)),
                "candidates": candidates,
            }
            output_handle.write(json.dumps(serializable(record), ensure_ascii=False) + "\n")
            output_handle.flush()
            written += 1
            if written <= 3 or written % 25 == 0:
                elapsed = time.perf_counter() - start_time
                print(
                    f"[{written}/{len(selected_indices)}] {query_name} "
                    f"coarse={coarse_top1_error:.2f}m oracle={float(oracle_best['error_m']):.2f}m "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    print(f"Saved {written} new records to {output_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
