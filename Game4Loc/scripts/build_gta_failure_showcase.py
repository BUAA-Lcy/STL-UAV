#!/usr/bin/env python3
"""Build a mentor-facing GTA-UAV failure-case report from an official eval log.

This script is intentionally analysis-only: it does not rerun retrieval or matching.
It combines per-query evidence from the complete evaluator log with the original
drone image, annotated positive satellite tile, and retrieved top-1 tile.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont


QUERY_RE = re.compile(
    r"Query\[(?P<index>\d+)/(?P<total>\d+)\] 结果: "
    r"Query=(?P<query>[^ |]+) \| Top1Gallery=(?P<gallery>[^ |]+) \| "
    r"检索Dis=(?P<coarse>[\d.]+)m \| 匹配后Dis=(?P<fine>[\d.]+)m"
    r"(?P<fallback> \(回退粗检索\))? \| retained=(?P<retained>\d+) \| "
    r"inliers=(?P<inliers>\d+)"
)


@dataclass
class Record:
    index: int
    total: int
    query: str
    gallery: str
    coarse: float
    fine: float
    fallback: bool
    retained: int
    inliers: int

    @property
    def inlier_ratio(self) -> float:
        return self.inliers / max(self.retained, 1)


def parse_args() -> argparse.Namespace:
    game4loc = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        type=Path,
        default=game4loc
        / "Log"
        / "vit_base_patch16_rope_reg1_gap_256_sbb_in1k_eval_GTA-UAV_same_match_on_20260418_0139.log",
    )
    parser.add_argument(
        "--data-root", type=Path, default=game4loc / "data" / "GTA-UAV-data"
    )
    parser.add_argument(
        "--pairs-json", type=str, default="same-area-drone2sate-test.json"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=game4loc / "docs" / "gta_failure_showcase"
    )
    return parser.parse_args()


def parse_log(path: Path) -> list[Record]:
    records = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = QUERY_RE.search(line)
        if not match:
            continue
        item = match.groupdict()
        records.append(
            Record(
                index=int(item["index"]),
                total=int(item["total"]),
                query=item["query"],
                gallery=item["gallery"],
                coarse=float(item["coarse"]),
                fine=float(item["fine"]),
                fallback=bool(item["fallback"]),
                retained=int(item["retained"]),
                inliers=int(item["inliers"]),
            )
        )
    if not records:
        raise RuntimeError(f"No per-query records parsed from {path}")
    return records


def load_metadata(path: Path) -> dict[str, dict]:
    return {
        item["drone_img_name"]: item
        for item in json.loads(path.read_text(encoding="utf-8"))
    }


def crop_square(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    side = min(image.size)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return image.crop((left, top, left + side, top + side))


def fit_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = crop_square(image)
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def build_case_panel(
    record: Record,
    meta: dict,
    data_root: Path,
    title: str,
    diagnosis: str,
    output: Path,
) -> None:
    query_path = data_root / meta["drone_img_dir"] / record.query
    gt_name = meta["pair_pos_sate_img_list"][0]
    gt_path = data_root / meta["sate_img_dir"] / gt_name
    retrieved_path = data_root / meta["sate_img_dir"] / record.gallery

    panel = Image.new("RGB", (1500, 680), "#f7f7f7")
    draw = ImageDraw.Draw(panel)
    draw.text((45, 25), title, fill="#111111", font=font(32, bold=True))
    labels = ["UAV query", "Annotated positive tile", "Retrieved top-1 tile"]
    paths = [query_path, gt_path, retrieved_path]
    for col, (label, path) in enumerate(zip(labels, paths)):
        x = 45 + col * 485
        panel.paste(fit_image(Image.open(path), (440, 440)), (x, 90))
        draw.text((x, 540), label, fill="#222222", font=font(23, bold=True))
        draw.text((x, 572), path.name, fill="#555555", font=font(17))

    stat = (
        f"coarse {record.coarse:.2f} m  ->  fine {record.fine:.2f} m    "
        f"retained {record.retained}    inliers {record.inliers}    "
        f"ratio {record.inlier_ratio:.3f}    fallback {'yes' if record.fallback else 'no'}"
    )
    draw.rounded_rectangle((45, 610, 1455, 660), radius=12, fill="#e8eef5")
    draw.text((65, 619), stat, fill="#17324d", font=font(21, bold=True))
    draw.text((1015, 47), diagnosis, fill="#9f2f21", font=font(21, bold=True))
    panel.save(output, quality=95)


def plot_overview(records: list[Record], output: Path) -> None:
    coarse = np.asarray([x.coarse for x in records])
    fine = np.asarray([x.fine for x in records])
    fallback = np.asarray([x.fallback for x in records])
    worse = fine > coarse + 1e-6

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.2))
    capped_coarse = np.minimum(coarse, 600)
    capped_fine = np.minimum(fine, 600)
    axes[0].scatter(
        capped_coarse[~worse], capped_fine[~worse], s=9, alpha=0.28, color="#2b7a78"
    )
    axes[0].scatter(
        capped_coarse[worse], capped_fine[worse], s=12, alpha=0.55, color="#d1495b"
    )
    axes[0].plot([0, 600], [0, 600], "--", color="#333333", linewidth=1)
    axes[0].set(
        xlabel="Coarse retrieval error (m, clipped at 600)",
        ylabel="Fine-localization error (m, clipped at 600)",
        title="Per-query error: red means geometry made it worse",
    )

    labels = [
        "fallback",
        "worse than coarse",
        "fine error > 100 m",
        "coarse error > 200 m",
        "coarse < 20 m,\nfine > 50 m",
    ]
    values = [
        fallback.mean() * 100,
        worse.mean() * 100,
        (fine > 100).mean() * 100,
        (coarse > 200).mean() * 100,
        ((coarse < 20) & (fine > 50)).mean() * 100,
    ]
    bars = axes[1].barh(labels, values, color=["#7f8c8d", "#d1495b", "#e07a5f", "#6c5b7b", "#f2cc8f"])
    axes[1].set(xlabel="Share of all evaluated queries (%)", title="Operational failure indicators")
    axes[1].bar_label(bars, fmt="%.1f%%", padding=4)
    axes[1].set_xlim(0, max(values) * 1.22)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pipeline(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.1))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")
    nodes = [
        (0.4, "UAV query"),
        (3.0, "Retrieval top-1"),
        (5.8, "VOP top-k angles"),
        (8.7, "Sparse matches"),
        (11.4, "Homography +\nprojection"),
    ]
    for x, label in nodes:
        box = FancyBboxPatch(
            (x, 1.55),
            2.0,
            1.0,
            boxstyle="round,pad=0.08",
            facecolor="#e8eef5",
            edgecolor="#294c60",
            linewidth=1.6,
        )
        ax.add_patch(box)
        ax.text(x + 1, 2.05, label, ha="center", va="center", fontsize=12, weight="bold")
    for x in [2.4, 5.0, 7.8, 10.7]:
        ax.add_patch(FancyArrowPatch((x, 2.05), (x + 0.55, 2.05), arrowstyle="->", mutation_scale=15))
    failures = [
        (3.95, 0.75, "F1 wrong / distant tile"),
        (6.8, 3.25, "F2 useful angle missing"),
        (9.7, 0.75, "F3 few or repetitive matches"),
        (12.4, 3.25, "F4 unstable geometry"),
    ]
    for x, y, label in failures:
        ax.text(x, y, label, ha="center", va="center", color="#b23a2b", fontsize=11, weight="bold")
        target_y = 1.55 if y < 2 else 2.55
        ax.add_patch(FancyArrowPatch((x, y + (0.18 if y < 2 else -0.18)), (x, target_y), arrowstyle="-|>", color="#b23a2b"))
    ax.text(
        7,
        0.15,
        "A fallback avoids some bad homographies, but then accuracy is capped by the coarse top-1 center.",
        ha="center",
        fontsize=11,
        color="#444444",
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = parse_log(args.log)
    by_index = {item.index: item for item in records}
    metadata = load_metadata(args.data_root / args.pairs_json)

    # Chosen from the complete 3,443-query log to show distinct mechanisms.
    selections = [
        (1023, "Case A — retrieval is already far away", "retrieval-limited"),
        (650, "Case B — good coarse result is destroyed", "low-support geometry"),
        (3012, "Case C — many inliers can still be wrong", "coherent false geometry"),
        (751, "Case D — accepted geometry causes a large jump", "unstable projection"),
        (1538, "Case E — high-inlier local alignment is biased", "partial/repetitive overlap"),
        (1840, "Case F — near-correct tile, wrong final position", "orientation / geometry ambiguity"),
    ]
    case_rows = []
    for number, (index, title, diagnosis) in enumerate(selections, start=1):
        record = by_index[index]
        output = args.output_dir / f"case_{number:02d}_q{index:04d}.jpg"
        build_case_panel(record, metadata[record.query], args.data_root, title, diagnosis, output)
        case_rows.append(
            {
                **record.__dict__,
                "inlier_ratio": record.inlier_ratio,
                "title": title,
                "diagnosis": diagnosis,
                "image": output.name,
                "gt_gallery": metadata[record.query]["pair_pos_sate_img_list"][0],
            }
        )

    plot_overview(records, args.output_dir / "failure_overview.png")
    plot_pipeline(args.output_dir / "failure_pipeline.png")

    coarse = np.asarray([x.coarse for x in records])
    fine = np.asarray([x.fine for x in records])
    summary = {
        "source_log": str(args.log),
        "query_count": len(records),
        "fallback_count": sum(x.fallback for x in records),
        "fallback_pct": 100 * np.mean([x.fallback for x in records]),
        "worse_than_coarse_count": int(np.sum(fine > coarse + 1e-6)),
        "worse_than_coarse_pct": 100 * np.mean(fine > coarse + 1e-6),
        "fine_error_over_100m_count": int(np.sum(fine > 100)),
        "fine_error_over_100m_pct": 100 * np.mean(fine > 100),
        "coarse_error_over_200m_count": int(np.sum(coarse > 200)),
        "coarse_error_over_200m_pct": 100 * np.mean(coarse > 200),
        "near_coarse_but_bad_fine_count": int(np.sum((coarse < 20) & (fine > 50))),
        "near_coarse_but_bad_fine_pct": 100 * np.mean((coarse < 20) & (fine > 50)),
        "selected_cases": case_rows,
    }
    (args.output_dir / "failure_statistics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
