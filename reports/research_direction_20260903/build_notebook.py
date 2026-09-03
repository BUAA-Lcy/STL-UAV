from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name("uav_gnss_denied_research_20260903.ipynb")


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    md(
        """
# GNSS 拒止环境下 UAV 视觉自定位与导航：论文方向审计（2026-09-03）

## tl;dr

**结论：当前方向不是完全错误，但其研究单位太窄。** 继续把贡献定义为“retrieval top-1 后，选择若干旋转再做单应性”很容易陷入 matcher / RANSAC 微调，且 2025–2026 年的新工作已经系统性转向：多候选空间推理、位置与航向联合估计、多帧/3D 几何、2.5D 地图注册，以及与里程计融合后的失效恢复。

最推荐的转向是：

> **将 VOP 改造成置信度可校准的联合位置–航向观测模型；在相邻卫星候选上输出多模态位姿似然，并通过运动先验进行时序融合与失效恢复。**

这不是“再加一个 heuristic”。它把论文问题从单帧点估计改为拒止导航所真正需要的 **belief update（置信分布更新）**。现有 VOP、teacher、稀疏几何和 evaluator 都能部分复用。

建议分两步：

1. **4–6 周可证伪主线（推荐）**：先做单帧的 multi-tile joint pose likelihood + calibration + abstention，验证是否解决 tile 边界、错误 top-1、灾难性 homography。
2. **通过后再做导航版**：加入 VO/VIO 或合成运动先验，评估轨迹误差、重定位时间、丢失后恢复率；不再只看逐帧 Dis@1。

不建议继续把 denser SuperPoint、更多尺度、更多 RANSAC 分支当主贡献；现有实验已表明“更多 inlier”并不稳定转化为更准的位置。
"""
    ),
    md(
        """
## Context & Methods

本笔记本是一个面向研究决策的文献审计，而不是完整系统综述。检索使用 Hugging Face Papers 混合搜索，并回到 arXiv 摘要/HTML 正文核对方法、限制和实验设定。重点时间窗为 2025-01 至 2026-09，同时保留 FoundLoc 等直接影响系统设计的早期工作。

检索词包括：`GPS-denied UAV visual localization navigation`、`UAV satellite cross-view geo-localization`、`UAV navigation foundation model`、`sequence temporal UAV satellite localization`、`orthographic geodata UAV 6-DoF`。

### Key Assumptions

- “拒止导航”要求连续、可恢复、可表达不确定性的状态估计；逐帧 top-1 精度只是其中一个子问题。
- arXiv 2026 论文中相当一部分仍是预印本；本笔记本将“新颖信号”与“已充分验证的结论”分开。
- 当前仓库的正式研究资产限定在 UAV–satellite fine localization；除非明确重开数据协议，否则不把 RGB-D/LiDAR 方案当作可直接复现的公平基线。
- 当前 GTA-UAV / UAV-VisLoc 官方测试 JSON 是稀疏抽样，不应被直接当作连续轨迹基准。
"""
    ),
    code(
        """
from pathlib import Path
import json
import math
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_colwidth", 80)
REPO = Path.cwd()
SEARCH_AS_OF = "2026-09-03"
assert (REPO / "AGENTS.md").exists(), "请从仓库根目录执行本笔记本"
print(f"repository={REPO}")
print(f"literature search as of {SEARCH_AS_OF}")
"""
    ),
    md("## Data\n\n### 1. 文献证据表"),
    code(
        """
papers = [
    dict(id="2310.16299", year=2023, title="FoundLoc", paradigm="VIO + VPR fusion", map="satellite tiles", temporal="yes", output="global position", status="arXiv", signal="绝对视觉观测应与 VIO 融合，而不是逐帧独立决策"),
    dict(id="2405.11936", year=2024, title="UAV-VisLoc", paradigm="retrieval benchmark", map="satellite maps", temporal="no", output="2D position", status="dataset paper", signal="大区域、地形和高度变化比理想一一配对更接近实际"),
    dict(id="2409.16925", year=2024, title="Game4Loc", paradigm="retrieval + metric localization", map="satellite tiles", temporal="no", output="2D position", status="benchmark", signal="连续区域与 partial match 使米制定位成为必要指标"),
    dict(id="2509.18350", year=2025, title="OrthoLoC", paradigm="2.5D registration + PnP", map="orthophoto + DSM", temporal="no", output="6-DoF", status="NeurIPS 2025", signal="covisibility、地图分辨率和 2.5D 几何是核心变量；低于 20% 共视明显退化"),
    dict(id="2510.01348", year=2025, title="Heightmap Gradients", paradigm="odometry + particle filter", map="prior heightmap", temporal="yes", output="trajectory belief", status="field system", signal="9 km 任务中恢复能力比瞬时低 RMSE 更重要；系统获挑战赛第一"),
    dict(id="2512.02737", year=2025, title="CAEVL / ViLD", paradigm="reference-only self-supervision", map="satellite imagery", temporal="no", output="place", status="WACV 2026", signal="仅参考图训练与域随机化可减少 paired UAV 数据依赖"),
    dict(id="2510.22582", year=2026, title="MobileGeo", paradigm="distillation + multi-view selection", map="satellite database", temporal="multi-view", output="place", status="arXiv/code", signal="边缘部署与不确定性感知蒸馏是独立研究轴，但仍偏检索"),
    dict(id="2603.07535", year=2026, title="Scale-Aware Semantic Geometry", paradigm="semantic metric scale", map="satellite crops", temporal="no", output="scale + place", status="arXiv", signal="尺度/FOV 错配可由物理语义锚点显式约束"),
    dict(id="2603.20778", year=2026, title="PiLoT", paradigm="pixel-to-3D registration", map="3D mesh", temporal="video", output="6-DoF + target", status="arXiv/code", signal="视频到地理 3D 图直接注册，>25 FPS，但依赖重型 3D mesh"),
    dict(id="2603.22153", year=2026, title="Bearing-UAV", paradigm="neighbor-map joint regression", map="4 neighboring tile features", temporal="navigation loop", output="2D position + heading", status="CVPR 2026", signal="明确反对 matching-to-tile；联合预测位置和航向"),
    dict(id="2604.01747", year=2026, title="3D Geometric Perception", paradigm="VGGT sequence -> BEV", map="satellite candidates", temporal="multi-view", output="3-DoF", status="arXiv", signal="共享几何表征统一 retrieval/alignment/pose，并隔离互斥候选"),
    dict(id="2605.07099", year=2026, title="InfoGeo", paradigm="object-centric information bottleneck", map="satellite database", temporal="no", output="place", status="arXiv", signal="对象结构比区域纹理更利于跨城市、天气泛化"),
    dict(id="2606.05011", year=2026, title="CIPER", paradigm="joint retrieval + pose transformer", map="aerial database", temporal="no", output="3-DoF set", status="arXiv/code", signal="共享编码、任务 token 与 set prediction 缓解级联误差"),
    dict(id="2606.31098", year=2026, title="PiLoT v2", paradigm="pixel-to-orthomap optimization", map="TDOM + DSM", temporal="video + sensors", output="6-DoF", status="arXiv", signal="1210 个局部位姿假设 + coarse-to-fine LM；60.51 ms 定位线程"),
]

papers_df = pd.DataFrame(papers)
papers_df["url"] = papers_df["id"].apply(lambda x: "https://arxiv.org/abs/" + x if "?" not in x else "https://arxiv.org/search/")
papers_df[["year", "title", "paradigm", "map", "temporal", "output", "status", "signal", "url"]]
"""
    ),
    md(
        """
### 2. 当前仓库的可复用证据

为避免混用实验快照，下表明确区分 2026-04-11/12 的 matched 四行表与 2026-04-18 的 sparse-side follow-up。后者没有同步刷新 dense / rotate 行。
"""
    ),
    code(
        """
current_results = pd.DataFrame([
    dict(snapshot="matched-20260411/12", method="dense DKM", dis1_m=50.11, ma20_pct=54.81, fallback_pct=None, sec_per_query=4.0410),
    dict(snapshot="matched-20260411/12", method="sparse", dis1_m=108.47, ma20_pct=14.61, fallback_pct=None, sec_per_query=0.0651),
    dict(snapshot="matched-20260411/12", method="sparse + rotate90/inlier", dis1_m=77.50, ma20_pct=36.45, fallback_pct=None, sec_per_query=0.2831),
    dict(snapshot="matched-20260411/12", method="sparse + VOP(q2000)", dis1_m=62.59, ma20_pct=43.54, fallback_pct=12.02, sec_per_query=0.3044),
    dict(snapshot="sparse-followup-20260418", method="sparse rerun", dis1_m=73.01, ma20_pct=37.87, fallback_pct=13.16, sec_per_query=0.2528),
    dict(snapshot="sparse-followup-20260418", method="sparse + VOP(full teacher)", dis1_m=57.63, ma20_pct=45.92, fallback_pct=9.53, sec_per_query=0.2652),
])
current_results
"""
    ),
    md("### 3. 序列数据可用性快速审计"),
    code(
        """
dataset_paths = {
    "GTA-test": REPO / "Game4Loc/data/GTA-UAV-data/same-area-drone2sate-test.json",
    "VisLoc-03/04-test": REPO / "Game4Loc/data/UAV_VisLoc_dataset/same-area-drone2sate-test.json",
}

audit_rows = []
for name, path in dataset_paths.items():
    rows = json.loads(path.read_text())
    filenames = [r["drone_img_name"] for r in rows]
    if name.startswith("GTA"):
        group_count = len({"_".join(x.split("_")[:2]) for x in filenames})
        indices = sorted(int(x.split("_")[-1].split(".")[0]) for x in filenames)
    else:
        group_count = len({x.split("_")[0] for x in filenames})
        indices = sorted(int(x.split("_")[-1].split(".")[0]) for x in filenames)
    gaps = [b - a for a, b in zip(indices, indices[1:]) if b >= a]
    audit_rows.append(dict(dataset=name, queries=len(rows), apparent_groups=group_count, median_index_gap=float(pd.Series(gaps).median())))

sequence_audit = pd.DataFrame(audit_rows)
sequence_audit
"""
    ),
    md(
        """
上述文件名保留了某种顺序信息，但现有 benchmark split 是抽样后的单帧任务，且索引邻接不等价于稳定的物理连续轨迹。因此：可以用于快速构造弱时序 smoke test，但不能直接支撑“长航程导航”主张。若做时序主线，需要重新导出连续 GTA 轨迹，或使用仓库中的真实 DJI flight records 建立独立导航验证协议。
"""
    ),
    md("## Results\n\n### 4. 研究范式迁移"),
    code(
        """
paradigm_groups = {
    "tile / retrieval only": ["UAV-VisLoc", "Game4Loc", "MobileGeo", "InfoGeo"],
    "joint spatial pose": ["Bearing-UAV", "CIPER"],
    "explicit 2.5D / 3D": ["OrthoLoC", "PiLoT", "3D Geometric Perception", "PiLoT v2"],
    "temporal belief fusion": ["FoundLoc", "Heightmap Gradients"],
    "data / scale robustness": ["CAEVL / ViLD", "Scale-Aware Semantic Geometry"],
}
counts = pd.Series({k: len(v) for k, v in paradigm_groups.items()}).sort_values()
ax = counts.plot.barh(figsize=(8, 4), color="#4472C4")
ax.set_title("Curated literature signals by research paradigm")
ax.set_xlabel("paper count (curated, not bibliometric)")
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """
这个图只表示本次精读样本的信号分布，不是文献计量结论。它说明值得关注的创新空间已经从“换 backbone / matcher”移动到 **空间结构、显式几何与时序 belief**。

### 5. 候选方向决策矩阵

评分为 1–5。`evidence` 表示外部文献和真实系统是否支持该问题的重要性；`novelty_room` 表示在当前项目语境中仍有多少可写空间；`feasibility` 与 `asset_reuse` 基于本仓库；`validation_readiness` 表示是否能在现有协议上快速证伪。分数是研究规划启发式，不是论文质量预测。
"""
    ),
    code(
        """
directions = pd.DataFrame([
    dict(direction="继续 VOP + matcher 微调", evidence=2.0, novelty_room=1.5, feasibility=5.0, asset_reuse=5.0, validation_readiness=5.0, fatal_risk="容易成为阈值/组件堆叠"),
    dict(direction="多 tile 联合位置–航向似然（推荐第一步）", evidence=4.5, novelty_room=4.2, feasibility=4.0, asset_reuse=4.8, validation_readiness=4.5, fatal_risk="需证明不是 Bearing-UAV/CIPER 的缩小复刻"),
    dict(direction="视觉似然 + VO/VIO 时序 Bayes 融合", evidence=5.0, novelty_room=4.5, feasibility=2.8, asset_reuse=4.0, validation_readiness=2.0, fatal_risk="现有 benchmark 缺连续轨迹"),
    dict(direction="TDOM/DSM 2.5D 6-DoF 注册", evidence=5.0, novelty_room=3.0, feasibility=2.0, asset_reuse=2.2, validation_readiness=1.8, fatal_risk="地图/传感器/6DoF GT 缺口大"),
    dict(direction="VGGT 多帧 3D→BEV 统一定位", evidence=4.5, novelty_room=2.5, feasibility=1.8, asset_reuse=2.0, validation_readiness=1.5, fatal_risk="算力高且与 2026 工作正面竞争"),
    dict(direction="对象/道路结构鲁棒匹配", evidence=3.5, novelty_room=3.2, feasibility=3.0, asset_reuse=3.0, validation_readiness=3.0, fatal_risk="检测域偏差可能取代纹理域偏差"),
])

weights = dict(evidence=0.25, novelty_room=0.25, feasibility=0.18, asset_reuse=0.17, validation_readiness=0.15)
directions["weighted_score"] = sum(directions[k] * w for k, w in weights.items())
directions.sort_values("weighted_score", ascending=False).reset_index(drop=True)
"""
    ),
    md(
        """
### 6. 推荐的新论文问题

**建议题目方向（工作名）**

> *From Fine Localization to Belief Update: Uncertainty-Calibrated Multi-Hypothesis UAV Geo-Localization for GNSS-Denied Navigation*

**核心问题**

给定检索得到的局部地图邻域，不直接输出一个 tile、一个旋转和一个 homography；而是输出联合状态观测：

\[
p(x, y, \\psi, v \\mid I_t, \\mathcal{M}_{\\mathrm{local}}),
\]

其中 \(v\) 是“当前视觉观测是否可用于绝对校正”的有效性变量。VOP 变成航向边缘分布/初始化器；几何 matcher 不再只有 accept/fallback，而是产生可校准的似然证据。

**与 Bearing-UAV / CIPER 的差异必须做实**

- 不是只做 deterministic coordinate regression，而是输出多模态、可校准的 joint likelihood。
- 不假设正确 tile 已经锁死：对 top-R 邻域候选进行空间一致的联合推理。
- 明确训练 `valid / abstain / recover`，使坏 homography 不会把状态拉飞。
- 第二阶段把这个似然作为粒子滤波器或因子图的 measurement model，报告恢复时间与轨迹风险。

**可复用资产**

- VOP teacher 的 useful-angle 与 pair-confidence supervision，可改造成离散航向 likelihood。
- 稀疏/稠密 matcher 的 retained matches、inlier ratio、projection validity、fallback 等日志，可用于 validity calibration。
- GTA-UAV 连续区域坐标和卫星多尺度 tile 几何，可构造跨 tile 邻域。
- 现有 dense DKM 可作为高成本 teacher，而不是必须作为推理主干。
"""
    ),
    md(
        """
### 7. 最小可证伪实验（先不要训练大模型）

1. **Oracle headroom**：对 retrieval top-R（建议 R=5 或空间邻接候选）× VOP top-k，统计 oracle Dis@1、oracle MA@20、正确候选是否经常不在 retrieval top-1。若 oracle 相对当前 57.63m 没有显著余量，立即停止。
2. **Calibration audit**：用现有日志特征预测 `final_error < 20m` 与 `worse_than_coarse`，比较 inlier count、ratio、VOP posterior、cycle consistency、dense-teacher score 的 AUROC/AUPRC/ECE。若无法校准，就不适合做 belief update。
3. **Multi-tile likelihood baseline**：不改 retrieval backbone；仅在 top-R / 相邻 tile 上产生候选，并用温度标定后的 geometry score 归一化。比较 point estimate 与 posterior expected risk。
4. **Synthetic motion stress test**：在 GTA 平面坐标上构造带噪运动先验，注入连续 N 帧视觉失效，比较逐帧定位、硬门限滤波、粒子滤波的最大误差与恢复帧数。
5. **Go / no-go**：只有同时满足“oracle 有余量”“置信度可校准”“连续失效后可恢复”，才进入完整时序模型。

建议的新指标：NLL、ECE、risk–coverage、catastrophic correction rate、time-to-relocalize、trajectory RMSE / max error、每次绝对校正成本。保留 Dis@1 与 MA@20，但不再让它们独占论文结论。
"""
    ),
    md(
        """
## Takeaways

1. **停止主线微调 matcher。** 当前瓶颈更像“错误假设被过度相信”，而不仅是“匹配点不足”。
2. **保留 VOP，但改变它的角色。** 它应成为多模态航向/位姿 likelihood 的组成部分，而不是 top-k 旋转调度器本身。
3. **短期先做 multi-tile + calibrated abstention。** 这是最能复用仓库、又能回应 2026 文献趋势的方向。
4. **长期论文要走向 navigation-level evaluation。** Heightmap Gradients 的实飞经验尤其重要：长程任务中，丢失后的恢复比每帧平均误差更关键。
5. **谨慎对待完全重启到 2.5D/3D。** OrthoLoC / PiLoT v2 证明了这条路线的价值，但当前仓库缺 DSM、连续 6-DoF GT 和匹配协议；除非愿意把数据工程作为论文的一半，否则风险显著更高。

### Primary sources

- [FoundLoc](https://arxiv.org/abs/2310.16299)
- [Game4Loc](https://arxiv.org/abs/2409.16925)
- [UAV-VisLoc](https://arxiv.org/abs/2405.11936)
- [OrthoLoC](https://arxiv.org/abs/2509.18350)
- [Kilometer-Scale GNSS-Denied UAV Navigation via Heightmap Gradients](https://arxiv.org/abs/2510.01348)
- [Beyond Paired Data / CAEVL](https://arxiv.org/abs/2512.02737)
- [Scale-Aware UAV-to-Satellite Cross-View Geo-Localization](https://arxiv.org/abs/2603.07535)
- [PiLoT](https://arxiv.org/abs/2603.20778)
- [Bearing-UAV](https://arxiv.org/abs/2603.22153)
- [Unifying UAV Cross-View Geo-Localization via 3D Geometric Perception](https://arxiv.org/abs/2604.01747)
- [InfoGeo](https://arxiv.org/abs/2605.07099)
- [CIPER](https://arxiv.org/abs/2606.05011)
- [PiLoT v2](https://arxiv.org/abs/2606.31098)

检索截止：2026-09-03。对 2026 预印本的结果应在正式写作前再次核对版本、代码和协议。
"""
    ),
]

nb = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
)
nbf.write(nb, OUT)
print(OUT)
