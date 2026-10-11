# 分支成果与同步记录

记录日期：2026-10-11（Asia/Shanghai）。本文件所在分支：`codex/vop-experiment`。
同步目标：`BUAA-Lcy/STL-UAV` 的同名分支（origin）。

## 当前分支状态

最新正式结论来自 2026-09-04 的 GTA-UAV 同区域匹配协议实验，
三行各 3443 个查询，检索固定。Legacy top-1 VOP：Dis@1 56.9898 m、
MA@20 46.5582%；校准自适应：48.8255 m、48.6494%，最终决定 KEEP。
未校准候选扩展：73.8959 m，作为主方法的决定为 REJECT。
自适应降低误差和灾难性修正，但 fallback 从 330 升到 537，
VOP＋matcher 时间从 0.260941 升到 0.452635 s/query；这些计时不是端到端延迟。
结论限于该同区域协议，不表示已验证跨区域或连续导航。

正式来源见 [official_summary.json](Game4Loc/reports/gta_pose_likelihood_20260903/official_summary.json)，
过程与解释见 [Process.md](Process.md)，实验入口为
[run_gta_pose_full_evaluation.sh](Game4Loc/scripts/run_gta_pose_full_evaluation.sh)。
研究约束见 [AGENTS.md](AGENTS.md)。本次仅整理已完成结论，没有新增训练或评估。

## 全部分支入口

| 分支 | 本次整理前的实现快照 | 用途 |
| --- | --- | --- |
| [NewRealDemo](https://github.com/BUAA-Lcy/STL-UAV/tree/NewRealDemo) | `0391fd6` | 当前真实录像 Demo：最后 30 秒、内嵌 GPS / 云台姿态、Google Earth Airbus 底图。 |
| [RealDataDemo](https://github.com/BUAA-Lcy/STL-UAV/tree/RealDataDemo) | `38002b4` | 历史真实录像演示：飞行日志 GPS 选图、DKM / SP+LG 匹配及可视化。 |
| [codex/vop-experiment](https://github.com/BUAA-Lcy/STL-UAV/tree/codex/vop-experiment) | `9bb3a8f` | 研究分支：检索后精定位、VOP 与校准的自适应候选选择。 |
| [feature/localization-pose-prior](https://github.com/BUAA-Lcy/STL-UAV/tree/feature/localization-pose-prior) | `dad79f9` | 历史姿态先验 / 旋转匹配与 VisLoc matcher 调参分支。 |
| [feature/purification-network](https://github.com/BUAA-Lcy/STL-UAV/tree/feature/purification-network) | `21ff62b` | 历史 purification / pose-gate 与数据准备实验，最近提交仍标记为 WIP。 |
| [main](https://github.com/BUAA-Lcy/STL-UAV/tree/main) | `500409c` | 原项目与早期匹配实现的主分支快照；新 Demo 和最新研究结果在独立分支。 |

上表哈希是本次文档整理前的实现快照，便于定位原有成果；后续文档提交会更新分支 HEAD。
各分支保持独立，未合并或重写彼此历史。整理前六个分支均与 origin 同步，
唯一注册工作树干净，无 Git stash。本次新增的是 README 和成果导航说明。

## 最近历史提交

```text
9bb3a8f 2026-09-04 docs(gta): report final matched same-area KEEP result
eeae707 2026-09-04 fix(gta): resolve detailed official audit log sources
59f28ee 2026-09-04 docs(gta): record completed raw multi-tile baseline
65e1e75 2026-09-04 docs(gta): record official baseline and log-source audit
3746dfc 2026-09-04 docs(gta): hand off official full-test evaluation
```

## 归档边界

Git 推送包含各分支已有的受版本控制文件及本次文档。
被忽略的数据集、权重、卫星缓存、运行依赖和生成视频仍在本地，不能把分支同步理解为
这些大文件已上传。各分支复现时先阅读对应 README 和成果说明，避免复用不同协议的指标。
本次为文档整理：检查本地链接、Git diff 空白错误与提交范围；不重新运行历史训练或实验。
