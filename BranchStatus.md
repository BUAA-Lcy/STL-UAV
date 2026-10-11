# 分支成果与同步记录

记录日期：2026-10-11（Asia/Shanghai）。本文件所在分支：`feature/localization-pose-prior`。
同步目标：`BUAA-Lcy/STL-UAV` 的同名分支（origin）。

## 当前分支状态

最近实现快照包含 VisLoc yaw / rotate 对齐、稀疏 matcher 控制及
`Game4Loc/scripts/run_visloc_matcher_tuning.py` 与
`Game4Loc/scripts/run_visloc_matcher_tuning_v2.py` 调参入口。
最近提交标题为“match部分最好参数”，这不构成已核验的正式最佳指标或论文结论。
本次未重新运行调参，也未把后续 VOP 或 Demo 分支的结果写成本分支的新成果。

原项目环境与训练评估说明见 [README.md](README.md) 后续章节。
最新论文研究结果请查看 `codex/vop-experiment` 分支；当前真实 Demo 查看 `NewRealDemo`。

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
dad79f9 2026-04-07 match部分最好参数
a35337a 2026-04-02 VisLoc rotate 改到正确的用法
c123ae0 2026-03-27 rotate VisLoc 3-27
e8292e3 2026-03-25 rotate模式
cae91c3 2026-03-25 delete W&B  increase max macthers
```

## 归档边界

Git 推送包含各分支已有的受版本控制文件及本次文档。
被忽略的数据集、权重、卫星缓存、运行依赖和生成视频仍在本地，不能把分支同步理解为
这些大文件已上传。各分支复现时先阅读对应 README 和成果说明，避免复用不同协议的指标。
本次为文档整理：检查本地链接、Git diff 空白错误与提交范围；不重新运行历史训练或实验。
