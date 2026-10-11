# 分支成果与同步记录

记录日期：2026-10-11（Asia/Shanghai）。本文件所在分支：`main`。
同步目标：`BUAA-Lcy/STL-UAV` 的同名分支（origin）。

## 当前分支状态

该分支保留原项目说明和早期本地匹配实现。最近提交仅修改 README，
历史记录包含 Sparse / SuperPoint＋LightGlue 匹配修改。此处不宣称原始 upstream
的未修改镜像，也不将其他分支的正式结果归入本分支。

原数据集与训练评估说明保留在 [README.md](README.md) 后续章节。
当前真实录像演示切换 `NewRealDemo`；正式精定位研究切换 `codex/vop-experiment`。

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
500409c 2026-05-12 1
4d3e830 2026-03-20 Depth实验前备份
6bca076 2026-03-20 Sp+lg -> Sparse
e3ab73b 2026-03-17 sp+lg
e652ac8 2026-03-14 Record the time of each sub-step of the match process.
```

## 归档边界

Git 推送包含各分支已有的受版本控制文件及本次文档。
被忽略的数据集、权重、卫星缓存、运行依赖和生成视频仍在本地，不能把分支同步理解为
这些大文件已上传。各分支复现时先阅读对应 README 和成果说明，避免复用不同协议的指标。
本次为文档整理：检查本地链接、Git diff 空白错误与提交范围；不重新运行历史训练或实验。
