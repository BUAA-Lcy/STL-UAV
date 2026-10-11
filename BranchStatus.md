# 分支成果与同步记录

记录日期：2026-10-11（Asia/Shanghai）。本文件所在分支：`feature/purification-network`。
同步目标：`BUAA-Lcy/STL-UAV` 的同名分支（origin）。

## 当前分支状态

最近提交为 `WIP: modify visloc.py`，涉及
`scripts/prepare_dataset/visloc.py` 数据准备修改。历史记录还包含 pose-gate /
retrieval 实验与云部署脚本；此分支仍保留实验快照，不将 WIP 视为已验证完成。

云部署入口见 [README_cloud.md](README_cloud.md)，包括 bootstrap、smoke test
与 pose-gate 实验命令。这些命令是历史复现入口，本次没有运行它们或核验新的实验指标。
最新 Demo 和正式精定位研究分别见 `NewRealDemo` 与 `codex/vop-experiment` 分支。

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
21ff62b 2026-04-06 WIP: modify visloc.py
f3a05e4 2026-04-06 4-6 retrieval暂时搁浅
5e41045 2026-03-23 feat(tools): 添加同区域残差主实验对比脚本
c27518f 2026-03-22 retrieval down
8bf2193 2026-03-21 上云
```

## 归档边界

Git 推送包含各分支已有的受版本控制文件及本次文档。
被忽略的数据集、权重、卫星缓存、运行依赖和生成视频仍在本地，不能把分支同步理解为
这些大文件已上传。各分支复现时先阅读对应 README 和成果说明，避免复用不同协议的指标。
本次为文档整理：检查本地链接、Git diff 空白错误与提交范围；不重新运行历史训练或实验。
