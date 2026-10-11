# 分支成果与同步记录

记录日期：2026-10-11（Asia/Shanghai）。本文件所在分支：`NewRealDemo`。
同步目标：`BUAA-Lcy/STL-UAV` 的同名分支（origin）。

## 当前分支状态

当前成果为 Aerial-to-ground2 录像的源帧 9151–10948，共 1798 帧，
60000/1001 fps、29.996633 秒。Google Earth Pro 导出 Airbus 2024-09-12 影像，
原图 8192×6133，78 张校准后瓦片；2026-10-06 完整解码、帧映射和七组抽查通过。
本次仅整理分支总结，没有重新生成影像或运行新的算法实验。

最快复现与卫星影像路径见 [README.md](README.md)，完整成果见
[NewRealDemo.md](NewRealDemo.md)，验证与索引见
[归档目录](Game4Loc/new_real_demo/results/google_earth_20260927)。
原始 MP4、Google 导出图、影像缓存与生成视频需从旧机器复制；这些文件未上传 GitHub。
当前算法为 GPS 辅助对应，尚未引入 UAV/卫星视觉匹配。

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
0391fd6 2026-10-06 docs(demo): document new-machine setup and imagery acquisition
060aa32 2026-10-06 feat(demo): archive GPS-assisted Google Earth real-video demo
38002b4 2026-09-03 feat(demo): refine real-data localization visualization
c677511 2026-05-12 5-12
0c357ea 2026-04-19 4-19定稿前
```

## 归档边界

Git 推送包含各分支已有的受版本控制文件及本次文档。
被忽略的数据集、权重、卫星缓存、运行依赖和生成视频仍在本地，不能把分支同步理解为
这些大文件已上传。各分支复现时先阅读对应 README 和成果说明，避免复用不同协议的指标。
本次为文档整理：检查本地链接、Git diff 空白错误与提交范围；不重新运行历史训练或实验。
