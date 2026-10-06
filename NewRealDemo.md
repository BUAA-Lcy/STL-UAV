# NewRealDemo

当前目标：真实录像的 GPS 辅助卫星 / 航拍对应演示。分支 `NewRealDemo` 从
`RealDataDemo` 创建。本目录从零实现，只读取原始 MP4 和卫星影像缓存；
不导入原有检索、匹配、VOP、轨迹修正或其结果。

## 当前交付

- 原片：`Game4Loc/data/Aerial-to-ground2/dji_fly_20260420_185426_25_1776683433584_video.mp4`
- 按最新要求截取最后 30 秒：源时间 `152.669183–182.665817 s`。
- 源帧编号（从 0 开始）：`9151–10948`，共 `1798` 帧，`60000/1001 fps`。
- 输出：`Game4Loc/work_dir/new_real_demo/`。
- `index.html`：离线播放器，可播放、拖动、逐帧前后、变速、全屏。
- `new_real_demo.mp4`：1920×800，左卫星 / 右航拍，原画面完整 16:9。
- `satellite_frames.mp4`：Google 版为 928×562（522 像素画面＋40 像素署名），每一帧对应演示视频同一帧的卫星画面。
- `frames.csv`：每帧原始 GPS、显示 GPS、源帧编号、源时间、遥测时间差、
  姿态、卫星像素位置、裁剪投影矩阵和地面视野四角。
- `manifest.json` / `verification.json`：来源、参数、限制和验证结果。

## 使用

在 WSL 运行（可从任意目录启动）：

```bash
cd /home/lcy/Workplace/GTA-UAV
/home/lcy/miniconda3/envs/gtauav/bin/python Game4Loc/new_real_demo/build.py
```

输出目录的 `index.html` 可直接在浏览器打开，也可直接播放 MP4。若需 HTTP：

```bash
/home/lcy/miniconda3/envs/gtauav/bin/python Game4Loc/new_real_demo/serve.py
```

浏览器打开 `http://127.0.0.1:8765`。不上传数据。

其他命令：

```bash
# 只做逐帧索引和抽帧预览
/home/lcy/miniconda3/envs/gtauav/bin/python Game4Loc/new_real_demo/build.py --preview-only
# 后续扩展到整段视频；单独输出，保留最后 30 秒版本
/home/lcy/miniconda3/envs/gtauav/bin/python Game4Loc/new_real_demo/build.py \
  --last-seconds 0 --output Game4Loc/work_dir/new_real_demo_full
# 导出指定源帧的卫星 PNG 和原尺寸航拍 PNG；不覆盖播放器索引
/home/lcy/miniconda3/envs/gtauav/bin/python Game4Loc/new_real_demo/build.py --export-frame 9151
# 验证最终视频、帧号及抽查两侧内容
/home/lcy/miniconda3/envs/gtauav/bin/python Game4Loc/new_real_demo/verify.py
```

依赖：现有 `gtauav` 中的 NumPy、OpenCV；FFmpeg 优先从 PATH / `--ffmpeg`
读取。本次单独安装 `imageio-ffmpeg 0.6.0` 到输出目录 `vendor/`，没有修改
Conda 环境。新机器可自行安装 FFmpeg 或 imageio-ffmpeg。

## 同步依据与适配

1. `telemetry.py` 读取 MP4 的 sample tables、presentation timestamps 和 `djmd`
   原始数据轨道，不依赖文件名 / 创建时间去外推 GPS。
2. 本片为 `DJIMavic3Classic / dvtm_wm260l.proto`。字段路径和单位根据
   [ExifTool 的 DJI 字段定义](https://exiftool.org/TagNames/DJI.html) 核对：
   GPS 弧度转角度，姿态十分之一度，高度毫米转米。
   `wm260l` 字段布局由本片数据核验，不宣称其被 ExifTool 官方直接支持。
3. 原视频 10949 个显示帧与 10949 条遥测 sample 对齐，最大时间差 0.85 ms。
   内嵌帧号会跳号（最终到 10976），不能当作文件内的显示帧编号。
4. 保留每个包的原始 GPS；连续显示在 GPS 更新点间线性插值。
5. 卫星图使用 WGS84 → Web Mercator XYZ 坐标，按 GPS、云台 yaw/pitch/roll
   和相对高度生成局部地面视野。左右均保留 16:9。
6. 显示使用显式的水平视角估计 `72°`，可通过 `--hfov` 修改。DJI 公布的是
   [84° FOV 和等效 24 mm 镜头](https://www.dji.com/mavic-3-classic/specs)；
   本实现的 72° 水平视角是展示用近似值，不是该录像的相机标定结果。

## 数据质量与边界

- 这是 GPS 辅助对应演示，不是无 GPS 视觉定位成果，也不是逐像素配准真值。
- 相对起飞高度近似作为离地高度，地面近似为平面；没有地形模型或相机标定。
- 视频创建时间与外部飞行 CSV 直接拼接会产生位置偏差，因此外部 CSV 不参与
  这个版本的最终逐帧映射。创建时间只写入来源记录。
- 云台在这段录像中的方向近似恒定是原始遥测值；未人为制造旋转轨迹。
- 当前卫星影像来自 Google Earth Pro 导出的 Airbus 2024-09-12 影像；
  早期版本使用 Esri World Imagery z18。建筑、植被随时间变化，GPS 也可能有偏差。
- 已检查的本地 z19 缓存 630 张都是同一张“Map data not yet available”占位图，
  SHA-256 `9eafd300d61393184a4abc1d458564cfd1cd9b6f9c4e9c74687045c0a0e5b858`，
  不用于演示。加载器拒绝这一已知占位图。
- 首版的位置请求曾被自动审批拒绝。用户随后明确要求下载高清瓦片，已授权
  向 Esri/ArcGIS 请求这段航线的影像。最新下载记录见输出目录的 `imagery.json`。

后续讨论算法时，以这一独立目录为起点，先明确输入限制和验证方式，再引入模型。

## 历史 Esri 卫星影像下载

以下命令会将影像选择切回 Esri；保留 Google 版本时无需执行。

```bash
/home/lcy/miniconda3/envs/gtauav/bin/python Game4Loc/new_real_demo/download_imagery.py --zoom 18
/home/lcy/miniconda3/envs/gtauav/bin/python Game4Loc/new_real_demo/build.py
/home/lcy/miniconda3/envs/gtauav/bin/python Game4Loc/new_real_demo/verify.py
```

下载器按照已准备的逐帧视野收集所需瓦片，只在完整验证通过后写入
`work_dir/new_real_demo/imagery.json`；生成器默认读取这一选择。旧缓存保留。
下载日期不等于卫星影像拍摄日期。

实测当前 Esri 服务在片段首、中、末的 z19/z20 均返回 404。另检查了
2018、2020、2022、2024、2025、2026 六个 Wayback 版本的首帧位置，
z19/z20 均不可用。Clarity 对照也未获取到影像。
本次采用可用的最新 z18，地图网格在当地约 0.46 m/px；这一数值是网格采样
间距，不是已验证的传感器原生空间分辨率。新图与旧缓存内容不同，预览中的
树冠和建筑细节更清晰；不将放大、锐化或占位图称作高清瓦片。

原始请求结果和对比图保留在 `work_dir/new_real_demo/hd_probe/`。

## Google Earth 导出接入

2026-09-27 已完成真实 Google Earth Pro 导出及导入校验，当前影像选择已切换为
Google Earth / Image © 2026 Airbus，桌面视图拍摄日期为 2024-09-12。
两张原始导出均为 8192 × 6133；背景差异第 60 百分位为 0，五个独立校准点的
最大误差为 0.0159 米。校准误差仅描述底图坐标映射，不代表无人机定位精度。
共生成 78 张所需 PNG 瓦片，覆盖最后 30 秒全部视野。
Google 版已通过 1798 帧完整解码、1798 条帧映射及 7 组画面抽查。
播放器已验证播放、首末帧跳转及单帧回退。使用支持 HTTP Range 的服务启动，
以便浏览器准确跳转 MP4（普通 `python -m http.server` 不支持这一功能）：

```bash
python Game4Loc/new_real_demo/serve.py
```

服务仅监听本机 `127.0.0.1:8765`。

Earth Pro 使用官方 Google 签名安装包中提取的程序，在 Windows 用户目录
`C:\Users\14177\AppData\Local\GoogleEarthProDemo\client\googleearth.exe`
运行；这不是已完成系统级 MSI 安装的声明。原始导出及配准报告在
`Game4Loc/work_dir/new_real_demo/google_earth_export/`。

```bash
/home/lcy/miniconda3/envs/gtauav/bin/python Game4Loc/new_real_demo/earth_export.py prepare
```

生成 `Game4Loc/work_dir/new_real_demo/google_earth_export/export_controls.kmz`，
包含最后 30 秒航线和九个已知 WGS84 坐标的校准标记。片段地面视野联合范围
约 372 × 298 米；标记额外留出 35 米边距。导出步骤：

1. 用 Google Earth Pro 打开 KMZ，选择正俯视、北向上，关闭地形与 3D 建筑。
2. 保留 Google 和影像提供方署名，选择高分辨率导出 `controls.jpg`。
3. 只隐藏 `Calibration controls` 文件夹；不改变相机、窗口或导出分辨率，
   再导出 `clean.jpg`。保持航线文件夹隐藏。
4. 记录实际导出视图上的拍摄日期和署名。网页首帧位置曾显示 Airbus /
   2024-09-12，但不能直接假设桌面导出必然选择同一影像日期。
5. 导入真实导出图，再生成视频：

```bash
/home/lcy/miniconda3/envs/gtauav/bin/python Game4Loc/new_real_demo/earth_export.py import \
  --image Game4Loc/work_dir/new_real_demo/google_earth_export/clean.jpg \
  --controls Game4Loc/work_dir/new_real_demo/google_earth_export/controls.jpg \
  --date '<实际影像日期>' --attribution 'Google Earth / <实际影像提供方>'
/home/lcy/miniconda3/envs/gtauav/bin/python Game4Loc/new_real_demo/build.py
/home/lcy/miniconda3/envs/gtauav/bin/python Game4Loc/new_real_demo/verify.py
```

四角控制点建立图像到地图坐标的平面变换，其余五点独立检查，最大误差须
小于 0.75 米。导入器同时检查两张图背景没有移动、干净图确实移除了标记、
所需卫星范围完全位于原图内。只在通过后更新 `imagery.json`。
原始导出与配准报告保留；用于渲染的 PNG 瓦片采用 z20 重采样网格，**这不表示
原影像达到 z20 的原生空间分辨率**。Google 版本的单独卫星视频/PNG也带署名。

这套控制点标定只解决底图的地理坐标映射。无人机画面仍通过 GPS、云台姿态、
高度和估计视角确定范围；不运行视觉检索，不运行 UAV/卫星特征匹配。

## 2026-10-06 成果归档

本分支提交独立 Demo 源码、本文档，以及
`Game4Loc/new_real_demo/results/google_earth_20260927/` 下的生成参数、
底图导入报告、完整验证报告和产物 SHA-256 清单。归档当天重新运行视频验证。
报告中的绝对路径记录原运行环境，迁移机器时应以本仓库相对路径为准。

原始录像、Google Earth 原始导出、影像缓存、生成视频及运行依赖继续保留在
本地数据/输出目录，不纳入 Git 提交。新机器需要这些输入才能重新生成相同结果；
GitHub 归档不是可直接播放的视频附件。现有机器按上面的 build/verify/serve
命令即可复现；Google 导出与重采样步骤见上一节。
