## NewRealDemo：新电脑最快复现（2026-10-06）

当前分支 `NewRealDemo` 继承自 `RealDataDemo`，独立实现真实 DJI 视频最后 30 秒的
**左侧卫星 / 右侧航拍 GPS 辅助演示**。当前使用 Google Earth Pro 导出的 Airbus
影像（拍摄日期 `2024-09-12`）；不调用旧检索、VOP 或视觉匹配算法，不需要模型权重、
PyTorch 或 GPU。原项目的训练与评估说明保留在本文后半部分。

详细方法和历史结果见 [NewRealDemo.md](NewRealDemo.md)，固定结果与逐帧索引见
[results/google_earth_20260927](Game4Loc/new_real_demo/results/google_earth_20260927)。
以下命令均在仓库根目录的 **Ubuntu / WSL Bash** 中运行，Windows 命令另行标明。

### 1. 先准备数据：GitHub 不包含影像与视频

旧电脑仓库位置为 `/home/lcy/Workplace/GTA-UAV`；Windows 对应
`\\wsl.localhost\Ubuntu\home\lcy\Workplace\GTA-UAV`。新电脑可以放在任意目录。
**要复现完全相同的底图，最快方式是从旧电脑复制以下文件，而不是重新在线下载。**

| 文件 / 目录（相对仓库根目录） | 用途与获取方式 |
| --- | --- |
| `Game4Loc/data/Aerial-to-ground2/dji_fly_20260420_185426_25_1776683433584_video.mp4` | 原始 DJI 录像，从旧电脑或原始存储介质复制；没有公开下载链接。保留原始 MP4，转码可能丢失 `djmd` 遥测轨道。 |
| `Game4Loc/work_dir/new_real_demo/google_earth_export/clean.jpg` | Google Earth 干净原始导出，8192×6133。 |
| `Game4Loc/work_dir/new_real_demo/google_earth_export/controls.jpg` | 同一视图带九个校准点的导出，必须与 clean.jpg 成对保存。 |
| `Game4Loc/work_dir/new_real_demo/google_earth_export/export_plan.json`、`export_controls.kmz` | 已知坐标和导出视图；可按第 4 节重新生成。 |
| `Game4Loc/work_dir/new_real_demo/google_earth_export/rectified_z20/` | 当前 Google 底图，78 张 `{x}_{y}.png` 和 `download_manifest.json`；可从两张原始导出重新标定生成。 |
| `Game4Loc/work_dir/new_real_demo/` 整个目录 | 最快迁移方案：包含已生成 MP4、播放器、逐帧数据、底图、来源报告；仅播放无需原始录像和模型。`vendor/` 可不复制，按第 2 节重装。 |
| `Game4Loc/work_dir/new_real_demo/satellite_cache/world_imagery_20260918/z18/` | 历史 Esri 缓存，11 张 JPG；当前 Google 版不依赖它。 |

这些大文件没有上传 GitHub，也没有隐藏的云盘下载地址。只有 Git clone、不提供原始录像或
底图，不能复现相同视频。重新下载的影像可能更新，导出像素数也不等于原始传感器分辨率。

### 2. 最小运行环境与工具

```bash
git clone --branch NewRealDemo https://github.com/BUAA-Lcy/STL-UAV.git
cd STL-UAV
conda create -n newrealdemo python=3.10 -y
conda activate newrealdemo
python -m pip install numpy==2.2.5 opencv-python-headless==4.13.0.92 requests==2.32.5 imageio-ffmpeg==0.6.0
python -c "import cv2,numpy,requests,imageio_ffmpeg; print(cv2.__version__,numpy.__version__); print(imageio_ffmpeg.get_ffmpeg_exe())"
```

若已安装 Python 3.10，可用 `python3 -m venv .venv`、`source .venv/bin/activate`
代替 Conda。这里使用 headless OpenCV，Demo 不需要 OpenCV 桌面窗口；同一新环境不要
同时安装多个 OpenCV 发行包，以免相互覆盖 `cv2`。
原机使用 `gtauav` 环境，但本 Demo 无需安装原项目整套训练依赖。

| 工具 | 用途 / 原机情况 |
| --- | --- |
| Python 3.10、NumPy、OpenCV | MP4 遥测解析、坐标投影、卫星裁剪、视频核验；无 GPU 要求。 |
| Requests | 仅 Esri 下载器需要；访问该服务会请求航线所在区域的瓦片。 |
| FFmpeg / imageio-ffmpeg 0.6.0 | 视频解码和编码。脚本优先 PATH 中的 FFmpeg，其次寻找 imageio-ffmpeg；可用 `build.py --ffmpeg /绝对路径/ffmpeg` 显式指定。原机编码器为 FFmpeg 7.0.2，不同版本的 MP4 字节哈希可能不同。 |
| Google Earth Pro 7.3.7 Windows x64 | 仅重新导出 Google 底图时需要，使用桌面版“保存图像”，不使用 Google 瓦片接口或 API Key。 |
| Git、浏览器 | 获取源码；通过本机 HTTP 服务播放、变速、全屏和逐帧查看。 |
| ExifTool 字段文档 | 仅用于开发时核对 DJI 字段；运行不需要 ExifTool、protoc 或 DJI SDK。 |

Google Earth Pro [官方安装器列表](https://support.google.com/earth/answer/168344?hl=en)、
[7.3.7 Windows x64 安装包](https://dl.google.com/dl/earth/client/advanced/current/googleearthprowin-7.3.7-x64.exe)、
[官方保存图像说明](https://support.google.com/earth/answer/148146?hl=en)。
新电脑优先正常安装。原机安装器曾卡在 UAC，后来从官方签名安装包提取到
`C:\Users\14177\AppData\Local\GoogleEarthProDemo\client\googleearth.exe` 运行；
这是原机应急路径，不是新电脑必须复用的安装位置。提取时临时用到的 olefile、cabextract、
libmspack 不属于 Demo 运行依赖，不需要重走安装包提取流程。Codex / 浏览器自动化工具
只用于当时操作与检查，也不是复现依赖。

### 3. 最快路径：迁移已有底图并重新生成

按第 1 节放好原始录像和 `rectified_z20/`，然后运行：

```bash
python Game4Loc/new_real_demo/build.py \
  --cache Game4Loc/work_dir/new_real_demo/google_earth_export/rectified_z20 \
  --zoom 20 --last-seconds 30 --hfov 72
python Game4Loc/new_real_demo/verify.py
python Game4Loc/new_real_demo/serve.py
```

打开 <http://127.0.0.1:8765/>。复制已有完整输出目录后，只运行最后一条即可播放。
默认原片路径见第 1 节；`build.py --video` 可以修改输入，但目前 `verify.py` 固定校验
默认这段原片和默认输出目录。其他录像需另建数据协议，不能复用本片的帧索引。

**迁移路径注意：** `imagery.json` 内保存了旧电脑缓存的绝对路径。因此新电脑首次构建
必须像上面一样显式传 `--cache` 和 `--zoom`，或按第 4 节重新 import，让脚本写入新路径。
只复制旧 `imagery.json` 后裸跑 build，可能仍访问 `/home/lcy/...`。使用 Range 服务
`serve.py`，不要用普通 `python -m http.server` 代替，否则视频跳转可能回到首帧。
WSL 端口若无法从 Windows 浏览器访问，可在 Windows 安装 Python，运行
`python <仓库路径>\Game4Loc\new_real_demo\serve.py`，同样只监听 127.0.0.1。

### 4. 重新获取 Google Earth 卫星图：导出、标定、切片

**下载方式是桌面导出，不存在本项目使用的 Google XYZ 瓦片下载 URL。**
先恢复本片的固定逐帧索引，解决首次下载时“需要底图才能生成索引”的依赖：

```bash
mkdir -p Game4Loc/work_dir/new_real_demo
cp Game4Loc/new_real_demo/results/google_earth_20260927/frames.csv \
   Game4Loc/work_dir/new_real_demo/frames.csv
cp Game4Loc/new_real_demo/results/google_earth_20260927/manifest.json \
   Game4Loc/work_dir/new_real_demo/manifest.json
python Game4Loc/new_real_demo/earth_export.py prepare
```

这一步不需要已有卫星缓存，也不需要加载原片；只为已归档的这 1798 帧生成导出范围。
会生成 `Game4Loc/work_dir/new_real_demo/google_earth_export/export_controls.kmz`
及 `export_plan.json`。若使用不同录像，不能复制这一固定索引冒充新航线。

1. 在 Google Earth Pro 打开 KMZ。Windows 若不能直接打开 WSL UNC 路径，先复制到
   Windows 本地目录，如 `C:\Temp\earth_demo_export\`。KMZ 已设置视图中心和九个标记；
   联合地面范围约 372×298 米，额外留出 35 米标记边距。
2. 使用正俯视、北向上，关闭地形和 3D 建筑；保持 Flight path 文件夹隐藏。
   等待影像清晰加载，记录界面实际拍摄日期与影像提供方。需要历史日期时在 Earth
   中选择该日期；若无法取得相同影像，应记录新日期，不能沿用旧记录。
3. 选择“文件 → 保存 → 保存图像”（Windows `Ctrl+Alt+S`），地图选项关闭标题和图例，
   防止图例生成第十个校准标记；保留 Google / 提供方署名。选择最高导出分辨率，
   原机为 8192×6133（尺寸随视口比例而变），保存为 `controls.jpg`。
4. 仅隐藏 Calibration controls 文件夹（或整个 KMZ，前提是航线一直隐藏），
   不移动相机、不改变窗口大小或分辨率，保存 `clean.jpg`。不要用屏幕截图代替高清导出。
5. 把两张 JPG 复制到仓库 `Game4Loc/work_dir/new_real_demo/google_earth_export/`，执行：

```bash
# 下列日期 / 署名是本次归档值；重新导出时填写界面实际值。
python Game4Loc/new_real_demo/earth_export.py import \
  --image Game4Loc/work_dir/new_real_demo/google_earth_export/clean.jpg \
  --controls Game4Loc/work_dir/new_real_demo/google_earth_export/controls.jpg \
  --date 2024-09-12 --attribution 'Google Earth / Image (c) 2026 Airbus'
python Game4Loc/new_real_demo/build.py
python Game4Loc/new_real_demo/verify.py
python Game4Loc/new_real_demo/serve.py
```

导入器检测九个标记，用四角建立变换，其余五点独立校验（最大误差必须小于 0.75 米），
检查两张图背景一致、标记已移除和全片视野覆盖，然后输出 `rectified_z20/*.png`、
`import_report.json`、`download_manifest.json`，最后更新 `imagery.json`。
z20 是约 0.115 m/px 的重采样网格，**不是卫星原始空间分辨率声明**。
当前校准最大误差约 0.0159 米，仅代表人工控制点的底图映射一致性，不是 UAV 定位精度。

### 5. 可选：重新下载 Esri 瓦片

服务：[Esri World Imagery MapServer](https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer)。
代码使用的瓦片地址为：

```text
https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}?blankTile=false
```

URL 顺序是 **z/y/x**，本地文件名是 **x_y.jpg**。先按第 4 节恢复 `frames.csv` 和
`manifest.json`（已生成过索引则无需覆盖），再运行：

```bash
python Game4Loc/new_real_demo/download_imagery.py --zoom 18 --workers 4
python Game4Loc/new_real_demo/build.py
python Game4Loc/new_real_demo/verify.py
```

下载位置：`Game4Loc/work_dir/new_real_demo/satellite_cache/world_imagery_<UTC日期>/z18/`。
完整下载和校验成功后会自动更新 `imagery.json`，**这会把当前底图选择从 Google 切换为 Esri**。
失败会保留原选择。2026-09-18 实测本片 z18 可用、抽测 z19/z20 不可用；不要把
“Map data not yet available”占位图或上采样当作高清影像。下载日期不等于拍摄日期。
重回 Google 版可执行第 3 节的显式缓存构建命令，或重新导入 Google 导出图。

### 6. 当前算法、结果与复现检查

流程：MP4 `djmd` 逐帧遥测 → GPS 更新点间线性插值（同时保留原始 GPS）→
WGS84 到 Web Mercator → 云台 yaw/pitch/roll、相对高度与 72° 水平视角估计
地面四角 → 对卫星底图做透视裁剪 → 与原片同帧并排输出。
假设地面平坦、相对起飞高度近似离地高度；没有相机实测标定、视觉特征匹配或无 GPS 定位。
底图校准与 UAV/卫星图像匹配是不同步骤。

| 检查项 | 归档结果 |
| --- | --- |
| 源帧范围（从 0 开始） | 9151–10948，共 1798 帧 |
| 帧率 / 片段时长 | 60000/1001 fps / 29.996633 s |
| 遥测最大时间差 | 0.85 ms |
| 缺失底图帧 | 0 |
| 双画面视频 | `new_real_demo.mp4`，1920×800 |
| 卫星视频 | `satellite_frames.mp4`，928×562，含 40 px 署名栏 |
| 自动验证 | 两个视频完整解码、1798 条映射、7 组原片与卫星裁剪抽查通过 |

验证输出为 `Game4Loc/work_dir/new_real_demo/verification.json`，来源与参数在同目录
`manifest.json`。对完全复制的产物，可与归档 `artifacts.json` 中 SHA-256 比对；
编码器或重新下载底图变化会影响文件哈希。显示区域相同不等于逐像素配准准确，
季节、拍摄日期、GPS、高度和视角估计都可能造成偏差。后续引入学习算法另行记录输入限制与评估协议。

---

<!-- PROJECT LOGO -->
<p align="center">

  <h3 align="center">Game4Loc: A UAV Geo-Localization Benchmark from Game Data</h3>

</p>

<h5 align="center">
  If you like our project, please give us a star ⭐️ for the continuous updates.
</h5>

<p align="center">
  By <a href="https://yux1angji.github.io/" target='_blank'>Yuxiang Ji*</a>,&nbsp;
  Boyong He*,&nbsp; Zhuoyue Tan,&nbsp; Liaoni Wu
</p>

<p align="center">
  <img src="resources/trajectory_demo_compress.gif" alt="Demo">
  <br>
  <i>
  Localization in flight trajectory after pre-trained on GTA-UAV dataset.
  </i>
</p>

<p align="center">
  <a href="https://yux1angji.github.io/game4loc/"><strong>[Project Page]</strong></a> |
  <a href="https://arxiv.org/abs/2409.16925"><strong>[Arxiv]</strong></a>
</p>

- [x] Part I: LR Dataset
- [x] Part II: Train and Test
- [x] Part III: Pre-trained Checkpoints
- [x] Part IV: HR Dataset and Supplementary

## <a id="news"></a> 🔥 News

- [Oct 20, 2025]: Bug fix for data angle calculation ⚠️
- [Dec 10, 2024]: Game4Loc is accepted by AAAI'25 🎉
- [Sep 28, 2024]: Official GTA-UAV dataset release 🚧

## <a id="table-of-contents"></a> 📚 Table of contents

- [Dataset Highlights](#dataset-highlights)
- [Dataset Access](#dataset-access)
- [Dataset Structure](#dataset-structure)
- [Train and Test](#train-and-test)
- [More Features](#more-features)
- [Pre-trained Checkpoints](#pre-trained-checkpoints)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)


## <a id="dataset-highlights"></a> 🌟 Dataset Highlights
![](resources/GTA-UAV-data-construction.jpg)
*GTA-UAV data construction*

<b><i>GTA-UAV</i> dataset</b> provides a large continuous area dataset (covering 81.3km<sup>2</sup>) for UAV visual geo-localization, expanding the previously aligned drone-satellite pairs to **arbitrary drone-satellite pairs** to better align with real-world application scenarios. Our dataset contains:

- 33,763 simulated drone-view images, from multiple altitudes (80-650m), multiple attitudes, multiple scenes (urban, mountain, coast, forest, etc.).

- 14,640 tiled satellite-view images from 4 zoom levels for arbitrarily pairing.

- Overlap (in IoU) of FoV for each drone-satellite pair.

- Drone (camera) 6-DoF labels for each drone image.

## <a id="dataset-access"></a> 💾 Dataset Access
The dataset is released in two versions: low resolution (512x384, 12.8G) and high resolution (1920x1440, 143.3G). Last updated in Oct 2025.

|                                      Low Resolution Version                                      |                                     High Resolution Version                                      |
|:------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------:|
| [HuggingFace🤗](https://huggingface.co/datasets/Yux1ang/GTA-UAV-LR) | [HuggingFace🤗](https://huggingface.co/datasets/Yux1ang/GTA-UAV-HR) |
| [BaiduDisk](https://pan.baidu.com/s/1_gfz8ni2O8haCHMcoGfedA?pwd=gtav) | Released soon |


The high resolution dataset will be released soon.

## <a id="dataset-structure"></a> 📁 Dataset Structure

### Directory Structure
```
├─ GTA-UAV
|   ├── drone/
|   |   └── images/
|   |       ├── 200_0001_0000000001.png
|   |       ├── 200_0001_0000000002.png
|   |       └── ...
|   ├── satellite/
|   |   ├── 6_0_0_0.png
|   |   ├── 6_0_0_1.png
|   |   └── ...
|   ├── cross-area-drone2sate-train.json
|   ├── cross-area-drone2sate-test.json
|   ├── same-area-drone2sate-train.json
└── └── same-area-drone2sate-test.json
```

### Example Entry in `x-area-drone2sate-x.json`

This entry provides a detailed description and paired satellite images for a single drone image in the training/test dataset:

```json
{
    "drone_img_dir": "drone/images",
    "drone_img_name": "500_0001_0000025682.png",
    "drone_loc_x_y": [4472.2036708036, 9460.91532053518],
    "sate_img_dir": "satellite",
    "pair_pos_sate_img_list": [
        "4_0_6_13.png"
    ],
    "pair_pos_sate_weight_list": [
        0.47341428718085427
    ],
    "pair_pos_sate_loc_x_y_list": [
        [4492.8, 9331.2]
    ],
    "pair_pos_semipos_sate_img_list": [
        "4_0_6_13.png",
        "5_0_12_27.png",
        "5_0_13_27.png"
    ],
    "pair_pos_semipos_sate_weight_list": [
        0.47341428718085427,
        0.27864086433392504,
        0.2149980058725643
    ],
    "pair_pos_semipos_sate_loc_x_y_list": [
        [4492.8, 9331.2],
        [4320.0, 9504.0],
        [4665.6, 9504.0]
    ],
    "drone_metadata": {
        "height": 494.70794677734375,
        "drone_roll": -0.2723846435546875,
        "drone_pitch": 1.981452226638794,
        "drone_yaw": 84.99999237060547,
        "cam_roll": -90.27238464355469,
        "cam_pitch": 1.981452226638794,
        "cam_yaw": 84.99999237060547
    }
},
```

### Metadata Details

- `drone_loc_x_y`: Provides the 2D location for the centre of drone-view image.

- `pair_pos_sate_img(weight/loc_x_y)_list`: Provides the positive paired satellite image / weight(IOU) / 2D location list.

- `pair_pos_semipos_sate_img(weight/loc_x_y)_list`: Provides the positive & semi-positive paired satellite image / weight(IOU) / 2D location list.

- `drone_metadata`: Provides the height (altitude above ground level), drone pose (roll, pitch, yaw), and camera pose (roll, pitch, yaw) information.


### Collect Your Own Data

You may want to collect your own data from simulated game environments, if so, you could refer [here](DeepGTAV/VPilot/datageneration_GeoLoc.py).

To configure the simulation and collection environment, please refer [DeepGTA](https://github.com/David0tt/DeepGTAV).
Notice that the compiled `DeepGTA` plugin for our GTA-UAV data simulation and collection is located at [here](DeepGTAV/DeepGTAV-PreSIL/bin/Release/).

### Pre-process for UAV-VisLoc

To pre-process the raw [UAV-VisLoc](https://github.com/IntelliSensing/UAV-VisLoc) data into a similar format as GTA-UAV, you can refer [this script](scripts/prepare_dataset/visloc.py).
What's more, you can also refer to it and modify (extend) it to fit your custom similar datasets.

## <a id="train-and-test"></a> 🚀 Train and Test

![](resources/pipeline.jpg)
*Proposed training and test pipeline*

First, install dependencies   
```bash
cd Game4Loc
# install requirements 
pip install -r requirements.txt
```

Then you could simply run the training experiments on GTA-UAV cross-area setting by
```bash
# run experiment (example: GTA-UAV cross-area setting)  
python train_gta.py \
    --data_root <The directory of the GTA-UAV dataset> \
    --train_pairs_meta_file "cross-area-drone2sate-train.json" \
    --test_pairs_meta_file "cross-area-drone2sate-test.json" \
    --model "vit_base_patch16_rope_reg1_gap_256.sbb_in1k" \
    --gpu_ids 0,1 --lr 0.0001 --batch_size 64 \
    --with_weight --k 5 --epoch 5
```

Or run the training experiments on UAV-VisLoc by
```bash
# run experiment (example: UAV-VisLoc same-area setting)  
python train_visloc.py \
    --data_root <The directory of the UAV-VisLoc dataset> \
    --train_pairs_meta_file "same-area-drone2sate-train.json" \
    --test_pairs_meta_file "same-area-drone2sate-test.json" \
    --model "vit_base_patch16_rope_reg1_gap_256.sbb_in1k" \
    --gpu_ids 0,1 --lr 0.0001 --batch_size 64 \
    --with_weight --k 5 --epoch 20
```

Or directly run evaluating script from [pre-trained checkpoints](#pre-trained-checkpoints) by
```bash
# run experiment (example: GTA-UAV cross-area setting)  
python eval_gta.py \
    --data_root <The directory of the GTA-UAV dataset> \
    --test_pairs_meta_file "cross-area-drone2sate-test.json" \
    --model "vit_base_patch16_rope_reg1_gap_256.sbb_in1k" \
    --checkpoint_start <The path of the pre-trained checkpoints> \
    --gpu_ids 0
```

## <a id="more-features"></a> 💡 More Features

### Finer Localization with Post-process
Some studies divide localization into two parts: retrieval and matching. Our work focuses on the first part. 
Nevertheless, we also provide support for finer localization based on image matching (Thanks to the excellent zero-shot capabilities of [GIM](https://github.com/xuelunshen/gim/)). 
Set `with_match=True` in [eval script](Game4Loc/eval_gta.py) if needed.

## <a id="pre-trained-checkpoints"></a> 🤗 Pre-trained Checkpoints
The pre-trained models (two version: cross-area/same-area) are available at [Huggingface](https://huggingface.co/Yux1ang/gta_uav_pretrained_models) and [BaiduDisk](https://pan.baidu.com/s/1jle03w4iB9MXg6ATZubSjg?pwd=37jq).

## <a id="license"></a> 🎫 License
This project is licensed under the [Apache 2.0 license](LICENSE).

## <a id="acknowledgments"></a> 🙏 Acknowledgments 
This work draws inspiration from the following code as references. We extend our gratitude to these remarkable contributions:

- [Sample4Geo](https://github.com/Skyy93/Sample4Geo)
- [DeepGTA](https://github.com/David0tt/DeepGTAV)
- [GTA-V-Wolrd-Map](https://github.com/Flamm64/GTA-V-World-Map)
- [GIM](https://github.com/xuelunshen/gim/)

## <a id="citation"></a> 📌 Citation
If you find our repository useful for your research, please consider citing our paper:
```bibtex
@inproceedings{ji2025game4loc,
  title={Game4loc: A uav geo-localization benchmark from game data},
  author={Ji, Yuxiang and He, Boyong and Tan, Zhuoyue and Wu, Liaoni},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={4},
  pages={3913--3921},
  year={2025}
}
```
