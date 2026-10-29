[English](README.md) · [한국어](README.ko.md) · [Español](README.es.md) · **中文** · [日本語](README.ja.md)

# DataCash

**只用大家手里已有的手机和手表，收集机器人学习所需的人手数据。**

普通人在家用手机和智能手表录制简短的机器人基础动作任务（抽纸巾、移杯子、拧盖子等共 12 种），并获得报酬。
人形机器人和具身智能公司购买的结果是：去除个人信息的视频、声音、手腕动作、握力和心率，统一在同一时间轴上，
以可直接训练的 **LeRobot v3.0** 数据集交付。

Meta Global AI Developer Hackathon（Indie Developer · Push the Frontier）参赛作品。

![DataCash](docs/1026_datacash_brand.png)

## 一览

| | |
|---|---|
| 录制者 | 安卓 App + 手表 App。选择任务 → 人脸确认 → 录制 → 回答 1–2 个问题 → “叮”，报酬到账 |
| 购买企业 | 在网页上**用自然语言提需求**（“仓库机器人用的搬箱子数据，20–50 岁，预算 3,000 美元”）→ AI 设计任务并向全世界开放 |
| 隐私 | 参与者的脸被换成**不存在的人**；未同意的路人、屏幕和文件被遮盖；处理完成后删除原始文件 |
| 触觉 | 不用手套 —— **手表自己振动，测量手吸收了多少振动**来估计握力（GripSense，UIST 2012） |

## Meta 模型用在哪里

| 模型 | 用途 | 代码 |
|---|---|---|
| **SAM 3.1** | 遮盖未同意的人、儿童、屏幕、镜子、相框和文件（视频跟踪）；分割被操作的物体 → `observation.objects` | `datacash/meta/sam.py`、`datacash/privacy/mask.py` |
| **Muse Spark 1.3** | 遮盖后的视频 → 步骤、指令、使用的手、物体、失误（结构化 JSON）；企业的自然语言需求 → 任务设计 | `datacash/understand/`、`datacash/requests.py` |
| **Muse Image** | 34 位不存在的人（7 个地区 × 男女 × 20–60 多岁），用于换脸 | `scripts/make_donors.py`、`datacash/donors/` |
| **Muse Voice Transcribe** | 找出录音中说出的个人信息并静音 | `datacash/privacy/speech.py` |
| **Muse Glimmer**（开放权重） | 30B 模型用 llama.cpp 在这台 Mac 本地运行 —— 逐帧复核 Spark 的步骤（无 API 费用） | `datacash/understand/glimmer.py`, `datacash/evolve.py` ([docs/model_evolution.md](docs/model_evolution.md)) |
| **Muse Spark Open**（开放权重，816B） | 申诉时的“更高级模型”，也为每个标签提供第二意见（Fireworks 托管）；两个模型一致的步骤成为 Glimmer 的训练标签 | `datacash/understand/spark_open.py`、`scripts/crosscheck.py` |

每次 API 调用的费用都记录在 `data/api_ledger.jsonl`；超出预算时只跳过该步骤。

## 核心技术

### 1. 保留数据价值的隐私处理
- **换脸而不是模糊：** 用 Muse Image 生成的人替换参与者的脸（LivePortrait + MediaPipe）。在同性别、年龄相近的人中选最像的一位，先试换 20 帧；如果人脸识别（SFace）会把一半以上的帧判为“同一个人”，就换下一位。
- **连续性：** 换出的帧过于相像时，不打马赛克，而是减弱表情重新生成 —— 视频不会中断。
- **遮盖：** SAM 3.1 只遮盖未经同意的内容：其他人、儿童、屏幕、镜子、相框、文件。已同意的家人保持可见，每人使用各自的虚拟脸。
- **删除原始文件：** 正式服务中，处理一结束就删除原始视频和音频（`purge_originals`）。黑客松演示服务器只关闭了这一调用，以便展示原始与替换后的对比。

### 2. 多传感器自动同步
无需拍手：手机↔手表时钟往返、18–20 kHz 声学信标、手表振动脉冲。在真实手机 + 手表录制中残差为 0.5–2.3 毫秒（`datacash/sync/`）。

### 3. 不用手套的触觉 —— 戴在手腕上的 GripSense
我们把 Goel、Wobbrock、Patel 的 *GripSense*（UIST 2012）移到了手腕上。手表短暂振动；握得越紧，手吸收的振动越多 —— 陀螺仪高频能量下降，手指 4–12 Hz 的抖动上升。每个人用自评的力度（松开 · 轻 · 平常 · 紧，无需秤）训练一棵决策树，输出握力等级 0–3（`datacash/touch/`）。 在 Galaxy Watch4 上实测（IMU 最高 100 Hz，校准 1 次，用第 1 轮训练测第 2 轮，再反过来）：25 棵树的随机森林加 2 秒平滑，4 级 59%（随机 25%，紧握 100%）；视频看到手接触物体期间，轻 · 平常 · 紧 3 级为 79%。
心率、心跳间隔和皮肤温度来自 Samsung Health Sensor SDK（`docs/samsung_health_sdk.md`）。

### 4. 自然语言需求 → 任务
企业输入需求后，Muse Spark（结构化输出）设计任务（动作、物体、需要变化的条件）、参与条件（年龄、性别）和传感器组合，服务器按预算给出报价。确认后，任务只对符合条件的参与者开放 —— 需求本身*就是*采集分布。

### 5. 有利润空间的处理成本
- 无法使用的录制（太短、看不到手、太暗）在**调用任何 API 之前**就在本地退回 —— 0 美元
- 先用本地物体检测器扫描，只把需要的交给 SAM（单人录制不查询 “person”，屏幕仅在出现时查询，静止物体每秒 1 帧）
- 4 段真实录制：SAM 费用 **2.92 → 0.27 美元（−91%）**，每小时视频约 5–7 美元（售价约 43 美元/小时）
- 用 Numba 重写 SAM 掩码解码：每段录制 593 秒 → 8 秒，与官方解析器逐位一致

## 数据集（LeRobot v3.0）

`observation.images.ego`（遮盖 + 换脸后的视频）· `observation.hands` · `observation.body` · `observation.imu.phone` · `observation.imu.wrist` ·
`observation.audio` · **`observation.touch`**（接触、握力等级 0–3）· `observation.touch.vibration` · **`observation.physio`**（心率、皮肤温度；仅限生物信息同意者）·
`observation.objects` · `annotation.step` · `annotation.primitive`（reach · grasp · … · release）· `sync.error_ms` + 质量证书。
可用真实的 `lerobot` 0.6.1 读取（`scripts/validate_lerobot.py`）。

企业按每小时数据选择传感器组合 —— 视频（必选）₩35,000 · 姿态 ₩20,000 · 声音 ₩5,000 · 手腕 IMU ₩15,000 · 触觉 ₩25,000 · 生物信息 ₩50,000（`datacash/pricing.py`）。

## 运行

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]" torch pyyaml rich
echo "MODEL_API_KEY=..." > .env                       # 不提交到仓库
# data/models：MediaPipe hand/pose/face landmarker .task、efficientdet_lite0.tflite、OpenCV yunet.onnx + sface.onnx
.venv/bin/python -m pytest                            # 用已知真值的合成会话验证
DATACASH_TOKEN=... .venv/bin/python -m datacash.server.app --port 8800   # 服务器 + 企业网页
cd app && ./gradlew assembleDebug                     # 安卓 App + 手表 App
```

仓库中不包含 LivePortrait —— 请下载 [KwaiVGI/LivePortrait](https://github.com/KwaiVGI/LivePortrait)（MIT）并将 `LP_ROOT` 指向它。

## 目录

| 目录 | 作用 |
|---|---|
| `app/` | 安卓 App（同意、任务、人脸确认、录制、标记失误、问题、上传、报酬、申诉）+ 手表 App |
| `web/` | 企业网页（自然语言需求、多传感器时间轴、报价、订单、下载）和运营后台 |
| `datacash/privacy` | SAM 遮盖 · 换脸 · 语音隐私 · 预扫描 |
| `datacash/sync` · `touch` · `labels` | 传感器同步 · 触觉 · 手和身体姿态 |
| `datacash/understand` | Muse Spark 理解 · Glimmer 复核 · 传感器验证 |
| `datacash/export` · `jobs.py` | LeRobot 导出 · 证书 · 处理流程 |
| `datacash/server` | 采集服务器 + 企业 API |

## 文档（韩语）

- 黑客松计划：[docs/1026_meta_hackathon.md](docs/1026_meta_hackathon.md) · 第一天记录：[docs/DAY1.md](docs/DAY1.md)
- 遮盖与换脸评估：[docs/privacy_eval.md](docs/privacy_eval.md) · Meta 模型检查：[docs/model_check.md](docs/model_check.md)
- 手表 SDK：[docs/samsung_health_sdk.md](docs/samsung_health_sdk.md) · 演示视频脚本：[docs/demo_video_script.md](docs/demo_video_script.md)

## 黑客松之前 / 之中

| | 内容 |
|---|---|
| 之前（10 月 26 日前） | 只有计划文档和品牌图片，没有代码 |
| 之中（韩国时间 10 月 27 日 01:00 起） | 本仓库中的全部代码、App、网页、Muse Image 生成的人物和文档 |

## 许可与来源

MediaPipe（Apache 2.0）· OpenCV YuNet 与 SFace（Apache 2.0）· LivePortrait（MIT）· LeRobot（Apache 2.0）· Numba（BSD）。
测试视频来自 Wikimedia Commons（仅 CC BY、CC BY-SA 或公有领域；来源见 `docs/privacy_eval.md`），不提交到仓库。
按照其条款，Samsung Health Sensor SDK 不提交到仓库。
