# Multimodal YOLO Paper — Research and Implementation Plan

**状态：** 研究准备阶段；本文档是交给 Codex 执行的工程与实验计划，不代表数据集或方法已验证。

**研究目标：** 基于 Ultralytics 构建可复用、尽量不依赖特定 YOLO 版本的 RGB–Thermal–Depth (RGB-T-D) 多模态二维目标检测框架，研究融合位置、模态可靠性与 Mixture-of-Experts (MoE) 路由对检测精度、鲁棒性及效率的影响。以高质量 SCI 研究论文为目标，但不预先承诺精度提升或期刊分区。

**当前优先级：** Phase 0–2 已完成并通过人工关口；**最终入选数据集为 R-LiViT 与 MID-3K**，下一步进入 Phase 3（Ultralytics 通用多模态基线）。

**最终数据集决策（2026-11-10 人工确认）：** 选用 **R-LiViT、MID-3K**；**InfraParis、MM5、VDT-2048 不进入正式基准**，仅保留 Phase 0 的审计记录、适配器与转换产物以备追溯，不用于训练、消融或论文主表。

---

## A. 科学问题与边界

### A1. 拟研究的问题

- **RQ1（通用性）：** 一个与检测头解耦的多模态模块能否低成本适配多个 Ultralytics YOLO 架构，并稳定优于相同架构的单模态及朴素融合基线？
- **RQ2（融合策略）：** RGB-T、RGB-D、T-D 和 RGB-T-D 的收益如何随场景、目标尺度和融合层级变化？是否存在新增模态导致负迁移的情况？
- **RQ3（MoE）：** 可学习的融合专家与动态路由，相对 Concat/Add/普通门控，是否在公平容量与训练协议下提供稳定收益？
- **RQ4（可靠性）：** 在深度缺失、热图噪声、低照度及模态轻微错位等条件下，可靠性引导的路由是否降低性能退化？
- **RQ5（效率）：** 精度增益是否值得额外 Params、FLOPs、显存和实际推理延迟？

### A2. 研究边界

- 首要任务是**二维边界框目标检测**，不是语义分割、显著性检测、单目标跟踪或三维检测；这些任务的标注不能不加审核地当成常规目标检测 Ground Truth。
- 模态名称固定为 `rgb`、`thermal`、`depth`。不要将 MID-3K 的 `intensity` 与 `thermal` 混淆。
- Depth 可来自原生深度成像或 LiDAR 投影，但须记录来源、单位、有效区域、插值以及参考相机。不得将单目预测深度混称为真实传感器深度。
- 保留原始数据与官方标注不可变；衍生结果均可重建并追溯。
- 不能将不同数据集的 mAP 简单直接平均并声称统一性能；类别空间和评估协议应分别说明。

---

## B. Phase 0 — 仓库与原始数据审计（先于开发转换器）

### B1. Codex 需要先检查的内容

1. 阅读当前仓库结构、Ultralytics Submodule/安装版本、已有 `datasets/`、`models/`、`scripts/`、训练与评估代码；**尽量增加独立文件，避免直接改动上游 Ultralytics 源码**。
2. 对五个下载目录执行只读扫描，生成 `raw_inventory.json`：路径、文件数、文件类型、文件大小、图像分辨率、数据位深、可能的标注格式、序列/场景/时间信息、标定文件以及缺失情况。
3. 为每个数据集建立 `dataset_profile.yaml`：官方主页、论文、版本/发布日期、数据许可、类别定义、原始模态、深度来源、是否已经配准、标注坐标系、官方划分、质量风险和转化方案。未知字段标记 `unknown`，不要猜测。
4. 先做轻量样本检查（例如每种模态、每种标注格式各取数例），再编写实际解析器；未下载完成的目录标记 `pending_download`，不要因缺少文件整体失败。
5. 形成数据集对应关系和风险清单，提交人工查看。**此阶段不启动正式训练。**

### B2. 五个数据集分别要重点核查的事项（Phase 0 审计记录）

下表是 Phase 0 对五个候选的核查结论，保留作为审计依据。最终只选用 MID-3K 与 R-LiViT。

| 数据集 | 默认任务定位（待实物核实） | 转换重点 / 风险 |
|---|---|---|
| MID-3K | RGB/T/Depth，官方人类检测框 | 四模态文件一一配对；RGB/Depth 与 Thermal 可能使用不同检测框和坐标参考；采集日期分组；LiDAR 派生深度及轻微时间错位 |
| InfraParis | 三模态道路场景，官方检测主要为行人 | 官方可用的 RGB/T/Depth 与标注划分；热像视场及投影；不能直接把语义分割多类 Mask 视作实例检测标签 |
| R-LiViT | RGB/T + LiDAR，多类别二维检测 | 确认 RGB-T 2D 标注类别/参考视图及可用标定；同步匹配点云→深度；不能错误地把 3D LiDAR 类别当 2D 类别 |
| MM5 | 五模态对齐室内物品，多类实例标注 | 选 RGB/T/Depth；解析 `ANNO_INST` + `META` 或官方 COCO 导出；保留原始实例类别与来源；相同场景多光照不能跨训练/测试拆开 |
| VDT-2048 | RGB/T/Depth 显著目标检测 | 官方 Mask 主要是显著/背景标签，**34 种物品不等于34类现成检测 GT**；可产出候选框，未经实例和类别核验不得纳入多类别 mAP 主表 |

参考链接：
- MID-3K: https://github.com/kennedyk1/MID-3K
- InfraParis: https://github.com/ENSTA-U2IS-AI/infraParis
- R-LiViT: https://github.com/XITASO/r-livit
- MM5: https://github.com/martinbrennernz/MM5-Dataset
- VDT-2048: https://github.com/VDT-2048/VDT-Dataset

**Phase 0 交付：** `reports/data_inventory/`、五份 profile、原始目录扫描日志、缺失/待定项目列表。没有数据文件时 Codex 只建立适配器骨架、单元测试样本和待办清单，不能伪造统计数字。

---

## C. Phase 1 — 统一数据格式与转换工具

### C1. 设计原则

统一的是**接口与数据契约**，不是强制把所有模态转成 8-bit RGB，也不是强行把所有数据集映射到同一组类别。使用“共享转换流水线 + 数据集特定 Adapter”：

`raw files → dataset adapter → pairing/calibration → canonical samples → detection labels → validator → statistics/visualization`

转换任务应可重复、幂等、支持 `--dry-run` / `--strict` / `--overwrite`，不覆盖原始数据，记录版本及全部参数。能生成硬链接/软链接时可按环境选择，以减少冗余存储；不能依赖链接才能复现。

### C2. 推荐标准目录

```text
data/
  raw/                         # 原始下载数据（只读；不上传 Git）
    mid3k/ infraparis/ r_livit/ mm5/ vdt2048/
  processed/
    <dataset>/<conversion_version>/
      images/
        rgb/{train,val,test}/<sample_id>.png
        thermal/{train,val,test}/<sample_id>.png
        depth/{train,val,test}/<sample_id>.png
      labels/{train,val,test}/<sample_id>.txt
      calibration/             # 可选，内参/外参/投影配置
      metadata/
        samples.jsonl          # 核心跨模态配对/质量/来源清单
        classes.json          # 数据集原始类→本地类映射
        split_manifest.csv    # sample_id, group_id, split, source
        conversion.json       # 参数、版本、哈希、生成时间
        exclusions.csv        # 排除原因，便于人工审阅
      dataset.yaml            # YOLO 标准类别等元数据；Loader 可根据 manifest 扩展
      quality_report.json
      previews/
```

可改用 PNG/TIFF/NPY 等无损容器存储非 RGB 模态，但要求 `samples.jsonl` 显式写入：`dtype`、`shape`、`units`、`scale`、`invalid_value`、`encoding`、`reference_camera`、`alignment_status` 和 `source_path`。**不要默认所有 depth 都以米保存，也不要对热像逐帧 Min-Max 后覆盖原始辐射数据。**

每组样本共享 `sample_id`，同一份 YOLO 标签仅定义在一个明确的**参考图像坐标系**（优先 RGB；若数据集标注或对齐限制要求以 Thermal 为基准，也要显式记录）。对模态不完全匹配的样本保留错误状态并在配置中决定排除，不进行无提示替换。

`labels/.../<sample_id>.txt` 采用标准 YOLO Detect：`class_id x_center y_center width height`，均归一化到**参考坐标系尺寸**。必须能够与源标注互相追溯。不同数据集使用各自的 `classes.json`，暂不强行跨数据集合并类别。

### C3. 必须实现的适配器及脚本

```text
src/mdet_data/
  schema.py                 # CanonicalSample / Annotation / ModalityInfo
  registry.py               # Adapter 注册与配置
  adapters/
    base.py
    mid3k.py
    infraparis.py
    r_livit.py
    mm5.py
    vdt2048.py
  transform/
    projection.py           # 点云投影、Z-buffer、遮挡/投影合法性
    masks_to_boxes.py       # 实例 Mask→Boxes，类别关联
    normalization.py        # 实验时的数值映射，不破坏 raw
    boxes.py                # 像素框/COCO/YOLO 转换
  validate.py
  stats.py
  preview.py
scripts/data/
  inspect_raw.py
  convert_dataset.py
  validate_dataset.py
  summarize_datasets.py
  generate_previews.py
configs/datasets/<dataset>.yaml
reports/data_inventory/
reports/data_statistics/
```

优先使用 `argparse` 或仓库既有配置机制实现统一命令。每个命令须支持按数据集、split、样本数、输出路径和 seed 配置。不要编写五套互不兼容的脚本。

### C4. 标注转换规则

1. **原生二维检测框：** 优先保留官方类别、官方 split、官方框。完成像素坐标、归一化坐标及投影后的往返校验。
2. **实例分割：** 使用每个实例的独立 Mask 生成最小外接框，并从官方 metadata 获取实例类别；过滤条件（最小可见面积、忽略标签等）配置化，保留被排除对象的日志；目标是可复现的 *derived detection annotations*。
3. **语义分割 / 二值显著性：** 只能生成 *candidate boxes*，**不默认产生可信多类别 GT**。对象相接、遮挡、多个显著对象时需人工抽检/重新标注；缺少类别来源时标记 `annotation_status=candidate_only`。
4. **LiDAR→Depth：** 按真实内外参、坐标系及时间戳，投影至选定的参考相机；实施 Z-buffer，输出深度图、有效像素掩码、覆盖率及匹配置信信息（有据可查才生成）。缺标定/时间同步时 `alignment_status=blocked`，不能直接按分辨率 resize 代替标定。
5. **热像/深度数值：** 原始位深数据保留；训练时预处理作为独立配置和消融因素。显式处理 NaN/Inf、深度 0 值/65535、动态范围及跨数据集不同单位。
6. **几何一致性：** 先核实 RGB、Thermal、Depth 的有效 FOV、对齐与裁剪；所有 Resize/Letterbox/Flip 必须对参考标签与相应模态使用正确、可追溯的几何变换。

### C5. 自动质量检查（转换必须通过）

- 图像/标签可读性，图像长宽/色彩通道/位深，解码失败清单。
- 跨模态 `sample_id` 一一对应，帧/序列/时间戳匹配，缺失率与投影成功率。
- `class_id` 存在性、YOLO 框范围 `[0,1]`、正面积、越界框、重复框、异常小框；所有自动修改可追溯。
- 场景/序列/采集日期重复与跨 split 泄漏（按 group / 序列 / 时间邻近检测）。
- 对齐诊断可视化：RGB-T 叠加、深度伪彩叠加、图像及标签框、投影可见区域；自动选择正常及异常代表样本。
- 标签覆盖度：随机审查*无标签图片是否确实无目标*；只标显著对象的图像不得错误纳入完整检测评估。
- 每个 Adapter 有小规模 fixture 单元测试及逆向坐标验证；完整转换结束生成可机读报告与 warning/error 清单。

**Phase 1 验收：** 至少一个数据集能够完成端到端 `inspect → convert → validate → preview → summary`；其余四个 Adapter 按实际文件结构完成适配（缺文件则明确阻塞），不修改源数据；构建独立 Loader Smoke Test，随机抽查多模态张量与标签同图对应。

---

## D. Phase 2 — 数据统计与人工最终选择（强制人工关口）

**状态：已通过（2026-11-10）。** 统计与选择报告已生成在 `reports/data_statistics/`；研究者已显式批准最终组合 **R-LiViT + MID-3K**，可进入 Phase 3。近重复/感知哈希检测按人工决定不实现，泄漏风险以 group / 序列 / 时间邻近检测为准；box 几何与存储估算、图表类交付不阻塞本关口。

### D1. Codex 自动输出统计与可视化

对每个数据集分 Train/Val/Test 和整体输出：

1. 样本数、有效三模态配对数、缺模态/失败对齐数、场景数、序列数、场景独立性及时间覆盖。
2. 原始类别列表、每类实例数、各类出现图像数、长尾分布、忽略类别占比；标注来源：官方框、实例 Mask 衍生框、待审核候选框。
3. 图片分辨率、纵横比、位深与强度分布，深度范围、无效深度比例、LiDAR 投影覆盖率及热像动态范围。
4. Box 长宽、面积、长宽比、每图目标数、COCO Small/Medium/Large（在明确采用的评估尺度下）、目标遮挡/截断信息（若有官方属性）。
5. 训练、验证、测试的类别与场景分布，数据泄漏风险、近重复样本数。
6. 每个数据集至少生成一个随机可视化网格、一个最差质量网格、一个各类稀有样本网格，展示 RGB/T/Depth 和参考框。
7. 估算转换后存储量、训练输入尺寸、基础训练资源需求（只作估算，实测后再更新）。

交付：`dataset_summary.csv`、`class_distribution.csv`、`modality_quality.csv`、`split_leakage_report.csv`、`dataset_selection_report.md`，以及 PNG/HTML 图表。

### D2. 人工决策表（Codex 不可擅自最终决定）

对每个数据集记录：

| 判定项 | 可能结果 |
|---|---|
| 三模态完整性 | ready / partial / blocked |
| 标注类型 | official_box / derived_instance_box / candidate_only |
| 类别覆盖 | 单类别 / 多类别（以实际统计为准） |
| 空间与时间配准 | verified / imperfect / unavailable |
| 可靠测试协议 | official / grouped_custom / unresolved |
| 数据许可与复现条件 | permitted / restricted / unknown |
| 预期角色 | main_benchmark / auxiliary / robustness_only / exclude |
| 人工最终选择 | selected / hold / rejected |

正式主结果表只接纳**可靠、完整、许可合适**的检测标注和测试协议。不要因为样本量大就放松评估标签要求。人工确认：最终数据集、任务类别、统一参考坐标、划分策略、正式指标及实验资源上限，并记录在案（本次决策：2026-11-10，R-LiViT + MID-3K，见本文档顶部；转换参数见 `configs/datasets/`）。

**Phase 2 验收：** ✅ 已通过（2026-11-10）：可复核数据报告已产生，研究者已批准最终数据集组合。最终选择为 **R-LiViT + MID-3K**，其余三个数据集排除；下一阶段为 Phase 3 基线。

---

## E. Phase 3 — Ultralytics 通用多模态基线

### E1. 工程目标

- 保留 Ultralytics 作为 Submodule 或锁定依赖，不覆盖上游代码；新功能优先写在独立 Python Package 中。
- 实现统一多模态 `Dataset/DataLoader`、输入 `ModalityAdapter`、插入式 `FusionBlock`、模型工厂与评估入口。
- 保持各 YOLO 检测头/损失/后处理的原生行为，在指定特征层插入跨模态融合；无法复用官方 Trainer 的地方只做最小适配，并给出可复现说明。
- “兼容多个版本”必须由自动构建、前向、反向、训练及推理测试来证明，而非仅靠 model 名称映射。

候选 YOLO：v8、v9、v10、11、12、13（社区版本须单独检查兼容性）、26。**不要假设当前仓库原生支持所有版本；先输出 compatibility matrix。** 先选 1 个版本（建议 YOLO11s 或仓库中最稳定的 s 规模版本）开发，再验证 2–3 个差异较大的版本，最终扩大覆盖。

### E2. 必须具备的基线

- `B0`: RGB-only YOLO（官方预训练与标准训练）。
- `B1`: Thermal-only，`B2`: Depth-only（输入适配与预训练策略固定）。
- `B3`: RGB-T，`B4`: RGB-D，`B5`: T-D（统一的朴素 Fusion）。
- `B6`: RGB-T-D Early Concat。
- `B7`: RGB-T-D Feature Add / Concat（同位置融合）。
- `B8`: RGB-T-D Gated Fusion（普通可学习门控）。

比较时对模型规模、预训练初始化、输入分辨率、Epoch/更新步数、数据增强、Batch/有效 Batch、优化器、验证频率、选优规则进行控制。另增加同参数量/同 FLOPs 容量对照，避免把大模型优势归因于融合设计。

### E3. 训练记录与评测协议

- 划分：官方 split 优先；缺 val 时仅从训练组按场景/序列分组生成，测试集始终冻结。R-LiViT 官方只提供 train/test 序列，val 必须从 train 组内切出且不得与 test 重叠；MID-3K 无官方划分，按采集时间组切分。两者的测试集在确定后即冻结，不得用 Test 调参。
- 采用验证集 mAP50-95 选择单个最佳 checkpoint；不得根据 Test 调超参数/挑模型。
- 核心指标：mAP50-95、mAP50、Precision、Recall、per-class AP、small-object AP（视标注与数量是否支持）、Params、FLOPs、推理延迟/FPS、显存。
- 对主要比较进行至少 3 seeds，报告 mean ± std，并可给配对差值与置信区间；探索阶段先单 seed，筛选后再重复。
- 记录 `dataset/split checksum`、Git SHA、Ultralytics 版本、配置、随机种子、训练环境、输入位深/归一化、checkpoint 路径及 hash。
- 新方法和基线评估条件相同；推理延迟固定硬件、输入大小、Batch=1（以及可选吞吐 Batch），明确是否包含模态预处理。

**Phase 3 验收：**至少一个正式选定数据集完成单模态和三模态基础融合，配置驱动训练与可复现评估；跨版本兼容性清单能自动产生。

---

## F. Phase 4 — 方法创新：先可证伪，再扩展

### F1. 先做研究假设，再写 MoE

预期问题：直接融合会把受噪声、缺失或对齐误差影响的模态引入主干，从而导致负迁移。**先通过 B0–B8 和分场景分析确认问题存在**，再决定是否需要 MoE。

### F2. 建议的主方法：Reliability-Aware Mixture-of-Fusion-Experts

- **特征提取：** RGB 分支尽量复用预训练骨干；Thermal 与 Depth 使用可配置的轻量适配分支。共享单个 YOLO Neck/Head。
- **融合专家：** 首选 `RGB-T`、`RGB-D`、`T-D`、`RGB-T-D` 四个轻量专家；所有专家输出相同形状的残差特征。可以加入 RGB Identity/Skip 路径。
- **Router：** 基于多模态特征与可观察的深度质量代理输出专家权重；先 Dense/Soft 路由，之后尝试 Top-2 Sparse，但只有真正跳过未激活专家计算才能声称节省 FLOPs。
- **融合位置：** 先固定 YOLO 的一个中层输出（如 stride 16 对应位置；以实际模型结构为准）测试，之后再尝试多尺度 P3/P4/P5。不要在第一版同时引入复杂对齐、Attention 和多尺度模块。
- **深度可靠性：** 区分有效测量、插值值与缺失区域（若数据能支持）；所有代理特征需记录定义，不能当作真实传感器置信度。

### F3. 方法消融（主模型及主数据集先做）

| ID | 设计 | 要回答的问题 |
|---|---|---|
| A0 | Concat/Add | 固定融合基线 |
| A1 | 普通 Gated Fusion | 动态权重是否足够 |
| A2 | Fusion-MoE（无质量输入） | 专家变换是否有额外价值 |
| A3 | Fusion-MoE + Reliability Router | 质量引导是否有效 |
| A4 | A3 + Depth Validity/Quality 特征 | 质量定义是否进一步改善 |
| A5 | A3/A4 多尺度版本 | 跨尺度设计是否必要 |
| A6 | A3/A4 Top-2 Sparse | 稀疏化能否改善速度/代价 |
| C1 | 容量匹配的普通卷积融合 | 增益是否仅由参数增加导致 |

若某模块没有稳定增益，移除它，不为了凑创新点而保留。要求可视化专家使用率、路由熵、逐尺度权重，并检查专家塌缩与跨场景行为。

---

## G. Phase 5 — 一区目标导向的完整实验设计

### G1. 论文实验主表（正式数据集决定后）

1. **数据集主比较：** 每个正式数据集独立主表：RGB-only、最强双模态、三模态简单融合、标准门控、所提 MoE、可复现的已有多模态基线（需调研并公平适配）。
2. **模态组合消融：** RGB、T、D、RGB-T、RGB-D、T-D、RGB-T-D；至少在一个主数据集上齐全，在另一数据集验证关键结论。
3. **融合位置消融：** 输入层、Backbone、Neck 特征，以及一个固定检测头的公平实验；以实际框架支持为准。
4. **核心模块消融：** A0–A6 + C1，逐项验证 Router、Reliability、Experts、Multi-scale、Sparse。
5. **跨架构泛化：** 最终选取多个结构不同的 YOLO（先 v8 / 11 / 26，再视兼容性加入 v9/v10/v12/v13），每个版本固定尺度比较 RGB-only / Base Fusion / Proposed；**不必每个版本重做全部消融**。
6. **鲁棒性：** 受控模态缺失（RGB/T/D 单独丢弃）、深度无效比例、热图噪声、RGB 暗化/模糊、轻度配准偏移；强度分级并画性能退化曲线。退化过程只作为评估条件，不调 Test 参数。若有真实夜间/低质量子集，单独报告自然场景结果。
7. **效率与部署：** Params、GFLOPs、Batch=1 latency/FPS、峰值显存、专家实际调用次数；实测 Dense 与 Sparse 的差异。
8. **定性与错误分析：** 同一个目标在不同模态的可见性，真阳性/漏检/误检案例，专家权重与失败案例；不要只挑成功样本。
9. **统计可靠性：** 核心模型、主要数据集 3 seeds 起步；报告均值标准差、逐种子完整分数及配对比较，必要时增加重复。
10. **跨域实验（可选但加分）：** 相同检测类别时测试 Train-A → Test-B 的零样本跨域；与“分别训练并测试”的设置严格区分，记录传感器归一化与类别映射。

### G2. 结果发表的先决条件

- 至少有一个**真实多类别**且可信的检测 Benchmark。最终数据集中 R-LiViT 提供 8 类二维检测标注，MID-3K 仅有行人标注；两个数据集类别空间不同，必须分别报告协议与结果，不得合并 mAP 后声称统一的通用多类别检测性能。
- 新方法不仅需要超过 RGB-only，也必须超过容量相近的朴素融合与普通门控。
- 跨 YOLO 版本的扩展性由构建、训练及性能结果同时支撑；所声称支持的版本必须锁定具体 commit/version。
- 数据许可与衍生检测标注的可发布性要单独评估；不公开受限原始数据。

---

## H. Phase 6 — 论文材料整理与复现

建议在代码成熟前就维护论文级材料：

```text
paper/
  outline.md                 # Introduction, Related Work, Method, Experiments, Discussion
  contribution_claims.md     # 每个创新主张与对应证据，需随实验更新
  related_work.csv           # 方法 / 模态 / 数据集 / 是否开源 / 指标 / 区别
  protocol.md               # 数据划分、评价协议、实验环境
  tables/                   # 主表、消融、效率、鲁棒性、跨架构
  figures/                  # 网络、数据集样例、路由、错误案例
  limitations.md            # 单类别数据、传感器对齐、领域差异等限制
experiments/
  registry.csv               # 每次训练与评估的唯一身份、配置、结果、版本
  configs/
  results/
```

不应先写“MoE 显著优于所有模型”等未经实验支持的结论；先积累带配置与 checkpoint hash 的可信结果，再确定论文贡献表达。

---

## I. 实施顺序与 Go / No-Go 关口

| 顺序 | 工作 | 必须提交的结果 | 人工批准后才能进入下一阶段？ |
|---|---|---|---|
| 0 | 原始数据与仓库审计 | Inventory + Profiles | ✅ 已完成 |
| 1 | 五个 Adapter 与统一转换 | Processed 数据 + Tests + Previews | ✅ 已完成 |
| 2 | 数据统计与选择 | Statistics + Selection Report | ✅ 已通过（2026-11-10，选定 R-LiViT + MID-3K） |
| 3 | 可复现 YOLO 基线 | RGB/T/D + 简单融合主表 | 是（确定 MoE 必要性） |
| 4 | MoE 与 Reliability 逐步研发 | 完整消融 + 容量对照 | 是（确认实际贡献） |
| 5 | 扩展数据集/跨 YOLO 版本/鲁棒性 | 完整论文结果矩阵 | 是（确定投稿定位） |
| 6 | 论文与复现包 | Tables, Figures, Methods, Limits | 最终检查 |

### 首轮 Codex 任务记录（Phase 0–2，已完成）

以下为已完成的第一轮工程任务，保留作为执行记录；其「停止等待人工决定」的要求已由 2026-11-10 的人工选择解除。

1. 仔细阅读现有仓库；输出 `docs/repo_audit.md`，说明可复用组件与修改点，不直接修改上游 Submodule。
2. 新建本计划中的统一 schema、dataset registry、5 个 Adapter 文件和可复用 CLI。**先以实际原始文件为准完成调查，再补各 Adapter；未下载的数据保留 pending 状态。**
3. 实现严格的 YOLO 标注转换、实例 Mask→Box、LiDAR→Depth 可选模块、数据合法性检查及统计；转换失败样本要产生诊断日志。
4. 在真实可获取的小样本上跑通 `inspect → convert → validate → summarize → preview`，进行端到端 Smoke Test；如无原始数据，使用合成 fixture 测试并明确其非真实统计。
5. 输出数据集对比报告，自动包含类别、实例数、模态完整性、标签可信度、分布、数据划分风险、代表性可视化及需人工决定的事项。
6. ~~此时停止，不做数据集最终取舍~~ 已解除：人工选择已于 2026-11-10 完成（R-LiViT + MID-3K），可进入 Phase 3；仍不臆造缺失的标定/标签，Phase 3 之后再开始 MoE。

### 交给 Codex 的首次任务指令

> 请先阅读 `Multimodal_YOLO_Paper_Plan.md`，只执行其中 Phase 0、Phase 1 和 Phase 2 的工程工作，不启动正式模型训练。优先完成五个候选数据集的只读原始文件审计、统一数据 Schema、可扩展 Adapter、YOLO Detection 标注转换、传感器对齐/Depth 投影的可选工具、数据合法性检查、可视化抽检与统计报告。必须保留原始数据、官方划分、原始类别、位深及可追溯元信息；将 VDT-2048 的二值显著性 Mask 视为候选标注而非完整多类别 Ground Truth。未下载完或缺标定的数据集标记 pending/blocked，不能猜测或伪造数据。尽量以新增独立文件实现，保留 Ultralytics 上游不变。提供运行命令、测试及实际产物路径。完成数据汇总后停止，等待我人工确定正式数据集，再进入 YOLO 基线和 MoE 设计。
