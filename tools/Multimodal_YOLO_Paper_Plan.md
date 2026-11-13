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

- [x] **状态：** 已完成（Inventory + Profiles 已交付）

### B1. Codex 需要先检查的内容

1. 阅读当前仓库结构、Ultralytics Submodule/安装版本、已有 `datasets/`、`models/`、`scripts/`、训练与评估代码；**尽量增加独立文件，避免直接改动上游 Ultralytics 源码**。
2. 对五个下载目录执行只读扫描，生成 `raw_inventory.json`：路径、文件数、文件类型、文件大小、图像分辨率、数据位深、可能的标注格式、序列/场景/时间信息、标定文件以及缺失情况。
3. 为每个数据集建立 `dataset_profile.yaml`：官方主页、论文、版本/发布日期、数据许可、类别定义、原始模态、深度来源、是否已经配准、标注坐标系、官方划分、质量风险和转化方案。未知字段标记 `unknown`，不要猜测。
4. 先做轻量样本检查（例如每种模态、每种标注格式各取数例），再编写实际解析器；未下载完成的目录标记 `pending_download`，不要因缺少文件整体失败。
5. 形成数据集对应关系和风险清单，提交人工查看。**此阶段不启动正式训练。**

### B2. 五个数据集分别要重点核查的事项（Phase 0 审计记录）

下表是 Phase 0 对五个候选的核查结论，保留作为审计依据。最终只选用 MID-3K 与 R-LiViT。

| 数据集     | 默认任务定位（待实物核实）         | 转换重点 / 风险                                                                                                            |
| ---------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| MID-3K     | RGB/T/Depth，官方人类检测框        | 四模态文件一一配对；RGB/Depth 与 Thermal 可能使用不同检测框和坐标参考；采集日期分组；LiDAR 派生深度及轻微时间错位          |
| InfraParis | 三模态道路场景，官方检测主要为行人 | 官方可用的 RGB/T/Depth 与标注划分；热像视场及投影；不能直接把语义分割多类 Mask 视作实例检测标签                            |
| R-LiViT    | RGB/T + LiDAR，多类别二维检测      | 确认 RGB-T 2D 标注类别/参考视图及可用标定；同步匹配点云→深度；不能错误地把 3D LiDAR 类别当 2D 类别                         |
| MM5        | 五模态对齐室内物品，多类实例标注   | 选 RGB/T/Depth；解析 `ANNO_INST` + `META` 或官方 COCO 导出；保留原始实例类别与来源；相同场景多光照不能跨训练/测试拆开      |
| VDT-2048   | RGB/T/Depth 显著目标检测           | 官方 Mask 主要是显著/背景标签，**34 种物品不等于34类现成检测 GT**；可产出候选框，未经实例和类别核验不得纳入多类别 mAP 主表 |

参考链接：

- MID-3K: https://github.com/kennedyk1/MID-3K
- InfraParis: https://github.com/ENSTA-U2IS-AI/infraParis
- R-LiViT: https://github.com/XITASO/r-livit
- MM5: https://github.com/martinbrennernz/MM5-Dataset
- VDT-2048: https://github.com/VDT-2048/VDT-Dataset

**Phase 0 交付：** `reports/data_inventory/`、五份 profile、原始目录扫描日志、缺失/待定项目列表。没有数据文件时 Codex 只建立适配器骨架、单元测试样本和待办清单，不能伪造统计数字。

---

## C. Phase 1 — 统一数据格式与转换工具

- [x] **状态：** 已完成（五个 Adapter、转换与校验链路已交付）

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
2. **实例分割：** 使用每个实例的独立 Mask 生成最小外接框，并从官方 metadata 获取实例类别；过滤条件（最小可见面积、忽略标签等）配置化，保留被排除对象的日志；目标是可复现的 _derived detection annotations_。
3. **语义分割 / 二值显著性：** 只能生成 _candidate boxes_，**不默认产生可信多类别 GT**。对象相接、遮挡、多个显著对象时需人工抽检/重新标注；缺少类别来源时标记 `annotation_status=candidate_only`。
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

- [x] **状态：** 已完成（2026-11-10 通过人工关口）

统计与选择报告已生成在 `reports/data_statistics/`；研究者已显式批准最终组合 **R-LiViT + MID-3K**，可进入 Phase 3。近重复/感知哈希检测按人工决定不实现，泄漏风险以 group / 序列 / 时间邻近检测为准；box 几何与存储估算、图表类交付不阻塞本关口。

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

| 判定项             | 可能结果                                               |
| ------------------ | ------------------------------------------------------ |
| 三模态完整性       | ready / partial / blocked                              |
| 标注类型           | official_box / derived_instance_box / candidate_only   |
| 类别覆盖           | 单类别 / 多类别（以实际统计为准）                      |
| 空间与时间配准     | verified / imperfect / unavailable                     |
| 可靠测试协议       | official / grouped_custom / unresolved                 |
| 数据许可与复现条件 | permitted / restricted / unknown                       |
| 预期角色           | main_benchmark / auxiliary / robustness_only / exclude |
| 人工最终选择       | selected / hold / rejected                             |

正式主结果表只接纳**可靠、完整、许可合适**的检测标注和测试协议。不要因为样本量大就放松评估标签要求。人工确认：最终数据集、任务类别、统一参考坐标、划分策略、正式指标及实验资源上限，并记录在案（本次决策：2026-11-10，R-LiViT + MID-3K，见本文档顶部；转换参数见 `configs/datasets/`）。

**Phase 2 验收：** ✅ 已通过（2026-11-10）：可复核数据报告已产生，研究者已批准最终数据集组合。最终选择为 **R-LiViT + MID-3K**，其余三个数据集排除；下一阶段为 Phase 3 基线。

---

## E. Phase 3 — Ultralytics 通用多模态基线

- [ ] **状态：** 进行中（E1、E1.5 已完成；E2 进行中，E3 未开始）

**结果口径约定：**「**正式训练结果**」= 200 epoch、3 seed、COCO 预训练、完整数据，其 val/test 数值可作为实验结论引用；「**预先测试 / 冒烟**」= 1–3 epoch 或 fraction<1 的短跑，只验证数据/训练/推理链路与显存，其 val mAP 不得作为基线或论文结果引用。

### E1. 工程目标

- [x] **状态：** 已完成（2026-11-10）

- 保留 Ultralytics 作为 Submodule 或锁定依赖，不覆盖上游代码；新功能优先写在独立 Python Package 中。
- 实现统一多模态 `Dataset/DataLoader`、输入 `ModalityAdapter`、插入式 `FusionBlock`、模型工厂与评估入口。
- 保持各 YOLO 检测头/损失/后处理的原生行为，在指定特征层插入跨模态融合；无法复用官方 Trainer 的地方只做最小适配，并给出可复现说明。
- “兼容多个版本”必须由自动构建、前向、反向、训练及推理测试来证明，而非仅靠 model 名称映射。

候选 YOLO：v8、v9、v10、11、12、13（社区版本须单独检查兼容性）、26。**不要假设当前仓库原生支持所有版本；先输出 compatibility matrix。** 先选 1 个版本（建议 YOLO11s 或仓库中最稳定的 s 规模版本）开发，再验证 2–3 个差异较大的版本，最终扩大覆盖。**2026-11-10 人工决定：本实验不纳入 YOLOv13**（当前 Ultralytics 8.4.91 亦未提供其模型配置）。

**开发验证记录（2026-11-10）：** 人工选定首个架构为 **YOLO26x**（x 规模，优先验证大模型下的工程正确性）。`ultralytics/cfg/mmodels/yolo26x-mm3-{if,ef,nif,bf,nf,hf}.yaml` 六个融合模式均已通过构建、前向、反向测试（`tests/test_multimodal.py::test_yolo26x_fusion_modes_build_and_backpropagate`），并在 **R-LiViT_mdet** 上各完成 1 epoch（10% 数据）的训练、验证、checkpoint 重载与配对推理；其中 BF 另在 **MID-3K_mdet** 上完成同样流程。该快速训练仅证明数据/训练/推理链路可用，不作为基线精度结论。2 模态模板 `yolo26x-mm2-{31,11}-{if,ef,nif,bf,nf,hf}.yaml` 已补齐：3+1（4 通道，RGB + 单通道 T 或 D）与 1+1（2 通道，T + D）两种布局，六个融合模式均通过构建/前向/反向测试（`tests/test_multimodal.py::test_yolo26x_mm2_templates_match_the_dataset_modality_split`），对应 B3–B5 的两类输入布局；两种模态组合的 `data.yaml` 子集与真实数据训练尚未执行。v8–v12 的 s 规模 mm3 六模式模板（共 30 个）已通过构建/前向/反向测试（`tests/test_multimodal.py::test_other_yolo_templates_build_and_backpropagate`），并各自在 R-LiViT_mdet 上完成 1 epoch 训练、验证与 `best.pt` 重载推理；同一生成器已补齐 v8–v12 的全部上游 scale（v8/11/12 各 n/s/m/l/x，v9 t/s/m/c/e，v10 n/s/m/b/l/x，共 156 个 mm3 模板），由 `tests/test_multimodal.py::test_every_template_scale_builds_and_grows_with_the_scale` 验证每个 scale 均可构建、前向且参数量随 scale 单调增长（真实数据仍只跑过 s）；v8x 的 mm3/mm2/pre-SPPF 模板与 YOLO11 seg 模板仅完成构建/前向/反向验证。证据与覆盖范围见 [`tools/multimodal_compatibility.md`](multimodal_compatibility.md)。**E1 工程目标（统一多模态接口 + 跨版本自动验证）至此完成（2026-11-10）。**

**配置统一（2026-11-10）：** Ultralytics 的 `yaml_model_load` 不支持跨文件继承，因此 180 个模板共用的契约（`channels`、`input_sections`、`operator`、`share_weight`、`pretrained` 策略）统一定义在 `ultralytics/cfg/mmodels/base.yaml`，由 `MultiModalModelMixin` 在构建前通过 `load_multimodal_config` 深度合并（模板内声明的键优先）。模板只保留自身差异：`mm2` 文件覆写通道切分，其余全部继承 `pretrained: {source: auto}` 且不手写层映射。**所有家族（v8/v9/v10/v11/v12/YOLO26）的模板均可加载本家族同规模 COCO 预训练权重**，层映射由 `derive_copy_plan` 依据 donor 与 target 的 YAML 图自动推导；`yolov9e`、`yolo11`/`yolo12` 的 head/neck 投影宽度与 `yolov9e` bf/ef/nif 的融合宽度已按 donor 实际输出通道修正。覆盖性由 `tests/test_multimodal.py::test_every_template_inherits_the_shared_contract`（配置）与 `::test_every_template_loads_its_family_coco_checkpoint`（27 个家族/规模组合 × 全部 180 个模板，跳过张量为 0）校验。

**预先测试（非正式结果）** — 连通性 smoke 结果（YOLO26x BF，1 epoch、fraction=0.1、640 px、batch 8、`pretrained=False`，RTX 5090）：

| 数据集       | Train box/cls/dfl     | Val box/cls   | P      | R      | mAP50-95 | 推理        |
| ------------ | --------------------- | ------------- | ------ | ------ | -------- | ----------- |
| MID-3K_mdet  | 4.016 / 9.519 / 0.029 | 4.146 / 8.602 | 0.0005 | 0.0343 | 0.0000   | 5.49 ms/img |
| R-LiViT_mdet | 4.368 / 5.659 / 0.016 | 4.393 / 5.587 | 0.0000 | 0.0038 | 0.0000   | 3.97 ms/img |

推理阶段两个数据集各用 4 张 val 主模态图（路径列表）+ 1 张（单文件）验证配对加载，均成功；置信度贴近 `conf=0.001` 下限（MID-3K 0.0119、R-LiViT 0.0016），符合 10% 训练量的预期。**这些数字只用于验证链路，不得作为基线或论文结果引用。**

**预先测试（非正式结果）** — 融合模式横向对比（R-LiViT_mdet，1 epoch、fraction=0.1、640 px、batch 4、`pretrained=False`，RTX 5090；batch 与上表不同，推理耗时不可直接比较）：

| 模式 | 融合位置                | 参数量      | 峰值显存 | Train box/cls/dfl     | Val box/cls   | P      | R      | mAP50-95 | 推理        |
| ---- | ----------------------- | ----------- | -------- | --------------------- | ------------- | ------ | ------ | -------- | ----------- |
| IF   | 输入端拼接              | 58,814,916  | 6.86 GB  | 4.483 / 5.620 / 0.016 | 4.411 / 5.581 | 0.0000 | 0.0240 | 0.0000   | 2.03 ms/img |
| EF   | encoder P3/P4/P5        | 112,334,148 | 11.85 GB | 4.414 / 5.498 / 0.016 | 4.405 / 5.586 | 0.0001 | 0.0114 | 0.0000   | 4.11 ms/img |
| NIF  | encoder P3/P4 + nape P5 | 115,287,876 | 12.00 GB | 4.311 / 5.397 / 0.015 | 4.393 / 5.588 | 0.0000 | 0.0000 | 0.0000   | 4.26 ms/img |
| BF   | backbone P3/P4/P5       | 121,817,412 | 12.41 GB | 4.368 / 5.659 / 0.016 | 4.393 / 5.587 | 0.0000 | 0.0038 | 0.0000   | 3.97 ms/img |
| NF   | neck P3/P4/P5           | 167,899,716 | 15.63 GB | 4.379 / 5.438 / 0.015 | 4.412 / 5.589 | 0.0000 | 0.0057 | 0.0000   | 5.44 ms/img |
| HF   | head P3/P4/P5           | 171,892,548 | 15.95 GB | 4.303 / 5.387 / 0.015 | 4.391 / 5.566 | 0.0000 | 0.0000 | 0.0000   | 5.50 ms/img |

BF 行取自上表的 batch 8 运行，其余为 batch 4，推理耗时不可直接横向比较。每个模式均重载各自 `best.pt`，对 4 张 val 主模态图（路径列表）推理，`conf=0.001` 下均输出 300 个框（`max_det` 上限），最高置信度 0.0016–0.0017，与 1 epoch/10% 数据量一致。**同样只用于链路验证。**

表中的峰值显存为同协议正式 run 的实测值（`torch.cuda.max_memory_allocated`，batch 4 / 640 px，RTX 3090，3 seed 最大值，见 E1.5 正式训练执行记录）；本 1-epoch 冒烟运行在 RTX 5090 且未单独记录显存。峰值显存主要由 batch、输入尺寸、模型结构与优化器状态决定，与 epoch 数无关。

### E1.5 — 融合方式预筛选（YOLO26x，正式训练前）

- [x] **状态：** 已完成（2026-11-13）

在铺开「全部数据集 × 全部模型」的正式训练之前，先用 **YOLO26x** 的六种融合模式（if / ef / nif / bf / nf / hf）做一次受控对比，确认融合位置的差异是否稳定可见，再决定正式实验把哪些模式作为主线、哪些降为消融。

- **为什么先用 YOLO26x：** 它的六个模板已完成构建、前向、反向测试，并且是当前唯一在真实数据上把六种模式全跑过一遍链路的架构，筛选成本最低。
- **前置条件：** 预训练权重现已对全部家族与规模可用，本轮统一使用 `pretrained: {source: auto}`；不开预训练时各模式都接近随机初始化，差异基本是噪声。
- **控制变量：** 同一数据比例、epoch、batch、输入尺寸、随机种子与增强，六个模式只改融合位置。
- **结论使用限制：** 六模式参数量从 58.8M（IF）到 171.9M（HF）相差约 3 倍，精度差异会混入容量因素。本步只用于**筛选与排序**，不能作为「融合位置更优」的论文结论；正式结论需要 F3 的容量对照（C1）与多 seed 支撑。
- **产物：** 一张「模式 → 参数量 / 训练损失 / val mAP / 推理延迟」表，作为正式实验选择融合模式与分配算力的依据。

#### E1.5 正式训练执行记录与结果（200 epoch × 3 seed，2026-11-13）

**训练配置：** R-LiViT_mdet（train 1704 / val 216 / test 480，8 类，5 通道 RGB+T+D）；YOLO26x mm3 六模式，统一加载同家族 COCO 预训练（`pretrained: {source: auto}`）；AdamW，lr0=1e-3、lrf=0.01、cos_lr、warmup 0.5 epoch、momentum 0.937、weight_decay 5e-4，cosine 按 200 epoch 退火；batch=4、accumulate=16（等效 nbs=64）、imgsz=640、epochs=200、close_mosaic=10、deterministic=True；seed 0/1/2 各占一张 RTX 3090（24 GiB）。batch=8 在 24 GiB 上 OOM（训练器自动降为 4），因此 18 个 run 统一以 batch=4 重启，`runs/e15_rlivit_b4_200ep/` 下每个 `args.yaml` 即实际配置。

**完成情况：** 18/18 run 完成（2026-11-13 06:48–06:55 UTC），总耗时约 38.6 h（3 GPU 并行）。每 epoch 耗时 IF 67–68 s、EF/NIF 107–110 s、BF 114 s、NF 140–144 s、HF 142–144 s；峰值显存 6.8–16.0 GB（IF 最低，HF 最高）。EF seed2 触发早停（best epoch 98，patience=100，实际 198 epoch）。逐 seed、逐类别的完整数据见 `runs/e15_rlivit_b4_200ep/e15_val_test_results.xlsx`（Summary / Val_Detail / Test_Detail / Test_PerClass）与 `test_eval/*.json`。

**正式训练验证集结果（200 epoch、3 seed；训练中每 epoch 验证，取最后一轮；mAP50-95）：**

| 模式 | 融合位置                | 参数量  | s/epoch | seed 0 | seed 1 | seed 2 | mean ± std          | mAP50  |
| ---- | ----------------------- | ------- | ------- | ------ | ------ | ------ | ------------------- | ------ |
| BF   | backbone P3/P4/P5       | 121.8 M | 114.4   | 0.4445 | 0.4424 | 0.4381 | **0.4417 ± 0.0032** | 0.6755 |
| IF   | 输入端拼接              | 58.8 M  | 67.9    | 0.4360 | 0.4448 | 0.4346 | 0.4385 ± 0.0055     | 0.6539 |
| HF   | head P3/P4/P5           | 171.9 M | 142.6   | 0.4356 | 0.4428 | 0.4285 | 0.4356 ± 0.0072     | 0.6634 |
| EF   | encoder P3/P4/P5        | 112.3 M | 108.7   | 0.4428 | 0.4326 | 0.4309 | 0.4354 ± 0.0064     | 0.6541 |
| NIF  | encoder P3/P4 + nape P5 | 115.3 M | 109.7   | 0.4314 | 0.4406 | 0.4295 | 0.4339 ± 0.0060     | 0.6610 |
| NF   | neck P3/P4/P5           | 167.9 M | 142.1   | 0.4318 | 0.4393 | 0.4082 | 0.4264 ± 0.0162     | 0.6506 |

`s/epoch` 为 3 seed 均值，逐 run 原始值在各 run 目录的 `pilot.json`；六模式峰值显存统一记录在 E1 的「融合模式横向对比」表。

**正式训练测试集结果（best.pt 按 val fitness 选出后一次性评估 480 张；mAP50-95）：**

| 模式 | seed 0 | seed 1 | seed 2 | mean ± std          | mAP50  | 与 BF 配对差 | val − test |
| ---- | ------ | ------ | ------ | ------------------- | ------ | ------------ | ---------- |
| BF   | 0.4397 | 0.4309 | 0.4324 | **0.4343 ± 0.0047** | 0.6723 | —            | +0.0073    |
| NIF  | 0.4309 | 0.4291 | 0.4331 | 0.4310 ± 0.0021     | 0.6707 | −0.0033      | +0.0028    |
| IF   | 0.4313 | 0.4316 | 0.4258 | 0.4296 ± 0.0033     | 0.6626 | −0.0048      | +0.0089    |
| EF   | 0.4262 | 0.4361 | 0.4209 | 0.4277 ± 0.0077     | 0.6663 | −0.0066      | +0.0077    |
| HF   | 0.4319 | 0.4237 | 0.4184 | 0.4247 ± 0.0068     | 0.6600 | −0.0097      | +0.0110    |
| NF   | 0.4203 | 0.4221 | 0.4216 | 0.4213 ± 0.0009     | 0.6613 | −0.0130      | +0.0051    |

**排序与判断：** BF 在 val 与 test 两个划分上都排第 1，NIF、IF 分列 test 第 2、3，NF、HF 垫底。BF 相对其余模式的 test 均值优势为 0.3–1.3 个点，但 3 个 seed 的配对差波动很大（例如 NIF 在 seed 0/1 为负、seed 2 为正），**n=3 时所有两两差异均不显著（t≈1.0–2.0，p≈0.2–0.4）**，只能按排序而非显著性使用。BF 在领先组中 val std 最小（0.0032），也是重复性最稳的模式。

**per-class 观察（test）：** escooter 是主要短板，AP50-95 仅 0.04–0.11；car ≈0.63、person ≈0.41、tramway ≈0.62–0.71。类别间差异远大于模式间差异，正式实验应把 escooter/小目标单独报告。

**筛选结论：** 正式实验主线选 **BF**；**NIF / IF / EF 保留为消融**；**NF / HF 降为补充实验**（参数量最大且 test 无优势）。该结论只用于分配算力，不能写成「backbone 融合最优」的论文结论：六模式容量相差约 3 倍，且 seed 数不足；F3 的容量对照（C1）与更多 seed 才能支撑正式结论。推理延迟留到 E3 按统一协议（batch=1）测量，本轮不作为筛选依据。

**已知问题（E2 前处理或记录）：**

- NF seed2 从 epoch 11 起 val box/cls/dfl loss 为 NaN（持续 127/200 epoch），val mAP 仍为有限值，训练正常结束；HF/NF 其余 seed 也有零散 NaN epoch。根因未定位；当前影响是 val loss 不能用于曲线或早停，不影响 mAP 与 test 评估。
- EF seed2 早停于 198 epoch（best 98），与其他 run 的 200 epoch 不完全对齐。
- 训练中验证使用 val，`best.pt` 也按 val fitness 选择；上表 test 列为干净留出结果，论文最终数字应以 test 为准，val 仅用于选模与对照。

### E2. 必须具备的基线

- [ ] **状态：** 进行中（2026-11-14 改为 BF-based branch 消融：仅 B0–B5 新训练，B7 复用 E1.5；B8/C1 移至 F3）

- `B0`: RGB-only YOLO（官方预训练与标准训练，单分支参照）。
- `B1`: Thermal-only，`B2`: Depth-only（单分支参照，单输入无融合）。
- `B3`: RGB-T，`B4`: RGB-D，`B5`: T-D（`mm2-*-bf`，与 B7 同一 BF 融合架构的双分支消融）。
- `B7`: RGB-T-D BF（三分支基准，复用 E1.5）。

B8（Gated Fusion）与 C1（容量匹配普通卷积）不属于 E2，放在 F3 方法消融。

比较时对模型规模、预训练初始化、输入分辨率、Epoch/更新步数、数据增强、Batch/有效 Batch、优化器、验证频率、选优规则进行控制。另增加同参数量/同 FLOPs 容量对照，避免把大模型优势归因于融合设计。

#### E2 执行方案（2026-11-13 确定，2026-11-14 改为 BF-based branch 消融）

**数据顺序：** 先 R-LiViT_mdet（E1.5 链路已就绪），完成后按同一矩阵铺开 MID-3K_mdet。

**统一训练配方（与 E1.5 完全一致）：** COCO 预训练初始化；AdamW lr0=1e-3、lrf=0.01、cos_lr、warmup_epochs=0.5（warmup_bias_lr=0.1、warmup_momentum=0.8）、momentum=0.937、weight_decay=5e-4；batch=4 × accumulate=16（nbs=64）；imgsz=640；200 epochs；close_mosaic=10；deterministic；patience=100；seeds 0/1/2。所有 run 共用同一 split、增强与更新步预算。

**E2 = 基于 BF 的 branch 消融：3 个单分支输入参照 + 3 个双分支 BF 融合；B7（三分支 BF）为基准。**

| ID | 输入 | 数据 YAML（R-LiViT_mdet） | 模型模板 | 分支/融合 | 来源 |
| --- | --- | --- | --- | --- | --- |
| B0 | RGB (3ch) | `data-rgb.yaml` | 官方 `yolo26x.yaml`（标准 `DetectionModel`） | 1 分支，无融合 | 新训练 ×3 |
| B1 | Thermal (1ch) | `data-thermal.yaml` | `yolo26x-mm1-if.yaml` | 1 分支，无融合 | 新训练 ×3 |
| B2 | Depth (1ch) | `data-depth.yaml` | `yolo26x-mm1-if.yaml` | 1 分支，无融合 | 新训练 ×3 |
| B3 | RGB-T (4ch) | `data-rgb-t.yaml` | `yolo26x-mm2-31-bf.yaml` | 2 分支，BF P3/P4/P5 | 新训练 ×3 |
| B4 | RGB-D (4ch) | `data-rgb-d.yaml` | `yolo26x-mm2-31-bf.yaml` | 2 分支，BF P3/P4/P5 | 新训练 ×3 |
| B5 | T-D (2ch) | `data-t-d.yaml` | `yolo26x-mm2-11-bf.yaml` | 2 分支，BF P3/P4/P5 | 新训练 ×3 |
| B7 | RGB-T-D (5ch) | `data.yaml`（E1.5） | `yolo26x-mm3-bf.yaml` | 3 分支，BF（**基准**） | 复用 E1.5 BF ×3 |

**已定决策：**

- E2 的模态贡献消融全部在 **BF 架构内**：B3–B5 与 B7 只差模态集合，分支数、融合点、neck/head、配方、seed 完全一致；不使用 IF 做双分支消融，避免融合架构与模态集合同时变化。
- B0–B2 是单分支参照，本身没有可融合的第二路特征；mm1 的 `IF` 只是单输入模板的命名，不是早期特征融合。
- B6（5 通道 IF）不再属于 E2；E1.5 的 IF 结果保留为“早期融合 vs BF”的补充证据，不进 E2 表。
- B1/B2 采用**单通道 mm1 模板**（不复制成三通道）：`channels: 1`、`input_sections: [1]`；COCO RGB 首层权重按输入通道均值适配（`input_init="mean"`），与 mm3 非 RGB 分支逻辑一致。已用 thermal/depth 各 1 epoch 真实数据冒烟验证（`copied=1091, transformed=1, skipped=0`）。
- B7 直接复用 E1.5 的 BF 三个 seed（同数据、同配方），不重复训练。
- B0 使用官方 `yolo26x.pt`；B1–B5 均由 COCO 权重迁移初始化。B3/B4 共用同一个 3+1 模板，仅数据 YAML 不同。
- 新增 run 共 18 个（B0–B5 各 ×3）；B3–B5 由 IF 改为双分支 BF 后单 run 约 4.5 h，B0–B2 约 3.5 h，单 seed 约 24 h，3×RTX 3090 墙钟约 24 h（约 72 GPU·h）；复用 B7 节省约 19 GPU·h。MID-3K 在 R-LiViT 完成后执行。
- 工程状态（2026-11-14）：3 epoch × 3 seed **预先测试**完成，18/18 run 正常（`runs/e2_rlivit_3ep/`，全部有 best.pt、无 OOM）：b0 73 s/epoch 6.0 GB，b1/b2 68–71 s 5.9 GB，b3/b4/b5 97–101 s 9.5–10.3 GB；B3–B5 的 BF 模板首次在真实数据上完成训练/验证。该测试只验证链路，3 epoch 的 val mAP 不作为基线结果；下一步进入 200 epoch 正式训练。

**E2 预先测试记录（2026-11-14，3 epoch × 3 seed；非正式训练结果）：**

| 条件 | 输入 | 参数量 | s/epoch | 峰值显存 | 3ep val mAP50-95 | 3ep val mAP50 | P | R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 RGB | 3ch | 59.0M | 73.0 | 5.97 GB | 0.2935±0.0164 | 0.4721±0.0206 | 0.693 | 0.413 |
| B1 Thermal | 1ch | 59.0M | 71.0 | 5.93 GB | 0.0971±0.0120 | 0.2297±0.0067 | 0.428 | 0.229 |
| B2 Depth | 1ch | 59.0M | 68.2 | 5.93 GB | 0.0016±0.0019 | 0.0055±0.0068 | 0.091 | 0.026 |
| B3 RGB-T | 4ch | 94.0M | 99.6 | 9.50 GB | 0.1876±0.0114 | 0.3437±0.0144 | 0.609 | 0.318 |
| B4 RGB-D | 4ch | 94.0M | 101.1 | 10.27 GB | 0.1696±0.0076 | 0.3196±0.0115 | 0.554 | 0.307 |
| B5 T-D | 2ch | 94.0M | 97.4 | 10.22 GB | 0.0340±0.0041 | 0.0960±0.0126 | 0.391 | 0.125 |

B7 未参与该测试（E1.5 为 200 epoch，不可比）。生成命令：`python tools/summarize_e2_baselines.py --project runs/e2_rlivit_3ep --e15-project <不存在的路径>`，避免把 E1.5 的 200 epoch B7 行并入 3 epoch 表。

### E3. 训练记录与评测协议

- [ ] **状态：** 未开始

- 划分：官方 split 优先；缺 val 时仅从训练组按场景/序列分组生成，测试集始终冻结。R-LiViT 官方只提供 train/test 序列，val 必须从 train 组内切出且不得与 test 重叠；MID-3K 无官方划分，按采集时间组切分。两者的测试集在确定后即冻结，不得用 Test 调参。
- 采用验证集 mAP50-95 选择单个最佳 checkpoint；不得根据 Test 调超参数/挑模型。
- 核心指标：mAP50-95、mAP50、Precision、Recall、per-class AP、small-object AP（视标注与数量是否支持）、Params、FLOPs、推理延迟/FPS、显存。
- 对主要比较进行至少 3 seeds，报告 mean ± std，并可给配对差值与置信区间；探索阶段先单 seed，筛选后再重复。
- 记录 `dataset/split checksum`、Git SHA、Ultralytics 版本、配置、随机种子、训练环境、输入位深/归一化、checkpoint 路径及 hash。
- 新方法和基线评估条件相同；推理延迟固定硬件、输入大小、Batch=1（以及可选吞吐 Batch），明确是否包含模态预处理。

**Phase 3 验收：**至少一个正式选定数据集完成单模态和三模态基础融合，配置驱动训练与可复现评估；跨版本兼容性清单能自动产生。

---

## F. Phase 4 — 方法创新：先可证伪，再扩展

- [ ] **状态：** 未开始

### F1. 先做研究假设，再写 MoE

预期问题：直接融合会把受噪声、缺失或对齐误差影响的模态引入主干，从而导致负迁移。**先通过 B0–B7 和分场景分析确认问题存在**，再决定是否需要 MoE。

### F2. 建议的主方法：Reliability-Aware Mixture-of-Fusion-Experts

- **特征提取：** RGB 分支尽量复用预训练骨干；Thermal 与 Depth 使用可配置的轻量适配分支。共享单个 YOLO Neck/Head。
- **融合专家：** 首选 `RGB-T`、`RGB-D`、`T-D`、`RGB-T-D` 四个轻量专家；所有专家输出相同形状的残差特征。可以加入 RGB Identity/Skip 路径。
- **Router：** 基于多模态特征与可观察的深度质量代理输出专家权重；先 Dense/Soft 路由，之后尝试 Top-2 Sparse，但只有真正跳过未激活专家计算才能声称节省 FLOPs。
- **融合位置：** 先固定 YOLO 的一个中层输出（如 stride 16 对应位置；以实际模型结构为准）测试，之后再尝试多尺度 P3/P4/P5。不要在第一版同时引入复杂对齐、Attention 和多尺度模块。
- **深度可靠性：** 区分有效测量、插值值与缺失区域（若数据能支持）；所有代理特征需记录定义，不能当作真实传感器置信度。

### F3. 方法消融（主模型及主数据集先做）

| ID  | 设计                             | 要回答的问题             |
| --- | -------------------------------- | ------------------------ |
| A0  | Concat/Add                       | 固定融合基线             |
| A1  | 普通 Gated Fusion（B8 模板 `yolo26x-mm3-bf-gated.yaml`） | 动态权重是否足够         |
| A2  | Fusion-MoE（无质量输入）         | 专家变换是否有额外价值   |
| A3  | Fusion-MoE + Reliability Router  | 质量引导是否有效         |
| A4  | A3 + Depth Validity/Quality 特征 | 质量定义是否进一步改善   |
| A5  | A3/A4 多尺度版本                 | 跨尺度设计是否必要       |
| A6  | A3/A4 Top-2 Sparse               | 稀疏化能否改善速度/代价  |
| C1  | 容量匹配的普通卷积融合（模板 `yolo26x-mm3-bf-conv.yaml`） | 增益是否仅由参数增加导致 |

> B8/C1 实现细节：两者都在 BF 的 backbone P3/P4/P5 融合点先做恒等初始化的 1×1 投影；B8 再乘 `2*sigmoid(gate(x))`（gate 零初始化，训练起点与 BF 完全一致），C1 改为加零初始化残差卷积。二者参数逐位相同（x scale、nc=2：132,448,068 vs BF 121,817,412，各 +10,630,656），差值只来自门控；`MultiModalFusion` 的 `gated`/`conv` 算子与单测已就绪，F 阶段直接复用。

若某模块没有稳定增益，移除它，不为了凑创新点而保留。要求可视化专家使用率、路由熵、逐尺度权重，并检查专家塌缩与跨场景行为。

---

## G. Phase 5 — 一区目标导向的完整实验设计

- [ ] **状态：** 未开始

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

- [ ] **状态：** 未开始

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

| 顺序 | 工作                           | 必须提交的结果                    | 人工批准后才能进入下一阶段？                   | 状态 |
| ---- | ------------------------------ | --------------------------------- | ---------------------------------------------- | ---- |
| 0    | 原始数据与仓库审计             | Inventory + Profiles              | ✅ 已完成                                      | ☑   |
| 1    | 五个 Adapter 与统一转换        | Processed 数据 + Tests + Previews | ✅ 已完成                                      | ☑   |
| 2    | 数据统计与选择                 | Statistics + Selection Report     | ✅ 已通过（2026-11-10，选定 R-LiViT + MID-3K） | ☑   |
| 3    | 可复现 YOLO 基线               | RGB/T/D + 简单融合主表            | 是（确定 MoE 必要性）                          | ☐    |
| 4    | MoE 与 Reliability 逐步研发    | 完整消融 + 容量对照               | 是（确认实际贡献）                             | ☐    |
| 5    | 扩展数据集/跨 YOLO 版本/鲁棒性 | 完整论文结果矩阵                  | 是（确定投稿定位）                             | ☐    |
| 6    | 论文与复现包                   | Tables, Figures, Methods, Limits  | 最终检查                                       | ☐    |

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
