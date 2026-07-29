# 广告牌缺陷检测与重识别论文实验计划

## 1. 项目目标

本项目目标是构建一套可用于论文发表的广告牌缺陷识别与重识别实验流程，替换此前因版权限制无法公开或复用的数据。新的实验应基于实验室自有数据完成数据整理、标注、伪标签生成、模型训练、推理部署和论文实验分析。

核心研究方向：

- 基于 Ultralytics YOLO 的属性目标检测，用于广告牌缺陷识别。
- 基于 FastReID 的广告牌重识别，用于跨视角、跨帧或跨采集段的广告牌身份关联。
- 基于 SAM3 或可用 SAM 系列模型的广告牌实例分割，用于快速构建广告牌区域标注。
- 基于历史数据训练缺陷分类器，为实验室数据生成缺陷伪标签。
- 复用并改造已有 ONNX 部署推理流程，形成检测、分类、重识别一体化 pipeline。

主要代码与数据参考：

- 属性检测代码库：`E:\repository\ultralytics`
- 重识别代码库：`E:\repository\fast-reid`
- 旧部署推理参考：`\\158.132.186.40\isds\huilin\isds\back up\cdu_seg_risk6_multiple_head1115\cdu_mount\onnx_depolyment.py`
- 新实验室数据格式参考：`\\10.22.50.44\individualdata\VMMS\2024-09-19_HyD_collect\Riegl\central_1\ladybug`
- 旧数据与历史标注参考：`\\158.132.186.40\isds\huilin\isds\back up\final_data`

## 2. 论文预期贡献

论文应避免只描述工程系统，建议将贡献收束为以下 3 到 4 点：

1. 提出面向移动测量数据的广告牌缺陷识别流程，整合广告牌定位、区域分割、缺陷属性识别和跨视角重识别。
2. 构建基于实验室自有数据的广告牌缺陷数据集，包括广告牌实例分割、缺陷类别或风险属性伪标签，以及用于重识别的广告牌身份标注。
3. 设计一种属性目标检测建模方式，在目标检测输出中同时预测广告牌位置和缺陷属性，相比检测后分类或普通检测模型提供更直接的风险识别能力。
4. 验证检测、分割、分类和重识别模块在真实移动测量场景中的效果，并分析伪标签质量、标注成本和部署效率。

## 3. 数据准备计划

### 3.1 新数据接入

输入数据以实验室数据为主，格式参考：

`\\10.22.50.44\individualdata\VMMS\2024-09-19_HyD_collect\Riegl\central_1\ladybug`

需要确认并记录：

- 图像来源：Ladybug 全景图、透视展开图或单相机视角图。
- 图像命名规则：时间戳、帧号、相机编号、路线编号是否可解析。
- 位姿或轨迹信息：是否包含 GPS/IMU/点云同步信息。
- 图像质量：模糊、过曝、遮挡、夜间、雨天等比例。
- 广告牌分布：广告牌数量、尺寸、密度、重复出现频率。
- 可公开性：确认实验室数据是否允许论文中展示样例图和统计信息。

产出：

- `data_inventory.md`：记录数据来源、数量、字段、版权和使用限制。
- `raw_data_index.csv`：每张图像的路径、帧号、采集段、相机视角、是否可用。

### 3.2 数据清洗

建议先抽取一个小规模可控子集，再扩展到完整数据：

- Pilot set：约 300 到 500 张图，用于验证标注流程和模型 pipeline。
- Train/val/test set：按采集路线或空间区域划分，避免相邻帧泄漏。
- ReID set：从多帧中选择重复出现的广告牌实例，建立 identity 标签。

划分原则：

- 不按随机图片划分，而按路线、街区、时间段或空间段划分。
- 测试集应包含未参与伪标签调参的区域。
- ReID 的 query/gallery 应避免同帧或近重复图像造成虚高结果。

## 4. 标注计划

### 4.1 广告牌分割标注

目标：得到广告牌实例级 mask 或 polygon，用于训练/评估分割模型，也可作为检测框来源。

流程：

1. 使用 SAM3 或当前可用 SAM 系列模型生成广告牌候选 mask。
2. 对 mask 自动转 polygon 或 bounding box。
3. 人工审核并修正错误 mask。
4. 将最终标注转换为 YOLO segmentation 格式。

重点检查：

- 一个广告牌被切成多个 mask。
- 多个相邻广告牌被合并成一个 mask。
- 玻璃、海报、店招、交通牌与广告牌混淆。
- 遮挡或倾斜广告牌的边界质量。

产出：

- `labels_seg/`：YOLO segmentation 标签。
- `billboard_seg_dataset.yaml`：数据集配置。
- `annotation_guideline.md`：广告牌定义、边界规则、忽略规则和典型案例。

### 4.2 缺陷属性标注与伪标签

由于人工完整标注缺陷成本较高，建议采用“旧数据训练分类器 + 新数据伪标签 + 人工抽检修正”的策略。

历史数据参考：

`\\158.132.186.40\isds\huilin\isds\back up\final_data`

流程：

1. 整理历史缺陷类别或风险属性定义。
2. 使用历史数据训练广告牌缺陷分类器。
3. 对新实验室数据中的广告牌 crop 进行分类推理。
4. 输出每个广告牌实例的缺陷属性伪标签和置信度。
5. 按类别、置信度和场景抽样人工核查。
6. 将高置信样本纳入训练集，将低置信样本放入待审池。

建议缺陷属性体系：

- structural damage：破损、变形、缺失。
- surface degradation：褪色、污渍、腐蚀。
- installation risk：倾斜、松动、悬挂异常。
- occlusion or visibility issue：遮挡、严重模糊、不可辨识。
- normal：无明显缺陷。

最终类别以历史数据实际标注为准，论文中必须给出清晰定义和样例。

产出：

- `defect_taxonomy.md`：缺陷类别定义。
- `pseudo_labels.csv`：图像、广告牌 ID、bbox/mask、缺陷类别、属性向量、置信度。
- `pseudo_label_audit.md`：人工抽检比例、准确率、常见错误。

### 4.3 ReID 标注

目标：为同一广告牌在多帧、多视角或多路线中的重复出现建立 identity。

流程：

1. 基于分割结果裁剪广告牌 crop。
2. 用已有 FastReID 模型或视觉特征进行初始聚类。
3. 人工确认 cluster，将同一广告牌分配相同 ID。
4. 构建 query/gallery 划分。

建议记录字段：

- `instance_id`
- `identity_id`
- `image_path`
- `bbox`
- `mask_path`
- `route_id`
- `frame_id`
- `camera_id`
- `quality_flag`

产出：

- `reid_train/`
- `reid_query/`
- `reid_gallery/`
- `reid_annotations.csv`

## 5. 模型训练计划

### 5.1 广告牌分割模型

目的：

- 提供广告牌区域。
- 辅助 crop 缺陷分类。
- 为检测框和 ReID crop 提供更干净的目标区域。

候选训练：

- YOLO segmentation baseline。
- 使用 SAM 生成的伪 mask 训练轻量分割模型。
- 对比 bbox detection 与 segmentation-derived bbox 的差异。

指标：

- mask mAP。
- box mAP。
- boundary quality qualitative examples。
- annotation time saving。

### 5.2 缺陷分类器

目的：

- 利用旧数据训练缺陷分类模型。
- 为新数据生成缺陷伪标签。

训练输入：

- 历史数据广告牌 crop。
- 历史缺陷类别标签。
- 必要时加入正常样本和 hard negative。

评估：

- classification accuracy。
- macro F1。
- per-class precision/recall。
- confusion matrix。
- 对新数据抽检集的 pseudo-label accuracy。

### 5.3 属性目标检测模型

当前 Ultralytics 项目中已有 `mdetect` / `msegment` 分支，可继续作为核心模型。

训练目标：

- 输入图像。
- 输出广告牌 bbox 或 mask。
- 同时输出缺陷属性向量。

需要整理：

- `nc`：广告牌主类别数量。
- `na`：属性数量。
- `nal`：属性层级或属性组数量。
- `names`：目标类别名。
- `attributes`：缺陷属性名。

对比实验：

1. YOLO detection + 后处理分类器。
2. YOLO segmentation + 后处理分类器。
3. 属性目标检测 `mdetect`。
4. 属性实例分割 `msegment`。

主要指标：

- box mAP。
- mask mAP。
- attribute accuracy。
- attribute macro F1。
- end-to-end defect recognition F1。
- inference latency。

### 5.4 广告牌 ReID 模型

代码库：

`E:\repository\fast-reid`

训练数据：

- 广告牌 crop。
- identity label。
- query/gallery split。

评估指标：

- Rank-1。
- Rank-5。
- mAP。
- 跨视角、跨距离、遮挡情况下的召回率。

实验设置：

- Baseline FastReID。
- 使用 mask crop vs bbox crop。
- 使用缺陷属性辅助检索的重排序策略，作为可选实验。

## 6. 推理与部署计划

参考旧部署脚本：

`\\158.132.186.40\isds\huilin\isds\back up\cdu_seg_risk6_multiple_head1115\cdu_mount\onnx_depolyment.py`

目标是形成新的可复现实验推理流程：

1. 输入实验室图像或视频帧。
2. 广告牌检测/分割。
3. 广告牌 crop 提取。
4. 缺陷属性识别。
5. ReID 特征提取。
6. 同一广告牌跨帧合并。
7. 输出结构化结果。

建议输出格式：

```text
image_id
instance_id
identity_id
bbox
mask_path
defect_attributes
defect_confidence
reid_feature_path
track_or_match_score
quality_flag
```

部署评估：

- 单张图像平均推理时间。
- 每个模块耗时占比。
- ONNX 与 PyTorch 结果差异。
- 批量处理稳定性。
- 失败样例记录。

## 7. 实验设计

### 7.1 核心研究问题

候选研究问题：

RQ1：属性目标检测是否比“广告牌检测 + 缺陷分类”的级联系统更适合广告牌缺陷识别？

RQ2：SAM 辅助标注和伪标签策略能否在减少人工标注成本的同时保持可接受的缺陷识别性能？

RQ3：广告牌 ReID 是否能提升移动测量场景下的缺陷记录一致性，减少重复统计和漏检？

### 7.2 主要实验

Experiment A：广告牌检测/分割性能

- 比较普通检测、分割模型、SAM 辅助标注训练模型。
- 输出 box/mask mAP 和典型可视化结果。

Experiment B：缺陷伪标签质量

- 历史数据训练分类器。
- 在新数据上抽样人工审核。
- 报告 pseudo-label precision、recall 或 accuracy。

Experiment C：属性目标检测性能

- 比较 cascade 方法与 `mdetect/msegment`。
- 报告检测性能、属性识别性能和 end-to-end 缺陷识别性能。

Experiment D：广告牌重识别性能

- 比较 bbox crop 与 mask crop。
- 报告 Rank-1、Rank-5、mAP。

Experiment E：完整 pipeline 评估

- 从原始图像到最终广告牌缺陷记录。
- 报告漏检、误检、重复记录、处理时间和失败类型。

### 7.3 消融实验

建议消融项：

- 是否使用 SAM 辅助 mask。
- 是否使用伪标签。
- 伪标签置信度阈值。
- bbox crop vs mask crop。
- detection + classifier vs attribute detection。
- 是否使用 ReID 合并重复广告牌。

## 8. 论文结构建议

暂定论文结构：

1. Introduction
   - 广告牌缺陷检测的城市安全意义。
   - 移动测量数据中的挑战：尺度变化、遮挡、视角变化、重复观测、标注成本。
   - 本文贡献。

2. Related Work
   - Billboard or traffic asset inspection。
   - Object detection and instance segmentation。
   - Defect recognition and attribute detection。
   - Pseudo-labeling and foundation-model-assisted annotation。
   - Object re-identification。

3. Dataset and Annotation
   - 实验室数据来源。
   - 数据清洗与划分。
   - SAM 辅助标注。
   - 缺陷类别体系。
   - 伪标签生成与质量控制。
   - ReID identity 标注。

4. Methodology
   - Overall pipeline。
   - Billboard segmentation/detection。
   - Attribute-based defect detection。
   - Defect classifier for pseudo-labeling。
   - Billboard ReID。
   - ONNX deployment workflow。

5. Experiments
   - Experimental setup。
   - Detection/segmentation results。
   - Pseudo-label quality。
   - Attribute detection comparison。
   - ReID results。
   - End-to-end pipeline results。
   - Ablation studies。

6. Discussion
   - 伪标签误差来源。
   - 标注成本与性能权衡。
   - ReID 对重复统计的价值。
   - 泛化限制。
   - 数据版权与可复现性说明。

7. Conclusion
   - 总结方法、结果和未来工作。

## 9. 质量控制与完整性检查

论文实验必须避免以下问题：

- 使用版权受限旧数据作为论文主结果。
- 旧数据和新数据混合后没有明确说明。
- train/val/test 存在相邻帧泄漏。
- 伪标签直接当真值使用但没有人工抽检报告。
- 只报告整体准确率，不报告类别不均衡下的 macro F1。
- ReID query/gallery 中出现近重复帧导致结果虚高。
- ONNX 部署结果与 PyTorch 结果不一致但没有验证。

需要保留的审计材料：

- 数据来源与授权说明。
- 标注规范。
- 数据划分脚本和随机种子。
- 模型配置 YAML。
- 训练命令。
- 推理命令。
- 评估脚本。
- 失败案例图。
- 伪标签抽检记录。

## 10. 阶段计划

### Phase 0：项目整理

目标：明确代码、数据、论文范围。

任务：

- 梳理 `E:\repository\ultralytics` 中 `mdetect/msegment` 的训练入口。
- 梳理 `E:\repository\fast-reid` 的训练和评估入口。
- 复制或重构旧 `onnx_depolyment.py` 的推理逻辑。
- 建立新实验目录结构。

产出：

- `project_structure.md`
- `experiment_config_template.yaml`
- `runbook.md`

### Phase 1：数据盘点与 pilot set

目标：完成小规模数据闭环。

任务：

- 从实验室数据中选取 300 到 500 张图。
- 建立图像索引。
- 使用 SAM3/SAM 生成广告牌 mask。
- 人工修正一批标注。
- 生成 YOLO segmentation 标签。

产出：

- Pilot segmentation dataset。
- Pilot annotation guideline。
- 初版数据统计。

### Phase 2：历史数据分类器与伪标签

目标：训练缺陷分类器并生成新数据伪标签。

任务：

- 整理历史数据缺陷类别。
- 训练缺陷分类器。
- 对 pilot set 和扩展数据生成伪标签。
- 抽样人工审核。

产出：

- defect classifier checkpoint。
- pseudo-label file。
- pseudo-label audit report。

### Phase 3：属性检测/分割训练

目标：重新训练核心网络。

任务：

- 准备 `mdetect/msegment` 所需标签格式。
- 编写数据集 YAML。
- 训练 baseline 和属性模型。
- 评估 box/mask/attribute 指标。

产出：

- trained weights。
- metrics.csv。
- confusion matrix。
- qualitative visualization。

### Phase 4：ReID 数据与模型

目标：完成广告牌重识别实验。

任务：

- 生成广告牌 crop。
- 标注 identity。
- 训练或微调 FastReID。
- 构建 query/gallery。
- 评估 Rank-k 和 mAP。

产出：

- ReID dataset。
- FastReID checkpoint。
- ReID evaluation report。

### Phase 5：端到端部署推理

目标：形成可复现实验 pipeline。

任务：

- 改造 ONNX 推理脚本。
- 接入检测/分割模型。
- 接入缺陷属性输出。
- 接入 ReID 特征提取和匹配。
- 输出统一结果表。

产出：

- `onnx_deployment_new.py`
- end-to-end results CSV。
- runtime benchmark。
- failure case report。

### Phase 6：论文写作

目标：完成论文初稿。

任务：

- 固定研究问题和贡献。
- 整理实验表格和图。
- 写 Dataset、Method、Experiments。
- 完成完整性检查：数据、代码、指标和引用均可追溯。

产出：

- paper outline。
- paper draft。
- figures and tables。
- reproducibility checklist。

## 11. 建议目录结构

```text
paper_project/
  data_inventory/
    data_inventory.md
    raw_data_index.csv
  annotation/
    annotation_guideline.md
    defect_taxonomy.md
    pseudo_label_audit.md
  datasets/
    billboard_seg/
    billboard_mdet/
    billboard_mseg/
    billboard_reid/
  configs/
    ultralytics/
    fastreid/
  experiments/
    seg_baseline/
    defect_classifier/
    mdetect/
    msegment/
    reid/
    onnx_pipeline/
  results/
    tables/
    figures/
    failure_cases/
  paper/
    outline.md
    draft.md
    references.bib
```

## 12. 当前待确认事项

- 实验室数据是否允许在论文中展示原图或局部 crop。
- SAM3 的具体可用版本、运行环境和授权条件。
- 旧数据 `final_data` 是否可用于训练辅助模型，或只能用于方法参考。
- 缺陷类别是否沿用旧项目的风险 6 类，还是重新定义为论文中的缺陷 taxonomy。
- 新数据是否包含跨帧或跨视角重复广告牌，是否足够支持 ReID 实验。
- 是否需要把点云、GPS 或位姿信息纳入论文方法。
- 目标会议/期刊方向：computer vision、remote sensing、urban infrastructure inspection 或 civil engineering。

