# YOLOv9 多模态目标检测与实例分割实现交接

## 目标与总览

本实现以 YOLOv9-C 分割模型为基础，完成两个空间对齐视觉模态的目标检测和实例分割。输入不是两张独立文件，而是一张六通道图像：前 3 个通道为模态 A，后 3 个通道为模态 B。网络先分别提取两路特征，在 P3/P4/P5 尺度融合，再以主分支输出部署结果，并通过 YOLOv9 的可逆辅助分支（PGI）提供额外训练监督。

```text
H×W×6 多通道文件
        │
        ├── [:, :, :3] ── 模态 A 骨干 ─┐
        └── [:, :, 3:] ─ 模态 B 骨干 ─┼─ Concat(P3/P4/P5) ─ 主颈部 ─ P3/P4/P5 ─┐
                                      │                                           ├─ DualDSegment ─ 部署输出
                                      └─ 路由特征 + 6ch 辅助骨干 ─ A3/A4/A5 ───────┘
```

涉及的 YOLOv9 实现文件：

- `models/segment/yolov9-c-dseg2.yaml`：网络拓扑。
- `models/common.py`：多模态首层模块 `Conv2B`、`Conv2B1`、`Conv2B2`、`Convm6`。
- `models/yolo.py`：`DualDDetect` 与 `DualDSegment`。
- `utils/segment/dataloaders.py`：六通道输入、同步增强和实例 mask 构造。
- `utils/segment/loss_tal_dual.py`：双头检测/分割损失。
- `segment/train_dual2.py`、`segment/val_dual.py`：训练与训练内验证。

## 数据约定

数据 YAML 指向数据集根目录和 `train.txt`、`val.txt`。文本文件的每一行是一个多通道图像路径；标签路径由图像路径中的 `/images/` 替换为 `/labels/` 后得到。

每个图像必须满足：

```text
图像：H × W × 6
通道 0..2：模态 A，三通道
通道 3..5：模态 B，三通道
```

加载器用 `skimage.io.imread()` 读取原始数组，再进行切分：

```python
image = io.imread(path)
im_a = image[:, :, :3]
im_b = image[:, :, 3:]
```

因此两个模态必须已经完成同名、同尺寸、同像素坐标的配准与打包。代码不在运行时执行文件配对或跨模态配准。

标签采用 YOLO segmentation 格式：

```text
class x1 y1 x2 y2 ... xN yN
```

其中所有顶点坐标均已归一化到 `[0, 1]`。加载时保留 polygon 作为分割真值，同时转换为 `class, x_center, y_center, width, height`，用于检测分配与损失计算。仅有五列 bbox 的检测标签不能提供实例 mask，分割训练应使用 polygon 标签。

## 数据处理与同步增强

`LoadImagesAndLabelsAndMasks2` 在每一个样本上执行以下流程：

1. 以相同缩放比例读取并 resize 两个模态。
2. 训练时随机选择 Mosaic；四张样本的两个模态使用相同拼接位置，标签和 polygon 同步平移。
3. 非 Mosaic 情况下，对两个模态做相同的 `letterbox2`，并把 polygon 和 bbox 映射到带 padding 的坐标系。
4. 用同一透视/仿射矩阵对两个模态、bbox 和 polygon 调用 `random_perspective2`。
5. 从变换后的 polygon 栅格化生成实例 mask。默认 `mask_ratio=4` 时，`640×640` 输入保存为 `160×160` 真值 mask。
6. 两个模态都做 HSV 颜色增强；上下/左右翻转的随机决定共用，bbox 和 mask 随之翻转。
7. 模态 A 转为 CHW 后做通道反转，模态 B 转为 CHW；二者沿 channel 维拼接，得到 `(6, H, W)`。

batch 的 collate 输出为：

```text
images : (B, 6, H, W), uint8
targets: (N, 6) = [batch_index, class, x, y, w, h]
masks  : overlap=True 时 (B, Hm, Wm)，否则为按实例拼接的 mask
paths, shapes
```

默认使用 overlap mask：每张图的一张 mask 用 `1..n` 区分实例，并按面积从大到小排序；TaskAlignedAssigner 返回的 GT 索引用于在损失阶段恢复对应二值实例 mask。若使用 `--no-overlap`，则保留每个实例的一张二值 mask。

## 网络搭建

训练应以六通道构建模型：

```python
model = SegmentationModel("models/segment/yolov9-c-dseg2.yaml", ch=6, nc=nc)
```

`Conv2B` 是模态拆分入口。它将六通道张量切为两个三通道张量，并以独立 `3→64` 卷积产生两路首层特征。`Conv2B1` 只取第一路继续建立模态 A 的 GELAN 金字塔，`Conv2B2` 从同一首层元组中取第二路建立模态 B 的 GELAN 金字塔。

在 P3、P4、P5 上，两个模态特征分别 concat，形成融合骨干特征。标准 YOLOv9 颈部在这些融合特征上产生主检测尺度 `P3/P4/P5`。

与此同时，`CBLinear` 从融合骨干路由多尺度信息；`Convm6` 直接从原始六通道输入开始另建辅助金字塔，并以 `CBFuse` 融合路由特征，形成 `A3/A4/A5`。这是训练期的 YOLOv9 PGI 辅助分支。

最终层定义为：

```yaml
[[A3, A4, A5, P3, P4, P5, proto_A_feature, proto_P_feature], 1, DualDSegment, [nc, 32, 256]]
```

`DualDSegment` 包含：

- 两套独立的三尺度 DFL 检测头：辅助 A 头和融合主 P 头；
- 两套三尺度、每位置 32 维的 mask coefficient；
- 两个 32 通道 prototype 图，分别服务辅助头和主头。

`npr=256` 在该 `DualDSegment` 实现中只被保存为属性；prototype 实际由 1×1 卷积直接生成 32 个通道，不使用标准 `Proto(c_, npr, nm)` 的中间 256 通道结构。

## 训练、损失与验证

训练循环将输入从 `uint8` 转换为 `[0, 1]` float，随后调用：

```python
pred = model(images)
loss, loss_items = ComputeLoss(model, overlap=True)(pred, targets, masks)
```

训练态的 `DualDSegment` 输出为两套：

```text
detect features: [auxiliary features, main features]
mask coefficients: [auxiliary coefficients, main coefficients]
prototypes: [auxiliary prototype, main prototype]
```

`ComputeLoss` 用两套独立的 TaskAlignedAssigner 将同一组 GT 分别分配给两个检测头。每套均计算：

- 分类 BCE/Focal loss；
- CIoU box loss；
- Distribution Focal Loss（DFL）；
- 实例 mask loss。

分割预测的重建方式为 `coefficient @ prototype`：32 维 mask coefficient 与 `(32, Hm, Wm)` prototype 相乘得到每个正样本的 mask logits。与 GT mask 做逐像素 BCE 后，损失只在对应预测框内统计，并除以归一化框面积。

辅助头的 box、DFL 和 segmentation 分量先乘 `0.25`；主头承担完整监督。最终使用的损失增益为 box `7.5`、segmentation `2.5 / batch_size`、classification `0.5`、DFL `1.5`。

训练内验证复用六通道 dataloader。验证和部署输出只使用融合主头：`DualDSegment.forward()` 在 eval 状态返回主头的 box/class、主头 32 维 coefficients 和主 prototype。NMS 后用 mask coefficient 与主 prototype 重建实例 mask，分别计算 box 与 mask mAP。

## 推荐训练命令

```bash
python segment/train_dual2.py ^
  --data data/signboard_rgbtc_0521.yaml ^
  --cfg models/segment/yolov9-c-dseg2.yaml ^
  --weights ckpt/yolov9-c-seg.pt ^
  --img 640 --batch 4 --epochs 500 ^
  --mask-ratio 4 --close-mosaic 15
```

现有训练脚本只在加载 `.pt` 权重的分支中明确传入 `ch=6`。如果希望从头训练，需要同步修正配置分支，使其同样以 `ch=6` 构建模型；否则模型 stride 初始化会使用三通道 dummy input，无法通过多模态入口。

## 运行边界与后续修正项

- 单独运行 `segment/predict.py` 仍使用通用三通道 `LoadImages`，不能直接输入六通道模型；应实现对应的多模态推理加载器，并将 warmup shape 改为 `(B, 6, H, W)`。
- 单独运行 `segment/val_dual.py` 的非训练分支应使用 `create_dataloader2`，且 warmup 同样应使用六通道；训练脚本传入 dataloader 的验证路径不受此影响。
- 不应开启 `--cache`：继承的基础缓存仅保存一个图像数组，不能正确恢复第二模态。
- Mosaic 中当前对两个模态分别调用 `copy_paste`，会产生独立随机选择，可能破坏模态像素级对齐。应只抽样一次复制实例/掩码，再将同一空间操作应用到两路图像。
- `Albumentations2` 对两路图像分别执行随机外观增强；若加入几何类 Albumentations，必须确保两个模态、bbox、polygon 和 mask 共用同一随机变换参数。

## YOLOv11-RGBT 参考实现与差异分析

以下分析基于 `E:\repository\YOLOv11-RGBT`。该仓库并非一个固定的 RGBT 网络，而是把多光谱支持做进 Ultralytics YOLO11 数据、训练、验证和推理链路；`use_simotm`、模型 YAML 的 `ch` 和训练参数 `channels` 共同决定输入形式与融合拓扑。

### 输入、配对和数据集

默认 RGBT 模式是 `use_simotm="RGBT"`、`channels=4`、模型 YAML 中 `ch: 4`：

```text
visible/<name>.<ext>  ── cv2 BGR ─┐
                                  ├─ merge(B, G, R, IR-gray) ── H×W×4
infrared/<name>.<ext> ─ gray ────┘
```

数据 YAML 或 TXT 只列出可见光路径。`BaseDataset.load_and_preprocess_image()` 用
`file_path.replace(pairs_rgb, pairs_ir)` 寻找配对红外文件；默认名称为
`pairs_rgb_ir=['visible', 'infrared']`，可由参数改为其他目录名。标签仍由可见光主路径的常规
YOLO 规则读取，因而两个模态必须同名、像素对齐并共享标注。

它还支持 `use_simotm="RGBRGB6C"`：可见光和红外都以 BGR 读取，运行时合成为
`B,G,R,B2,G2,R2` 六通道。对单通道、16 位和任意通道多光谱图像，仓库还定义了 `Gray`、
`Gray16bit`、`Multispectral`、`Multispectral_16bit` 等模式。

与本项目原 YOLOv9 实现不同，YOLOv11-RGBT 的原始数据是两份独立文件，不要求预先写成
六通道 TIFF/PNG。融合后的数组才进入 dataset；它会以 `_RGBT.npy` 或 `_RGBRGB6C.npy` 后缀缓存，
所以 RAM/disk cache 保存的是完整多模态样本，而不是只保存其中一个模态。

### 同步预处理、训练与推理

1. `DetectionTrainer` 将 `use_simotm` 与 `pairs_rgb_ir` 传给 `build_yolo_dataset()`；batch 预处理仍是通用的
   `float()/255`，因此得到的 tensor 是 `(B, 4, H, W)` 或 `(B, 6, H, W)`。
2. 两幅图先合并为一个 HWC 多通道数组，再进入 YOLO11 的 Mosaic、MixUp、CopyPaste、LetterBox、
   RandomPerspective 和翻转链路。因此同一次空间变换天然作用于两个模态和同一组标签。
3. `RandomPerspective` 对四通道数组直接变换；对六通道数组显式将 `:3` 与 `3:` 分开调用同一矩阵，
   再 concat，避免 OpenCV 对六通道 warp 的限制。
4. 4C/6C 分别使用 `RandomHSV4C` / `RandomHSV6C`：前 3 通道走 BGR HSV，6C 的后 3 通道也独立走 HSV。
   因此 RGBT 的红外灰度不会被当作彩色图像增强，而 RGBRGB6C 的第二路会接受彩色外观增强。
5. `BasePredictor` 把同样的 `use_simotm` 与配对名称传给 `LoadImagesAndVideos`；图像和视频都能在推理时
   自动打开对应的 IR 文件/视频并合成为 4C 或 6C 输入。这使训练、验证和独立推理使用同一套配对逻辑。

注意：`_resize_images()` 只在两个模态原始尺寸不同的情况下分别 resize。它并不执行相机标定或重投影，
仍把像素级对齐作为数据前提；若两者宽高比不一致，不能仅靠该代码保证正确配准。

### 结构选择：配置驱动，而不是单一网络

`ultralytics/cfg/models/11-RGBT/` 提供多个网络 YAML。通用切分算子是
`SilenceChannel(c_start, c_end)`，它对 NCHW 张量做通道切片。

| 方案                        | 代码中的拓扑                                                                                              | 适用含义                                                                                           |
| --------------------------- | --------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| earlyfusion                 | `SilenceChannel[0,4]` 后直接送入一个常规 YOLO11 骨干                                                      | 首层直接学习 RGB 与 IR 的联合卷积。                                                                |
| midfusion（基础 RGBT 方案） | RGB 与 IR 各自完整 C3k2 骨干；P3/P4/P5 concat 后共享 SPPF、C2PSA、PAN 和一个 `Detect`/`Segment` 头        | 先模态专用编码，再多尺度特征融合。                                                                 |
| mid-to-late                 | 两支先各自到 P3，再在 P3/P4/P5 融合并共享后续下采样/检测颈部                                              | 融合发生在中后段。                                                                                 |
| latefusion                  | 两支各自跑完整骨干和 PAN；最终 concat 同尺度检测特征，再接一个检测头                                      | 融合最晚，计算量最大。                                                                             |
| scorefusion                 | 两支各自跑完整网络，六张尺度特征同时送入通用 `Detect`                                                     | 实际代码是把六组 anchors/predictions一起 decode，再由后处理 NMS 汇总；没有额外的显式分数加权模块。 |
| share                       | `ChannelToNumber` 把 RGB 与 IR 扩到 batch 维，经过同一套共享权重的骨干，再由 `NumberToChannel` 合回通道维 | 参数最省，但假设两模态可共享表征。                                                                 |
| midfusion-P3-PGI            | 融合主分支之外，用 CBLinear/CBFuse 生成辅助分支并接 `DetectAux`                                           | 可逆辅助监督的检测版。                                                                             |
| midfusion-MCF               | 可见光侧通过零初始化 `ZeroConv2d` 注入红外主分支，并以 ADD 逐尺度相加                                     | ControlNet 风格的可控跨模态注入，需要专门的预训练/权重转换流程。                                   |

基础 `yolo11-RGBT-midfusion.yaml` 的 RGBT 通道划分是 `[0:3]` 的 visible 和 `[3:4]` 的 IR，
两支各自产生 P3/P4/P5 后 concat；融合后只有一个标准 YOLO11 检测头。分割版
`yolo11-RGBT-midfusion-seg.yaml` 仅将最后一层替换为标准 `Segment[nc, 32, 256]`，并不创建第二个
prototype 或第二个 mask-coefficient 头。

PGI 变体 `yolo11-RGBT-midfusion-P3-PGI.yaml` 才使用双输出 `DetectAux`：前三个尺度是主分支、后三个尺度是辅助分支。
训练时两支均计算分类、box 和 DFL；辅助损失系数为 `0.25`。推理时 `DetectAux` 只 decode 并返回前三个主分支尺度，
可通过 `switch_to_deploy()` 删除辅助头。

### 与原 YOLOv9 多模态实例分割实现的区别

| 维度           | 原 YOLOv9 `yolov9-c-dseg2`                                                             | YOLOv11-RGBT 基础 midfusion                                                                     | 影响                                                                                             |
| -------------- | -------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 默认输入       | 一个预合成的 H×W×6 文件，两个三通道模态                                                | 两个文件在运行时配对；默认是 RGB(3)+IR(gray,1)=4C，也可选 RGBRGB6C                              | YOLO11 更易复用常见 RGBT 数据集；YOLOv9 适合已打包的多通道传感器样本。                           |
| 样本配对       | 没有运行时配对，固定按 `:3` / `3:` 切分                                                | 由 `visible→infrared` 的字符串替换配对，名称可配置                                              | YOLO11 更灵活，但依赖目录/文件命名严格一致。                                                     |
| 空间增强       | 两张数组分别处理；letterbox、透视和翻转显式同步                                        | 合并后走统一增广；6C 透视时仍确保两半共用同一矩阵                                               | YOLO11 的几何同步更集中，且避免了 YOLOv9 Mosaic 中两次独立 `copy_paste` 随机抽样造成的潜在错位。 |
| 缓存           | 基础缓存未保存第二模态，不能可靠开启 `--cache`                                         | 缓存合成后的 4C/6C NPY，支持 RAM/disk cache                                                     | YOLO11 的大规模训练数据管线更完整。                                                              |
| 主网络         | 自定义 `Conv2B/1/2` 双流 GELAN；P3/P4/P5 concat 后为主颈部                             | 标准 YOLO11 双 C3k2 流，融合后走 SPPF、C2PSA 和 PAN                                             | 两者都是中期多尺度 concat，但骨干与颈部世代不同。                                                |
| PGI/辅助分支   | `dseg2` 固定包含辅助 A3/A4/A5；训练即双头                                              | 基础 midfusion 无辅助支路；仅选用 `*-PGI.yaml` 才有 `DetectAux`                                 | YOLOv9 的辅助监督是该实例分割结构的一部分；YOLO11 将其作为可选实验拓扑。                         |
| 分割头         | `DualDSegment` 有两套检测头、两套 32D coefficients、两张 prototype；两支都算 mask loss | `midfusion-seg` 是一套标准 `Segment`，仅使用融合特征与一张 prototype                            | YOLOv9 直接对辅助分支施加实例分割深监督；YOLO11 基础方案只有融合主头的分割监督。                 |
| 部署输出       | `DualDSegment` 显式只返回主 P 分支结果                                                 | 基础 midfusion 从始至终只有一个头；PGI 的 `DetectAux` 推理时只返回主头                          | 两者部署模型都是单输出，但 YOLO11 基础结构更简单。                                               |
| 损失           | 自定义双 TaskAlignedAssigner；辅助 box/DFL/seg 先乘 0.25，主支全量监督                 | 标准 YOLO11 TAL + BCE/box/DFL；只有 `DetectAux` 才增加 0.25 辅助检测损失                        | YOLOv9 的辅助分支覆盖检测和分割；YOLO11 PGI 变体的辅助监督是检测级。                             |
| 任务与推理闭环 | 训练内验证支持六通道，但独立 `predict.py` / `val_dual.py` 仍沿用三通道路径             | detect、segment、pose 的 trainer/validator/predictor 都透传多模态参数；图像和视频均支持配对加载 | YOLO11 的工程闭环更适合直接训练、验证、部署。                                                    |
| 从头训练       | `train_dual2.py` 的无权重分支错误地按 `ch=3` 建模                                      | `DetectionModel` 从 YAML 的 `ch` 建模，4C/6C 配置可直接从头初始化                               | YOLO11 对输入通道所有权更清晰。                                                                  |

结论：两者的共同核心是“模态专用编码后，在多尺度特征层融合”。原 YOLOv9 实现更强调
PGI 双头和双分割监督；YOLOv11-RGBT 更强调把配对读取、通道模式、缓存、训练/验证/推理与多种融合
YAML 统一为可配置框架。若迁移你的任务，优先保留 YOLO11 的运行时配对与统一增广/推理管线；若要保留
YOLOv9 的训练优势，则需要在 YOLO11 的 `Segment` 路径中扩展 `DetectAux` 对应的双 mask coefficient、双 prototype
和辅助 segmentation loss，而不能只套用检测版 `DetectAux`。

## 下一阶段：统一多模态融合框架设计

### 可行性与现状

该设计可行。`E:\repository\YOLOv11-RGBT` 已分别用 YAML 分支实现 early、mid、mid-to-late、late、score 和
weight sharing；其中 `ChannelToNumber` / `NumberToChannel` 通过把两路样本折叠到 batch 维后复用同一套模块，证明
“同一模块实例被多模态重复调用”的共享权重路线可行。

本项目当前已具备任意路像素对齐模态的读取、`ModalSplit`、YAML 分支和标准检测/实例分割训练闭环；现有的
三路 YOLO11 示例属于 backbone 输出处的 `Concat` 融合。下面的六种策略、`Add`、融合模块注册、阶段化共享和
score fusion 均是后续实现范围，不能误称为当前已经支持的功能。

### 统一阶段定义

为避免不同 YOLO 世代把同一层称作 backbone 或 neck，融合 YAML 必须显式声明边界，而不是在代码中根据模型名
猜测层号。本文采用如下术语：

```text
input → encoder → nape → (backbone 输出) → neck → head feature → Detect / Segment
```

- `nape` 是 **backbone 尾部** 的语义提炼部分；它不同于 head 中的 `neck`。这一命名沿用本设计，避免把两个
  位置混为一谈。
- 模板的默认划分是：YOLOv8、YOLOv9、YOLO26 的最后一个 backbone 层为 `nape`；YOLOv10、YOLO11 的最后两个
  backbone 层为 `nape`；YOLOv12 没有 `nape`，整个 backbone 都是 encoder。
- P2/P6 等变体可以改变层数和多尺度输出，因此上述只是模板约定。每个 YAML 都必须提供自己的
  `encoder_end`、`nape_end` 和各尺度输出锚点，不能把该约定写死为版本判断。
- 本文中的 `neck` 仅指 head 中 `Detect` / `Segment` 等最终预测层之前的 PAN/FPN 路径。

所有融合点的多路特征必须具有相同 batch、空间尺度和语义 stride。`Concat` 在通道维拼接后需要显式投影层
（通常为 `1×1 Conv`）；`Add` 在相加前必须用 adapter 将通道数对齐。框架不应静默 resize、截断或广播特征。

### 六种训练期融合策略

| 代码  | 名称              | 拓扑与融合位置                                                                                      | 基线算子与训练含义                                                 |
| ----- | ----------------- | --------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| `IF`  | `input_fusion`    | 各模态通道直接堆叠，送入一条 encoder。                                                              | 只有一个分支；首层联合学习全部传感器，复用原生损失。               |
| `EF`  | `encoder_fusion`  | 每模态独立运行 encoder；在 encoder 输出处逐尺度融合，再进入共享 nape、neck 和预测头。               | `Concat + 1×1 Conv` 或对齐后的 `Add`；适合较早交换模态信息。       |
| `NIF` | `nape_in_fusion`  | 独立 encoder 将特征送入 nape；融合模块插入 nape 的内部结点，而不是只在其输入/输出处融合。           | 需要专用 `NapeFusionBlock`；适合注意力、门控、交叉调制等复杂融合。 |
| `BF`  | `backbone_fusion` | 每模态完整运行 encoder+nape；在 backbone 的 P3/P4/P5 等输出处逐尺度融合，再进入共享 neck 和预测头。 | 当前三路示例的扩展形式，是首个应稳定的双/多模态基线。              |
| `NF`  | `neck_fusion`     | 每模态独立运行完整 backbone 和 neck；在 Detect 前的同尺度 neck 输出处融合。                         | 模态专用表征最深、算力最高；融合后使用一个共享预测头。             |
| `HF`  | `head_fusion`     | 每模态独立运行到 head 的原始预测特征或 logits；在 decode/NMS 前融合。                               | 必须定义与融合输出相容的预测投影和损失；不能直接融合 `Results`。   |

`HF` 不等同于 score fusion。score fusion 保留多个独立预测头，分别 decode 和 NMS 后再合并候选框/类别分数；
它是部署期后处理策略，默认不参与反向传播、不能复用单头训练损失，也需要单独的导出和评估路径。若希望把
多头 raw logits 在训练期融合，应归入 `HF`，并实现相应的 loss contract。

### 融合模块接口

第一阶段只提供 `concat` 与 `add` 两种稳定基线；融合点的 YAML 不应散落手写 `Concat` 和通道投影，而应通过
统一配置解析为对应模块。例如：

```yaml
multimodal:
  fusion: BF
  operator: concat # concat 或 add
  fusion_points: [P3, P4, P5]
  share_weight: false
```

自定义融合模块通过注册表暴露，统一满足 `forward(features: list[Tensor]) -> Tensor`。注册项必须声明支持的模态数、
输入/输出通道、每尺度独立还是跨尺度处理、是否可导出，以及是否需要额外损失。`NIF` 使用同一注册机制，但其模块
应接收 nape 内部指定结点的特征；不得在通用 dataloader 或训练循环中增加针对某个模型的特殊分支。

### `share_weight` 语义

增加 YAML 参数 `share_weight: bool`。当其为 `true` 时，融合点之前由 `shared_stages` 指定的完整阶段复用**同一
模块实例**，不是复制权重后周期性同步：

```yaml
multimodal:
  fusion: BF
  operator: concat
  share_weight: true
  shared_stages: [encoder, nape] # 省略时默认共享融合点之前的全部完整阶段
```

- 实现应泛化参考仓库的固定 RGB(3)+IR(1) `ChannelToNumber`：先由每模态 `input_adapter` 投影至共同通道数，
  将 `(B, M, C, H, W)` 折叠为 `(B×M, C, H, W)`，经过共享模块后再按原模态顺序展开。这样可支持任意模态数与
  不等输入通道数；adapter 本身只有在形状兼容时才能共享。
- `share_weight: true` 对 `IF` 无意义，应在配置校验时拒绝，不能静默忽略。对于 `EF`/`NIF`/`BF`/`NF`/`HF`，共享范围
  只能是融合点之前的完整阶段；跨过融合点共享会改变该策略的语义。
- 带 BatchNorm 的共享模块必须在折叠后的 `B×M` batch 上更新统计量。这会混合模态分布，应作为明确的实验选择；
  如需模态独立统计量，应使用独立归一化层或冻结统计量，而不是伪造“部分共享”。

### 训练、部署与验证边界

- `IF`、`EF`、`NIF`、`BF`、`NF` 在融合后接一个标准预测头时可以复用现有检测/分割损失。`HF` 和 score fusion
  需要各自的 loss 或后处理实现，不能仅靠 YAML 改线。
- 所有特征级融合都要求同尺度特征严格对齐；数据侧仍以前述像素对齐和统一几何增强为前提。
- 每个新增模板至少验证：模型构建、`(B, total_channels, H, W)` 前向、P3/P4/P5 stride、concat/add 通道检查、
  单 batch loss、独立 val/predict 和导出。score fusion 另需验证多头 NMS/类别合并与每模态预测的可追溯性。
- 实现顺序应为：`IF`/`BF` 的 concat 基线 → add 与通用融合注册表 → `EF`/`NF` → 泛化权重共享 → `NIF` → `HF` 和
  部署期 score fusion。这样先固定所有权和特征边界，再引入需要自定义训练/导出契约的复杂策略。
