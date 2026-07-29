# Ultralytics 任意模态多模态 YOLO 设计交接

## 目标

在不把任何特定 YOLOv9 或 YOLO11-RGBT 实现当作代码基线的前提下，为本仓库设计支持**任意数量模态、任意正整数
通道布局**的多模态 YOLO 扩展。它必须遵循 Ultralytics 的模型、任务和配置约定，并能持续合入上游更新：对
Ultralytics 原始代码的 diff 保持尽可能小，所有多模态行为集中在独立扩展层。

`E:\repository\YOLOv11-RGBT` 仅用于验证融合方法的可行性：它证明了输入融合、分支融合、检测后融合和通过
batch 维折叠共享权重均可实现。本项目不复制该仓库的模块、YAML、损失或数据管线，也不以其 RGBT 两路、4C 输入
或 YOLO11 拓扑作为限制。

## 设计原则

1. **扩展而非分叉。** 默认 YOLO、默认数据集、`engine/`、原生 trainer/validator/predictor、损失和 exporter
   不知道多模态的存在。多模态入口通过子类和组合替换它们的窄接口。
2. **YAML 描述拓扑，扩展层描述语义。** 每个多模态模型 YAML 明确模态布局、阶段边界、融合点和融合算子；不在
   代码中根据 `yolov8`、`yolo11` 等文件名猜测层号。
3. **一处实现，所有生命周期复用。** 配对读取、通道堆叠、缓存、几何增强和推理文件映射属于同一数据扩展；train、
   val、predict 和 export 不得各自实现一次。
4. **先复用原生契约。** 能以标准 `Detect`/`Segment` 和原生损失表达的策略不新增 head 或 loss；需要多头、raw-logit
   融合或后处理的策略必须显式声明其训练、验证和导出契约。
5. **最小上游接触面。** 禁止复制 `parse_model`、trainer 或 predictor。若 stock YAML 解析器无法解析一个扩展模块，
   只允许增加一个通用、可上游化的模块注册 hook；不得把模态、融合模式或版本分支硬编码进原始解析器。

## 数据与输入契约

一个样本由主模态路径锚定，其他模态保留其相对于各自根目录的相对路径。数据 YAML 声明全部模态；模型 YAML 的
`channels` 必须等于全部模态通道和。例如 RGB(3)+thermal(1)+depth(1)：

```yaml
path: /datasets/rgbtd
train: images/rgb/train
val: images/rgb/val
channels: 5
names: { 0: person }
modalities:
  - { name: rgb, path: images/rgb, channels: 3, color: bgr }
  - { name: thermal, path: images/thermal, channels: 1, suffix: .png }
  - { name: depth, path: images/depth, channels: 1, suffix: .png }
```

- 模态数为 `M >= 2`，每路通道数为正整数；输入是按声明顺序堆叠的 `(B, C_total, H, W)`。
- 所有原始模态必须像素对齐、同宽高。加载器只能对完整样本执行同步 resize，不承担相机标定、重投影或时序同步。
- 当前图像预处理以 `uint8` 为契约；16-bit、float 或科学传感器格式必须在数据准备或明确的 adapter 中转换，不能由
  通用 loader 暗中缩放。
- Mosaic、MixUp、CopyPaste、透视、letterbox 和翻转必须在完整 HWC 通道栈上仅执行一次。外观增强只作用于显式
  声明为 BGR 的三通道模态；不得把热、深度或多光谱通道当成 RGB。
- 缓存保存完整融合样本，且缓存键包含布局和总通道数，避免命中单模态缓存。

## 扩展层边界

多模态代码应位于以下独立位置；其中当前已存在的文件可继续演进，但不可将逻辑回流到原生实现：

```text
ultralytics/
├── data/multimodal.py                 # 配对、堆叠、缓存与同步增强的 Dataset
├── models/multimodal/
│   ├── model.py                        # MultiModalYOLO 与 task_map 覆盖
│   ├── tasks.py                        # 多通道模型初始化与阶段校验
│   ├── train.py / val.py / predict.py  # 仅重写数据集/输入边界
│   ├── graph.py                        # 阶段清单、YAML 校验和模板构造
│   └── fusion.py                       # 算子注册、adapter 与共享权重路由
│   └── modules.py                       # ModalSplit、融合叶子模块、fold/unfold
├── cfg/models/multimodal/              # 新增的多模态模型模板，不改原始模型 YAML
└── tests/test_multimodal.py             # 独立回归矩阵
```

允许的原始代码改动白名单应尽量收敛为一个通用的“外部模块名 → `nn.Module`”注册 hook，使标准 YAML 解析器能够找到
`ModalSplit` 和统一的融合入口。该 hook 不得了解 RGB、IR、模态数、融合模式或自定义算子名称。若上游提供正式的
插件/模块注册机制，应删除本地 shim 并直接使用上游接口。除该 hook 和必要的懒导出外，不修改 `nn/tasks.py`、
`engine/`、`models/yolo/`、`data/augment.py`、默认 YAML 或默认 CLI 行为。

`MultiModalYOLO` 只替换 detect 和 instance-segmentation 的 dataset-facing 组件；pose、OBB、分类或其他任务必须
在各自的 trainer、validator、predictor、指标和导出测试全部接通后才能宣称支持。

## 阶段模型与版本约定

多模态图被统一描述为：

```text
input → encoder → nape → backbone outputs → neck → head features → Detect / Segment
```

`nape` 是本设计中 **backbone 尾部** 的语义提炼阶段，与 head 内 PAN/FPN 的 `neck` 不同。模板的初始约定如下，
但每个 YAML 必须显式保存自己的边界和 P3/P4/P5 等输出锚点：

| 模型系列               | encoder / nape 默认切分                    |
| ---------------------- | ------------------------------------------ |
| YOLOv8、YOLOv9、YOLO26 | backbone 最后一层为 nape，其余为 encoder。 |
| YOLOv10、YOLO11        | backbone 最后两层为 nape，其余为 encoder。 |
| YOLO12                 | 不定义 nape，整个 backbone 都是 encoder。  |

P2/P6、检测/分割、或未来模型变体可改变实际层数和特征尺度，因此这张表只能指导模板作者，不能成为运行时版本分支。
`graph.py` 应在构建前验证配置中的阶段边界、模态分支数、各融合点 stride 和通道数。

## 六种训练期融合方式

| 代码  | 名称              | 拓扑                                                               | 训练基线                                       |
| ----- | ----------------- | ------------------------------------------------------------------ | ---------------------------------------------- |
| `IF`  | `input_fusion`    | 所有模态通道直接堆叠后进入一条 encoder。                           | 使用原生 head/loss；不保留模态专用分支。       |
| `EF`  | `encoder_fusion`  | 各模态独立 encoder，逐尺度融合后进入共享 nape、neck 和预测头。     | 最早的特征级交互。                             |
| `NIF` | `nape_in_fusion`  | 各模态独立 encoder，融合模块插入 nape 的内部结点。                 | 适合注意力、门控、交叉调制；需要专用模块。     |
| `BF`  | `backbone_fusion` | 各模态完成 encoder+nape，在 backbone P3/P4/P5 等输出处逐尺度融合。 | 共享 neck 与一个预测头，是首个稳定多模态基线。 |
| `NF`  | `neck_fusion`     | 各模态独立完成 backbone+neck，在 Detect/Segment 前融合同尺度特征。 | 模态专用计算最深，成本最高。                   |
| `HF`  | `head_fusion`     | 各模态到达 raw head feature 或未 decode logits，再融合。           | 必须提供与融合输出匹配的预测投影和 loss。      |

`HF` 不等于 score fusion。score fusion 保留多个独立预测头，在 decode/NMS 后合并候选框、类别或置信度；它是独立的
部署期后处理策略，默认不可微、不能复用单头 loss，且需要单独的 metric 与 exporter 路径。参考仓库的 score fusion
只说明这一策略存在，不定义本项目的训练行为。

## 融合算子与自定义接口

第一阶段的融合算子只有：

- `concat`：按通道连接同尺度特征，后接显式 `1×1 Conv`/normalization 投影到下游期望通道数；
- `add`：只在每路特征经 adapter 对齐相同空间尺度和通道数后相加。

自定义融合模块统一由 `fusion.py` 注册，满足 `forward(features: list[Tensor]) -> Tensor` 或明确声明的多尺度变体。
注册项必须声明支持的模态数、输入/输出通道、所需 stride、是否跨尺度、是否可导出、是否需要额外 loss。`NIF` 通过
同一注册表在 nape 的指定内部结点调用，不得向通用 dataloader、训练循环或原生模型模块增加按模型名分支。

融合前，所有输入必须具有同一 batch、空间尺寸和语义 stride。框架不得自动插值、截断或广播；配置错误应在建模时
失败。基础 YAML 可只使用 stock `Concat`、`Conv` 和 adapter，因此未来增加一个复杂融合模块不应触碰原始解析器。

建议的模型 YAML 语义如下：

```yaml
channels: 5
multimodal:
  input_sections: [3, 1, 1]
  fusion: BF
  operator: concat
  fusion_points: [P3, P4, P5]
  share_weight: false
```

## `share_weight` 语义

`share_weight: true` 表示融合点之前由 `shared_stages` 指定的完整阶段复用**同一模块实例**，而不是各分支复制权重
后定期同步。若未指定 `shared_stages`，默认共享融合点之前的所有完整阶段，但每模态 input adapter 除外：

```yaml
multimodal:
  fusion: BF
  operator: concat
  share_weight: true
  shared_stages: [encoder, nape]
```

- 任意通道布局不能直接共享首层。每模态先经过独立 `input_adapter` 到共同通道数，再将
  `(B, M, C, H, W)` 折叠为 `(B×M, C, H, W)`，运行一次共享模块后按原顺序展开。这是对参考仓库固定两路
  `ChannelToNumber` / `NumberToChannel` 的泛化。
- YAML 必须以 `ModalFold` 和 `ModalUnfold: [M]` 显式包围共享阶段；`share_weight: true` 但图中缺少任意一个
  模块时，扩展层必须拒绝建模。框架不根据任意 YAML 的层号猜测 encoder/nape 范围；`shared_stages` 是语义声明，
  fold/unfold 才是实际共享边界。
- `IF` 不存在融合前的多分支，设置 `share_weight: true` 必须校验报错，不能静默忽略。
- 共享范围不能跨过指定融合点，否则会改变 `EF`、`BF`、`NF` 或 `HF` 的语义。需要不同共享范围时使用
  `shared_stages`，而不是新增一组几乎重复的 YAML。
- 共享 BatchNorm 在 `B×M` 上更新统计量，意味着混合模态分布。这是明确的实验选择；若需要模态独立统计量，应
  使用独立归一化层或冻结统计量。

## 训练、部署与验证

- `IF`、`EF`、`NIF`、`BF`、`NF` 在融合后接一个标准 `Detect`/`Segment` 时复用现有 Ultralytics loss、指标和
  exporter。`HF` 和 score fusion 必须先定义各自的 loss/后处理/导出契约。
- 实例分割的特征级融合仍使用一个标准 prototype 和一组 mask coefficient，除非某个实验明确增加自定义 head、
  双 prototype 和对应的损失；这些不属于基础多模态设施。
- 任何新策略至少通过：模型构建、`(B, C_total, H, W)` 前向、各输出 stride、concat/add 形状错误、单 batch loss、
  standalone val、文件 predict、缓存、导出和加载回归。score fusion 额外验证 NMS 合并、类别冲突和每模态预测
  可追溯性。
- 每次合入 Ultralytics 更新后，先运行原生相关任务的测试，再运行独立多模态矩阵。审查时检查
  `git diff <upstream-base> -- ultralytics/`：除扩展目录和通用注册 hook 外的改动都应被视为回归风险。

## 当前实现与后续顺序

当前扩展已实现任意 `M`/正整数通道的配对输入、通用 YAML 模块注册、`ModalSplit`、`MultiModalFusion` 的
`concat`/`add`、融合元数据校验和 `ModalFold`/`ModalUnfold` 的真实共享阶段。基础模板包括
`yolo11-mm3-bf-seg.yaml`（BF）和 `yolov8x-mm3-pre-sppf.yaml`（EF，P5 在 SPPF 前融合）。它们均复用标准单头
Detect/Segment 路径。

下一步依次为：

1. 为 IF、NF 增加覆盖检测/分割的模板和完整训练、验证、导出回归；
2. 通过融合注册表增加 NIF 自定义模块及其独立导出声明；
3. 为 HF 定义 raw feature/logit 投影和 loss 契约；
4. 最后单独实现 score fusion 的后处理、指标和 exporter，不能将其伪装成普通可训练 YAML 模式。

这份设计将参考实现的融合思想转化为本仓库的独立、可测试、可上游追踪的扩展架构，而不是迁移任何特定 YOLOv9 或
YOLO11-RGBT 网络。
