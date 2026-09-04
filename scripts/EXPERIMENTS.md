# 实验脚本说明

这些脚本只覆盖检测、属性头和真正的多标签分类，不修改或调用
`segmentation` 训练代码。默认的 mdet 训练协议来自 `mayolo_r1.py`：

- stage 1：全模型训练，默认 100 epochs；
- stage 2：加载 stage 1 的 `best.pt`，冻结检测部分，只微调属性头，默认 100 epochs；
- 论文中的 `w4` 在当前实现中对应训练参数 `mdet`，脚本中的 `--w4` 会自动映射为 `mdet`；
- 检测主指标使用 `mAP50`，`mAP50:95` 可以保留为补充结果。

先在 `mdet` 环境中进入仓库根目录，并把示例中的数据集、权重和矩阵路径换成实际路径。

## E1：w4 敏感性

默认直接运行 `0.25、0.5、1.0` 三组：

```powershell
python scripts/train_mdet_experiments.py w4 `
  --data path/to/billboard_mdet.yaml `
  --model ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml `
  --pretrain yolov10x.pt `
  --w4-values 0.25 0.5 1.0 `
  --project runs/experiments/E1_w4 `
  --stage1-epochs 100 --stage2-epochs 100
```

提交正式训练前可先检查运行矩阵：

```powershell
python scripts/train_mdet_experiments.py w4 `
  --data path/to/billboard_mdet.yaml `
  --model ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml `
  --pretrain yolov10x.pt --dry-run
```

## E2.1–E2.3：GIA/GCA 消融

三个命令的参数形式相同，使用重复的 `--variant NAME=CONFIG_YAML` 指定变体。
这样不会假定某个尚未提交的 GIA 位置或 GCA 结构配置；每个变体都必须提供真实 YAML。

例如 E2.1：

```powershell
python scripts/train_mdet_experiments.py gia-position `
  --label E2_1_GIA_position `
  --data path/to/billboard_mdet.yaml `
  --pretrain yolov10x.pt `
  --variant baseline=ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml `
  --variant gia_p3=path/to/yolov10x_gia_p3.yaml `
  --variant gia_neck=path/to/yolov10x_gia_neck.yaml `
  --w4 0.5 `
  --project runs/experiments/E2_1_GIA
```

E2.2 和 E2.3 只需替换子命令、`--label` 和 `--variant`：

```powershell
python scripts/train_mdet_experiments.py gca-structure `
  --label E2_2_GCA_structure `
  --data path/to/billboard_mdet.yaml --pretrain yolov10x.pt `
  --variant baseline=ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml `
  --variant gat=ultralytics/cfg/models/exp_ablation0107/yolov10x_GCA.yaml `
  --w4 0.5 --com-path path/to/co_occurrence_matrix6.csv `
  --project runs/experiments/E2_2_GCA

python scripts/train_mdet_experiments.py gia-gca `
  --label E2_3_GIA_GCA `
  --data path/to/billboard_mdet.yaml --pretrain yolov10x.pt `
  --variant baseline=ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml `
  --variant gia_gca=ultralytics/cfg/models/exp_ablation0107/yolov10x_GIA_GCA.yaml `
  --w4 0.5 --com-path path/to/co_occurrence_matrix6.csv `
  --project runs/experiments/E2_3_GIA_GCA
```

`exp_ablation0107` 中的 GCA YAML 含有 Linux 下的 `/nfsv4/...` 矩阵路径。
传入 `--com-path` 后，脚本只在 `project/_generated_configs/` 生成替换后的副本，
不会改写原 YAML；不传时会主动报错，避免训练读到错误矩阵。

## E2.4–E2.5：HO

E2.4 不需要重新写 loss 或训练流程。用下面的 `ho` 命令完成一次 100+100
训练，然后分别评估 checkpoint 的原生路径和 one-to-many 路径：

```powershell
python scripts/train_mdet_experiments.py ho `
  --label E2_4_HO `
  --data path/to/billboard_mdet.yaml --pretrain yolov10x.pt `
  --variant ho_gca=ultralytics/cfg/models/exp_ablation0107/yolov10x_HO_GCA.yaml `
  --w4 0.5 --com-path path/to/co_occurrence_matrix6.csv `
  --project runs/experiments/E2_4_HO

python scripts/eval_mdet_experiments.py ho `
  --weights runs/experiments/E2_4_HO/E2_4_HO_ho_gca_w4_0p5_seed_0_stage2/weights/best.pt `
  --data path/to/billboard_mdet.yaml `
  --mode both `
  --project runs/experiments/E2_4_HO_eval
```

`native` 保留 checkpoint 的默认 head 选择，`one2many` 显式调用
`use_one2many_head()`。脚本会为两种模式重新加载权重，避免前一次验证切换
head 后影响后一次结果。E2.5 直接复用选定的最佳 GIA+GCA+HO checkpoint，
不需要再人为增加一组训练。

## E3：YOLOv8–YOLOv13 与 MAYOLO 多规模

使用 `versions`，每个变体一个 `--variant`。不同版本通常需要不同预训练权重，
用 `--pretrain-map NAME=CHECKPOINT` 绑定：

```powershell
python scripts/train_mdet_experiments.py versions `
  --label E3_versions `
  --data path/to/billboard_mdet.yaml `
  --variant yolov8n=ultralytics/cfg/models/experiments/yolov8n-mdetect.yaml `
  --variant yolov10x=ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml `
  --variant yolov11x=ultralytics/cfg/models/experiments/yolov11x-mdetect.yaml `
  --variant yolov12x=ultralytics/cfg/models/experiments/yolov12x-mdetect.yaml `
  --variant yolov13x=ultralytics/cfg/models/experiments/yolov13x-mdetect.yaml `
  --variant mayolox=ultralytics/cfg/models/mayolo/mayolovx.yaml `
  --pretrain-map yolov8n=yolov8n.pt `
  --pretrain-map yolov10x=yolov10x.pt `
  --pretrain-map yolov11x=path/to/yolo11x.pt `
  --pretrain-map yolov12x=path/to/yolo12x.pt `
  --pretrain-map yolov13x=path/to/yolo13x.pt `
  --pretrain-map mayolox=path/to/mayolovx.pt `
  --w4 0.5 --project runs/experiments/E3_versions
```

当前仓库未必包含每个版本、每个 size 的 YAML 和 `.pt` 权重；脚本不会伪造缺失
配置，按实际存在的文件增删 `--variant` 即可。

## E4：RT-DETR 属性检测多规模

使用 `rtdetr` 子命令。它会调用 `RTDETR` 包装器和 `myolo_r1.py` 中的 RT-DETR
属性头冻结逻辑：

```powershell
python scripts/train_mdet_experiments.py rtdetr `
  --label E4_rtdetr `
  --data path/to/billboard_mdet.yaml `
  --variant rtdetr_l=ultralytics/cfg/models/rt-detr/rtdetr-l-md.yaml `
  --pretrain-map rtdetr_l=path/to/rtdetr-l.pt `
  --w4 0.5 --project runs/experiments/E4_rtdetr
```

如果新增 `rtdetr-x-md.yaml` 等属性头 YAML，直接继续添加 `--variant` 和对应
的 `--pretrain-map`。

## E5：真正的多标签目标检测 YOLOv10

这不是 `mdetect`。使用现有的 `scripts/train_multilabel.py`，一个物理框对应
一个 n-hot 标签向量：

```powershell
python scripts/train_multilabel.py `
  --model path/to/yolov10x.pt `
  --data path/to/data_multilabel.yaml `
  --epochs 100 --imgsz 640 --batch 16 `
  --project runs/experiments/E5_multilabel --name yolov10x
```

## E6：目标检测 + 多标签分类双阶段

先从 E1/E3/E7 得到检测器 checkpoint，再对检测器裁剪结果训练分类器：

```powershell
python scripts/train_two_stage.py `
  --detector-checkpoint runs/experiments/E3_versions/.../weights/best.pt `
  --model ultralytics/cfg/models/11/yolo11n-cls.yaml `
  --data path/to/detection_crops_multilabel.yaml `
  --epochs 100 --imgsz 224 --batch 16 `
  --project runs/experiments/E6_two_stage --name yolov10x_yolo11n
```

分类数据 YAML 的 `train/val/test` 应指向裁剪图目录，`labels` 指向 sidecar 标签
目录；每个同名 `.txt` 文件只包含该 crop 的类别 ID，例如 `0,3`。正式结果应
使用预测框生成的 crop 评估，不能只报告 GT 框裁剪的结果。脚本会将 detector
checkpoint 写入 `two_stage_manifest.json`，便于追溯来源。

## E7：稳定性

下面示例对 YOLOv10x 和 MAYOLOx 各运行三个 seed（默认 0、1、2），每次仍是
100+100：

```powershell
python scripts/train_mdet_experiments.py stability `
  --data path/to/billboard_mdet.yaml `
  --variant yolov10x=ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml `
  --variant mayolox=ultralytics/cfg/models/mayolo/mayolovx.yaml `
  --pretrain-map yolov10x=yolov10x.pt `
  --pretrain-map mayolox=path/to/mayolovx.pt `
  --seeds 0 1 2 --w4 0.5 `
  --project runs/experiments/E7_stability
```

脚本会把每次运行的配置追加到 `experiment_manifest.jsonl`，可据此汇总均值、
标准差和置信区间。

## E8：鲁棒性

E8 是推理/评估实验，不需要训练新模型：固定同一个最佳 checkpoint，分别构造
不同亮度和不同场景的测试集，然后使用现有 mdet `val`/`predict` 接口评估。不要
把亮度或场景变换混入训练集，否则就不再是独立鲁棒性测试。当前新增脚本只负责
训练与 HO 对照，E8 的变换数据生成和统计仍应作为独立评估步骤维护。

## 输出与复现

每个 mdet 请求会在 `--project/experiment_manifest.jsonl` 写入：配置文件、权重、
data、w4→mdet 映射、seed、两个 stage 的 epoch 和 run name。正式训练前建议先
使用 `--dry-run` 检查路径及实验矩阵；脚本不会自动开始 GPU 训练，也不会自动
重试失败实验。
