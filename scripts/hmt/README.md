# HMT 数据处理与训练

所有命令从 Git Bash、仓库根目录执行。默认使用：

- 数据源：`//158.132.186.40/isds/huilin/tp/0803_hmt_data_check/bdd_hmt`
- 新数据：`//158.132.186.40/isds/huilin/tp/0803_hmt_data_check/bdd_hmt_update`
- 数据处理环境：`common_py312`
- 训练环境：`yolov8_deploy`

## 只处理数据

```bash
bash scripts/hmt/prepare_hmt.sh
```

脚本会调用 `yolo_data_manager` 做 check/stats；HMT 专用的类别重映射、分组切分和训练集裁剪由 `tools/hmt_prepare_dataset.py` 完成。默认生成：

```text
bdd_hmt_update/
├── sua_t_update/
├── sua_rgb_update/
└── bp_cube_update/
```

再次生成时需要显式允许重建仅由本流程产生的 `_update` 目录：

```bash
HMT_REBUILD=1 bash scripts/hmt/prepare_hmt.sh
```

## 训练

先完成数据处理，再一键训练三个模型：

```bash
bash scripts/hmt/run_hmt.sh
```

如果更新数据已经存在：

```bash
SKIP_PREPARE=1 bash scripts/hmt/run_hmt.sh
```

只训练单个数据集：

```bash
SKIP_PREPARE=1 HMT_DATASET=rgb bash scripts/hmt/train_hmt.sh
```

常用覆盖参数示例：

```bash
HMT_EPOCHS_RGB=300 HMT_IMGSZ_RGB=768 HMT_DEVICE=0 bash scripts/hmt/train_hmt.sh
```

训练时不再读取 `HMT_UPDATE_ROOT`；脚本从你修改后的 YAML `path` 找到清单，并在运行目录生成绝对路径清单。训练结果默认写入仓库下的 `runs/hmt_update_runs`，可用 `HMT_RUN_ROOT` 覆盖。

## 处理约定

- 原始标签中的 `background=0` 不作为检测类别，源类别 1..N 被压缩为标准 YOLO 0..N-1。
- RGB 从 `sua_rgb` 原始类别重建：Broken Low/High、Cracked Tile、Spalling 合并为 Broken；其他类别按名称映射。映射记录在每个输出目录的 `label_map.json`。
- SUA 按连续文件序列分组，cube 按去掉 `_L/_R/_B/_F` 后的场景名分组，避免同一序列/场景进入不同集合。
- RGB/cube 仅为训练正样本生成带重叠 tile；验证和测试使用原图。`train_balanced.txt` 只做训练清单采样/重复，不复制修改原始图像。
