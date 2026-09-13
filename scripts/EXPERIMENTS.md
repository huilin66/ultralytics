# 实验脚本说明

这些脚本只覆盖检测、属性头和真正的多标签分类，不修改或调用
`segmentation` 训练代码。默认的 mdet 训练协议来自 `mayolo_r1.py`：

- stage 1：全模型训练，默认 100 epochs；
- stage 2：加载 stage 1 的 `best.pt`，冻结检测部分，只微调属性头，默认 100 epochs；
- 后续正式实验统一固定为 `100+100`；只有 E0 的 epoch 敏感性实验使用不同 epoch；
- 论文中的 `w4` 在当前实现中对应训练参数 `mdet`，脚本中的 `--w4` 会自动映射为 `mdet`；
- 检测主指标使用 `mAP50`，`mAP50:95` 可以保留为补充结果。

先在 `mdet` 环境中进入仓库根目录，并把示例中的数据集、权重和矩阵路径换成实际路径。
下面命令中的反引号 `` ` `` 是 PowerShell 换行符；在 Linux/bash 中请使用反斜杠
``\``，并通过 `python`/`python3` 执行，不能使用 `bash` 执行 `.py` 文件。

Linux/bash 的 E1 写法：

```bash
python scripts/train_mdet_experiments.py w4 \
  --data path/to/billboard_mdet.yaml \
  --model ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml \
  --pretrain yolov10x.pt \
  --w4-values 0.25 0.5 1.0 \
  --project runs/experiments/E1_w4
```

## E0：训练时长与颜色增强选择（优先完成）

这些独立实验必须使用“独立配置的独立训练”，不能从一次 500 epochs 的日志中截取
第 100/200/300/400/500 epoch 代替独立实验。总 epoch 数会影响学习率和增强策略，
因此截取到的中间 checkpoint 不等价于单独配置的短训练。

### E0.1：Stage1-only epoch 敏感性

下面命令会分别启动 100、200、300、400、500 epochs 的 Stage1-only 训练。每组都从
同一个预训练检测器开始，不进入 Stage2；结果会写入不同的 run 目录，并追加到
`experiment_manifest.jsonl`，其中 `stage2_epochs` 为 0。

```bash
python scripts/train_mdet_experiments.py stage1-sweep \
  --data path/to/billboard_mdet.yaml \
  --model ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml \
  --pretrain yolov10x.pt \
  --stage1-values 100 200 300 400 500 \
  --w4 0.5 \
  --project runs/experiments/E0_stage1_sweep
```

如果只想先试跑参数链路，可加 `--dry-run`；`--dry-run` 不会启动 GPU 训练。

### E0.2：HSV 颜色增强消融

锈蚀和褪色属性依赖颜色与亮度信息，因此补充三组固定 Stage1=100 epochs 的独立实验：

- `current`：当前参数 `hsv_h=0.015, hsv_s=0.7, hsv_v=0.4`；
- `disabled`：关闭 HSV，三个参数均为 0；
- `reduced`：弱 HSV，`hsv_h=0, hsv_s=0.2, hsv_v=0.2`。

```bash
python scripts/train_mdet_experiments.py hsv-ablation \
  --data ultralytics/cfg/mayolo_r1/mayolo_v3.yaml \
  --model ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml \
  --pretrain yolov10x.pt \
  --epochs 100 \
  --w4 0.5 \
  --project runs/experiments/E0_hsv_ablation
```

每组都会从同一预训练权重独立开始，结果写入不同目录，并在 manifest 中记录实际 HSV 参数。

### E0.3：固定 Stage1 checkpoint 的 Stage2-only epoch 敏感性

根据 E0.1 的结果，后续固定 Stage1 为 `N*=100`，再把该组的 `best.pt` 作为唯一
初始化权重，独立运行 Stage2 的 50、100、150、200 epochs：

```bash
python scripts/train_mdet_experiments.py stage2-sweep \
  --data path/to/billboard_mdet.yaml \
  --model ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml \
  --stage1-checkpoint runs/experiments/E0_stage1_sweep/E0_stage1_stage1_100_w4_0p5_seed_0/weights/best.pt \
  --stage1-epochs 100 \
  --stage2-values 50 100 150 200 \
  --w4 0.5 \
  --project runs/experiments/E0_stage2_sweep
```

上面的 checkpoint 路径仅是命名示例，需要替换为远程服务器上实际生成的路径。
四组 Stage2 都从同一个 `N*` checkpoint 开始，且每组独立配置自己的 Stage2 总
epoch 数；不能用 Stage2=200 的第 50/100 epoch 代替 Stage2=50/100 的独立训练。

最终至少保留以下对照：Stage1-only 的最佳时长、固定该 Stage1 checkpoint 后的最佳
Stage2 时长，以及同一 `N*+K*` 设置下的完整 Stage1+Stage2 结果。所有选择只看
验证集，测试集留到最终模型确定后再运行。

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
每个变体都必须提供真实 YAML；E2.1 的完整 12 个变体已经写入 `run.sh`。

例如 E2.1：

```powershell
python scripts/train_mdet_experiments.py gia-position `
  --label E2_1_GIA_position `
  --data path/to/billboard_mdet.yaml `
  --pretrain yolov10x.pt `
  --variant gia5=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_5.yaml `
  --variant gia7=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_7.yaml `
  --variant gia8=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_8.yaml `
  --variant gia9=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_9.yaml `
  --variant gia10=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_10.yaml `
  --variant gia5_7=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_5_7.yaml `
  --project runs/experiments/E2_1_GIA_position
```

每个普通 GIA 位置都有对应的 `_Res.yaml` 版本。E2.1 不重复训练 baseline；训练轮数、`w4` 和 HSV 参数使用脚本默认值（当前为 100+100、0.5、0/0.2/0.2）。

E2.2 和 E2.3 只需替换子命令、`--label` 和 `--variant`：

```powershell
python scripts/train_mdet_experiments.py gca-structure `
  --label E2_2_GCA_structure `
  --data path/to/billboard_mdet.yaml --pretrain yolov10x.pt `
  --variant baseline=ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml `
  --variant gat=ultralytics/cfg/models/exp_ablation/yolov10x_GCA.yaml `
  --w4 0.5 --com-path path/to/co_occurrence_matrix_train.csv `
  --project runs/experiments/E2_2_GCA

python scripts/train_mdet_experiments.py gia-gca `
  --label E2_3_GIA_GCA `
  --data path/to/billboard_mdet.yaml --pretrain yolov10x.pt `
  --variant baseline=ultralytics/cfg/models/experiments/yolov10x-mdetect.yaml `
  --variant gia_gca=ultralytics/cfg/models/exp_ablation/yolov10x_GIA_GCA.yaml `
  --w4 0.5 --com-path path/to/co_occurrence_matrix_train.csv `
  --project runs/experiments/E2_3_GIA_GCA
```

`exp_ablation` 中的 GCA YAML 含有 Linux 下的 `/nfsv4/...` 矩阵路径。
传入 `--com-path` 后，脚本只在 `project/_generated_configs/` 生成替换后的副本，
不会改写原 YAML；不传时会主动报错，避免训练读到错误矩阵。

### E2.2：基于固定 Stage1 的 GCA/GNN 结构比较

审稿人要求比较不同图模型时，使用下面的 `gca-stage2`。它不会为每个 GNN
重新训练 Stage1，而是统一加载已经确定的 baseline checkpoint：

`runs/experiments/E1_w4/E1_w4_base_w4_0p5_seed_0_stage1/weights/best.pt`。

门控 `com_gat`、门控固定矩阵 `com`、标准 GCN、学习型 GAT、GraphSAGE 和 GIN
都从这个 checkpoint 开始，并使用相同的 Stage2=100、w4=0.5
和随机种子。新增 GNN 变体使用与 GIA-v2 相同的逐通道零初始化残差门控，确保
stage2 初始状态等价于 baseline。所有含图结构的变体读取仅由 train split 生成的
`co_occurrence_matrix_train.csv`；`--com-path` 会把 YAML 中的 Linux 路径替换为
当前机器上的实际路径。

```bash
python scripts/train_mdet_experiments.py gca-stage2 \
  --label E2_2_GCA_stage2_residual \
  --data ultralytics/cfg/mayolo_r1/mayolo_v3.yaml \
  --stage1-checkpoint runs/experiments/E1_w4/E1_w4_base_w4_0p5_seed_0_stage1/weights/best.pt \
  --stage1-epochs 100 \
  --stage2-epochs 100 \
  --variant gca_current_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_residual.yaml \
  --variant gca_com_residual=ultralytics/cfg/models/exp_ablation/yolov10x_GCA_com_residual.yaml \
  --variant gcn=ultralytics/cfg/models/exp_ablation/yolov10x_GCN.yaml \
  --variant gat_learned=ultralytics/cfg/models/exp_ablation/yolov10x_GAT_learned.yaml \
  --variant graphsage=ultralytics/cfg/models/exp_ablation/yolov10x_GraphSAGE.yaml \
  --variant gin=ultralytics/cfg/models/exp_ablation/yolov10x_GIN.yaml \
  --w4 0.5 \
  --batch 16 \
  --seed 0 \
  --com-path /path/to/co_occurrence_matrix_train.csv \
  --project runs/experiments/E2_2_GCA_stage2_residual
```

`gca-structure` 保留用于历史的完整两阶段结构实验；本节的 `gca-stage2` 才是
针对 reviewer 要求、控制 Stage1 初始化一致的 GNN 比较入口。`gca_current_residual` 使用
当前仓库已有的 `com_gat` 的门控版本，而 `gca_com_residual` 使用固定共现矩阵的
门控版本，便于区分历史实现与残差稳定化后的实现。

如果六个旧的逐 logit 残差 GCA/GNN 变体均未改变 hard argmax 指标，使用固定的 E1
Stage1 checkpoint 运行新的 multiclass-aware 比较。此前的五个 Stage2=100 GNN 结果
已经完成，保留在 `E2_2_GCA_GNN_margin_residual`，不会被当前 `run.sh` 重复训练。

上一轮 11 个 Stage2 特征图任务以及后续 gain=2/4/8、local gain=4 筛选已经完成。
所有 gain<=4 的 hard 指标都与 baseline 完全相同；gain=8 虽提高了
`F1_macro_global`，但验证集和测试集的 `F1_macro` 均下降。因此 feature graph
方向关闭，不再进入新的结构或矩阵实验。

远程汇总保存在 `runs/experiments/E2_2_feature_gain/summary.csv`，该结果作为
负结果归档。当前 `run.sh` 已加保护，默认不会启动 feature graph；仅在需要复现
历史结果时显式设置 `ENABLE_FEATURE_GRAPH=1`。

历史“5×5”的 context_cross/context_conditional 在固定矩阵下是同一结构，
实际只有 4 个结构。其验证集最高 `F1_macro` 约为 `0.68653`，相对基线
`0.68413` 有约 `+0.00240` 的微弱提升，但测试集没有同步提升；因此保留为
候选信号，不把它当作已经稳定验证的收益。历史重复结果不能视为独立证据，
旧命令已注释保留。

新模块从 cv4 分类器之前提取视觉特征，构造每属性 16 维节点，保留原分类器。
输出为 `z + feature_gain * gamma * delta_z`；gain 默认 1.0，gamma 初始 0.1，
输出层零初始化。训练脚本的 `--feature-gain` 只在生成 YAML 副本时写入第六个
参数，不修改源 YAML；manifest 同时记录该值。
连续共现边权与源属性正类概率共同控制传播。conditional CSV 行是条件源，
列是目标，因此加载时转置为目标行、源列。所有算子都是本项目的加权变体。

历史实验固定 Stage1 checkpoint、Stage2=100、batch=16、seed=0、w4=0.5；
保持 AdamW 学习率 1e-4，不混入学习率消融。检测部分冻结，两个属性分支均训练。
根据验证集 `F1_macro` 和逐属性 F1 选择候选，test 只用于最终报告。

scripts/check_feature_graph.py 检查初始等价、学习梯度、矩阵影响和两分支接入。
可设置 AttributeFeatureGraph.enabled=False 关闭整个修正分支来评估同一权重；
该操作同时关闭视觉适配，因此需要结合 local 模型解释图的贡献。

### E2.6：固定共现先验 head-only 结构

这组实验承接 5×5 的微弱验证集信号，但不再增加属性节点 GNN，也不解冻
backbone/neck。新增模块都位于标准 `cv4` 属性分支，最后通过零初始化残差接入，
所以 Stage2 初始预测与 baseline 对齐：

1. `prior_bias`：对二分类 margin 做不确定性门控的先验偏置，作为最低成本控制；
2. `prior_channel`：用先验支持和视觉全局描述生成 channel attention；
3. `prior_spatial`：用视觉能量、先验支持和不确定性生成 7×7 spatial attention；
4. `prior_moe`：在 local expert 与 prior-conditioned expert 之间自适应混合；
5. `prior_texture`：复用 `TextureAttention`，但只在先验支持的空间门控下写回。

使用统一 Stage1 checkpoint 的 Stage2 批量入口如下。默认跑 5 个 cross 结构；若夜间
预算允许，增加 `--matrix-modes cross conditional` 即得到 10 个独立任务：

```bash
python scripts/train_mdet_experiments.py prior-stage2 \
  --label E2_6_prior_head \
  --model ultralytics/cfg/models/exp_ablation/yolov10x_com_prior.yaml \
  --data ultralytics/cfg/mayolo_r1/mayolo_v3.yaml \
  --stage1-checkpoint runs/experiments/E1_w4/E1_w4_base_w4_0p5_seed_0_stage1/weights/best.pt \
  --stage1-epochs 100 --stage2-epochs 100 \
  --prior-types bias channel spatial moe texture \
  --matrix-modes cross conditional \
  --w4 0.5 --batch 16 --seed 0 \
  --com-path /path/to/co_occurrence_matrix_train.csv \
  --project runs/experiments/E2_6_prior_head
```

若先做快速筛选，将 `--stage2-epochs` 改为 50，并只保留 `cross`；最终候选再用
100 epochs 和 seed 1/2 复核。选择标准仍是验证集 `F1_macro`，同时检查每个属性的
F1、precision、recall；`F1_macro_global` 单独上涨不能作为接受条件。YOLO/RT-DETR/
YOLO26 整网更换和局部解冻留到这轮之后，不与本轮混杂。

### E2.7：直接 logit/margin prior 结构

E2.6 的 10 个结构虽然 head 参数发生更新，但属性混淆矩阵完全不变，说明
feature-level residual 的写入幅度不足以改变 hard prediction。E2.7 保持相同的
Stage1 checkpoint、冻结策略、`w4=0.5`、batch 和 seed，只把先验融合位置改为二分类
logit/margin：

1. `logit_blend`：不确定性门控的视觉 margin 与 prior log-odds 插值；
2. `logit_bias`：不确定性门控的 prior log-odds 加性偏置；
3. `logit_mlp`：由视觉 margin、prior log-odds、support 和 uncertainty 生成非线性修正；
4. `cross_attention`：根据源属性视觉 margin 动态重加权共现边，再进行 logit 融合；
5. `dynamic_gate`：用逐像素 gate 决定视觉 margin 与 prior margin 的融合比例。

默认批量命令由仓库根目录的 `run.sh` 提供，输出到
`runs/experiments/E2_7_prior_logit`。两种矩阵解释各跑一次，共 10 个 Stage2-only
任务；backbone、neck 和检测分支不解冻。除 hard F1 外，后续应补充属性概率/校准指标，
因为单看 argmax 可能漏掉连续分数的改善。

### E2.8：鲁棒先验融合结构

E2.7 的验证集提升没有泛化到测试集，因此 E2.8 固定相同的 Stage1 checkpoint、
冻结策略、`w4=0.5`、batch 和 seed，改测五种更保守的属性 head：

1. `logit_diffusion`：在 signed margin 空间做固定图扩散，并限制残差；
2. `confidence_blend`：按源属性视觉置信度重加权共现支持，降低不确定源节点的污染；
3. `agreement_temperature`：只根据先验与视觉预测的一致性调节 margin 温度，不直接翻转类别；
4. `stochastic_blend`：训练时随机丢弃先验边，测试时恢复完整矩阵，用于先验正则化；
5. `lowrank_attention`：在固定共现边上叠加低秩 target/source attention，限制可学习自由度。

仓库根目录的 `run.sh` 默认输出到 `runs/experiments/E2_8_prior_robust`，两种矩阵解释
各跑一次，共 10 个 Stage2-only 任务；backbone、neck 和检测分支不解冻。

### E2.9：ML-GCN 风格 label-correlation classifier

参考 ML-GCN 的 label-node 设计，E2.9 不把共现矩阵直接当作 logits bias，而是在属性
分支中维护可学习的 label embedding，通过图传播生成 label-aware spatial classifier，
最后以小幅 residual 写回视觉 margin。固定 E2.8 的 Stage1 checkpoint、冻结策略、
`w4=0.5`、batch 和 seed，比较以下五种结构：

1. `label_gcn`：两层固定有向共现图 GCN；
2. `label_gcn_threshold`：阈值化共现边并保留 self-loop；
3. `adaptive_label_gcn`：固定先验图与低秩可学习图融合；
4. `dynamic_label_gcn`：固定先验图与图像条件动态图融合；
5. `label_attention`：label-aware spatial attention 后再做先验消息传播。

仓库根目录的 `run.sh` 默认输出到 `runs/experiments/E2_9_label_gcn`，两种矩阵解释
各跑一次，共 10 个 Stage2-only 任务；backbone、neck 和检测分支不解冻。

## E2.4–E2.5：HO

E2.4 不需要重新写 loss 或训练流程。用下面的 `ho` 命令完成一次 100+100
训练，然后分别评估 checkpoint 的原生路径和 one-to-many 路径：

```powershell
python scripts/train_mdet_experiments.py ho `
  --label E2_4_HO `
  --data path/to/billboard_mdet.yaml --pretrain yolov10x.pt `
  --variant ho_gca=ultralytics/cfg/models/exp_ablation/yolov10x_HO_GCA.yaml `
  --w4 0.5 --com-path path/to/co_occurrence_matrix_train.csv `
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

## E3：YOLOv8–YOLOv13、YOLO26 与 MAYOLO 多规模

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
  --variant yolov26x=ultralytics/cfg/models/experiments/yolov26x-mdetect.yaml `
  --variant mayolox=ultralytics/cfg/models/mayolo/mayolovx.yaml `
  --pretrain-map yolov8n=yolov8n.pt `
  --pretrain-map yolov10x=yolov10x.pt `
  --pretrain-map yolov11x=path/to/yolo11x.pt `
  --pretrain-map yolov12x=path/to/yolo12x.pt `
  --pretrain-map yolov13x=path/to/yolo13x.pt `
  --pretrain-map yolov26x=yolo26x.pt `
  --pretrain-map mayolox=path/to/mayolovx.pt `
  --w4 0.5 --project runs/experiments/E3_versions
```

当前仓库未必包含每个版本、每个 size 的 YAML 和 `.pt` 权重；脚本不会伪造缺失
配置，按实际存在的文件增删 `--variant` 即可。

正式开始 E3 全量训练前，建议先用最大版本做 GPU memory smoke test。该脚本只
测试 mdet，不调用 segmentation；默认覆盖 YOLOv8x、YOLOv9e、YOLOv10x、
YOLOv11x、YOLOv12x、YOLOv13x、YOLO26x、MAYOLOx 和 RT-DETR-L，每个模型真实
训练 2 epochs，并将峰值显存写入 `gpu_memory_smoke_summary.csv`。建议先用
`batch=16`，确认所有模型都能跑通后，再据峰值为正式训练统一确定 batch size：

```bash
python scripts/gpu_memory_smoke_test.py \
  --data path/to/billboard_mdet.yaml \
  --device 0 --imgsz 640 --batch 16 --epochs 2 \
  --project runs/gpu_memory_smoke
```

如果预训练权重不在默认搜索路径，用重复的
`--pretrain-map NAME=CHECKPOINT` 覆盖，例如：

```bash
python scripts/gpu_memory_smoke_test.py \
  --data path/to/billboard_mdet.yaml \
  --pretrain-map yolov13x=/path/to/yolov13x.pt \
  --pretrain-map mayolox=/path/to/mayolovx.pt
```

脚本会在单个模型 OOM 或权重缺失时记录状态并继续后续模型；`max_memory_allocated`
和 `max_memory_reserved` 均按每个模型单独清空缓存、重置峰值后统计。

如果只做 YOLO26 的完整多规模实验，可以由脚本自动加入 n/s/m/l/x 配置和对应的
`yolo26n.pt`–`yolo26x.pt` 预训练权重：

```bash
python scripts/train_mdet_experiments.py versions \
  --data path/to/billboard_mdet.yaml \
  --include-yolo26 --yolo26-sizes n s m l x \
  --w4 0.5 --project runs/experiments/E3_yolo26
```

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
